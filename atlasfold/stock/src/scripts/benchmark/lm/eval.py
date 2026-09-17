import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

CONTACT_THRESHOLD = 8.0
TRAIN_SEQ_SEP = 6
EVAL_SEQ_SEP = 24
N_BOOTSTRAPS = 10
N_TRAIN = 20
MAX_LENGTH = 510
LOGREG_C = 0.15
LOGREG_MAX_ITER = 50
REQUIRE_L_CONTACTS = True

DEFAULT_ROOT_PATH = Path("data/esm_structural_split")
SPLIT_LEVELS = ("superfamily",)
CV_PARTITIONS = ("4",)


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-path", type=Path, default=DEFAULT_ROOT_PATH)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)


class ESMStructuralSplitDataset(torch.utils.data.Dataset):
    base_folder = "structural-data"

    def __init__(
        self,
        split_level: str,
        cv_partition: str,
        split: str,
        root_path: str | Path = DEFAULT_ROOT_PATH,
    ):
        if split not in {"train", "valid"}:
            raise ValueError(f"split must be 'train' or 'valid', got {split!r}")
        self.split_level = split_level
        self.cv_partition = cv_partition
        self.split = split
        self.root_path = Path(root_path)
        self.base_path = self.root_path / self.base_folder
        self.pkl_dir = self.base_path / "pkl"
        self.split_file = (
            self.base_path / "splits" / split_level / cv_partition / f"{split}.txt"
        )
        with open(self.split_file) as f:
            self.names = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        pkl_path = self.pkl_dir / name[1:3] / f"{name}.pkl"
        with open(pkl_path, "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed.*",
                    category=Warning,
                )
                record = dict(pickle.load(f))
        if len(record["seq"]) > MAX_LENGTH:
            record["seq"] = record["seq"][:MAX_LENGTH]
            record["ssp"] = record["ssp"][:MAX_LENGTH]
            record["coords"] = record["coords"][:MAX_LENGTH]
            record["dist"] = record["dist"][:MAX_LENGTH, :MAX_LENGTH]
        dist = record["dist"].astype(np.float32, copy=False)
        valid_pair_mask = np.isfinite(dist)
        record["name"] = name
        record["contact_map"] = dist < CONTACT_THRESHOLD
        record["valid_pair_mask"] = valid_pair_mask
        return record


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return (x + x.transpose(-1, -2)) / 2


def apc(x: torch.Tensor) -> torch.Tensor:
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum((-1, -2), keepdim=True)
    return x - (a1 * a2) / a12.clamp_min(torch.finfo(x.dtype).eps)


def attention_pair_features(
    attentions: torch.Tensor,
    length: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    n_layers: int,
    n_heads: int,
    token_offset: int = 1,
) -> np.ndarray:
    attns = attentions[
        :, :, token_offset : length + token_offset, token_offset : length + token_offset
    ]
    expected_shape = (n_layers, n_heads, length, length)
    if tuple(attns.shape) != expected_shape:
        raise ValueError(
            f"Unexpected attention shape {tuple(attns.shape)}, expected {expected_shape}"
        )
    attns = apc(symmetrize(attns.float()))
    i_idx_t = torch.as_tensor(i_idx, dtype=torch.long, device=attns.device)
    j_idx_t = torch.as_tensor(j_idx, dtype=torch.long, device=attns.device)
    features = attns[:, :, i_idx_t, j_idx_t]
    features = features.permute(2, 0, 1).reshape(len(i_idx), n_layers * n_heads)
    return features.cpu().numpy()


def attention_pair_scores(
    attentions: torch.Tensor,
    length: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    n_layers: int,
    n_heads: int,
    score_cache: dict,
    token_offset: int = 1,
) -> np.ndarray:
    attns = attentions[
        :, :, token_offset : length + token_offset, token_offset : length + token_offset
    ]
    expected_shape = (n_layers, n_heads, length, length)
    if tuple(attns.shape) != expected_shape:
        raise ValueError(
            f"Unexpected attention shape {tuple(attns.shape)}, expected {expected_shape}"
        )
    attns = apc(symmetrize(attns.float()))
    cache_key = (str(attns.device), attns.dtype)
    if cache_key not in score_cache["torch_weights"]:
        score_cache["torch_weights"][cache_key] = torch.as_tensor(
            score_cache["coef"].reshape(n_layers, n_heads),
            dtype=attns.dtype,
            device=attns.device,
        )
    weights = score_cache["torch_weights"][cache_key]
    score_map = (attns * weights[:, :, None, None]).sum(dim=(0, 1))
    score_map = score_map + score_cache["intercept"]
    i_idx_t = torch.as_tensor(i_idx, dtype=torch.long, device=score_map.device)
    j_idx_t = torch.as_tensor(j_idx, dtype=torch.long, device=score_map.device)
    return score_map[i_idx_t, j_idx_t].cpu().numpy()


def pair_indices(data: dict, min_seq_sep: int) -> tuple[np.ndarray, np.ndarray]:
    length = len(data["seq"])
    i_idx, j_idx = np.triu_indices(length, k=min_seq_sep)
    valid = data["valid_pair_mask"][i_idx, j_idx]
    return i_idx[valid], j_idx[valid]


def make_bootstrap_split(
    n_items: int,
    bootstrap_i: int,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_train >= n_items:
        raise ValueError(f"n_train={n_train} must be smaller than dataset size {n_items}")
    rng = np.random.default_rng(bootstrap_i)
    train_indices = np.sort(rng.choice(n_items, size=n_train, replace=False))
    train_mask = np.zeros((n_items,), dtype=bool)
    train_mask[train_indices] = True
    eval_indices = np.nonzero(~train_mask)[0]
    return train_indices, eval_indices


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, seed: int):
    logreg = LogisticRegression(
        C=LOGREG_C,
        l1_ratio=1.0,
        solver="liblinear",
        random_state=seed,
        max_iter=LOGREG_MAX_ITER,
    )
    logreg.fit(X, y)
    if logreg.n_iter_[0] >= logreg.max_iter:
        print(
            f"  Logistic regression did not converge within {LOGREG_MAX_ITER} iterations"
        )
    return logreg


def evaluate_contact_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    length: int,
) -> dict[str, float]:
    top_l_k = min(length, len(scores))
    top_l5_k = max(1, min(length // 5, len(scores)))
    top_l = np.argpartition(-scores, top_l_k - 1)[:top_l_k]
    top_l5 = np.argpartition(-scores, top_l5_k - 1)[:top_l5_k]
    return {
        "P@L": float(labels[top_l].mean()) if len(top_l) > 0 else float("nan"),
        "P@L/5": float(labels[top_l5].mean()) if len(top_l5) > 0 else float("nan"),
    }


def logistic_regression_scores(features: np.ndarray, logreg) -> np.ndarray:
    coef = logreg.coef_[0].astype(features.dtype, copy=False)
    scores = features @ coef
    scores += logreg.intercept_[0]
    return scores


def fit_bootstrap_model(
    extract_attention_features,
    dataset: ESMStructuralSplitDataset,
    train_indices: np.ndarray,
    bootstrap_i: int,
):
    X_parts = []
    y_parts = []
    for index in tqdm(train_indices, desc=f"Train bootstrap {bootstrap_i}", leave=False):
        data = dataset[int(index)]
        i_idx, j_idx = pair_indices(data, TRAIN_SEQ_SEP)
        if len(i_idx) == 0:
            continue
        features = extract_attention_features(data["seq"], i_idx, j_idx)
        labels = data["contact_map"][i_idx, j_idx].astype(np.int64)
        X_parts.append(features)
        y_parts.append(labels)
    if not X_parts:
        raise RuntimeError(f"Bootstrap {bootstrap_i} produced no training pairs")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Bootstrap {bootstrap_i} has only one contact class")
    return fit_logistic_regression(X, y, seed=bootstrap_i)


def evaluate_bootstrap_model(
    extract_attention_features,
    logreg,
    dataset: ESMStructuralSplitDataset,
    eval_indices: np.ndarray,
    extract_attention_scores=None,
) -> dict:
    results = {"P@L": [], "P@L/5": [], "names": [], "skipped": []}
    score_cache = None
    if extract_attention_scores is not None:
        score_cache = {
            "coef": logreg.coef_[0],
            "intercept": float(logreg.intercept_[0]),
            "torch_weights": {},
        }
    ordered_indices = sorted(eval_indices, key=lambda idx: len(dataset[int(idx)]["seq"]))
    pbar = tqdm(ordered_indices, desc="Eval", leave=False)
    for index in pbar:
        data = dataset[int(index)]
        i_idx, j_idx = pair_indices(data, EVAL_SEQ_SEP)
        pbar.set_postfix(L=len(data["seq"]))
        if len(i_idx) == 0:
            results["skipped"].append(data["name"])
            continue
        if REQUIRE_L_CONTACTS:
            n_contacts = int(data["contact_map"][i_idx, j_idx].sum())
            if n_contacts < len(data["seq"]):
                results["skipped"].append(data["name"])
                continue
        labels = data["contact_map"][i_idx, j_idx].astype(np.int64)
        if extract_attention_scores is None:
            features = extract_attention_features(data["seq"], i_idx, j_idx)
            scores = logistic_regression_scores(features, logreg)
        else:
            scores = extract_attention_scores(
                data["seq"],
                i_idx,
                j_idx,
                score_cache,
            )
        metrics = evaluate_contact_scores(scores, labels, len(data["seq"]))
        results["P@L"].append(metrics["P@L"])
        results["P@L/5"].append(metrics["P@L/5"])
        results["names"].append(data["name"])
    return results


def run_contact_benchmark(
    extract_attention_features,
    train_dataset: ESMStructuralSplitDataset,
    n_bootstraps: int,
    n_train: int,
    eval_dataset=None,
    benchmark_name: str = "contact benchmark",
    extract_attention_scores=None,
) -> None:
    print(f"Benchmark: {benchmark_name}")
    print(
        "Definition: C-alpha distance < "
        f"{CONTACT_THRESHOLD:g}A, train |i-j| >= {TRAIN_SEQ_SEP}, "
        f"eval |i-j| >= {EVAL_SEQ_SEP}"
    )
    print(f"Bootstrap: n_bootstraps={n_bootstraps}, seed=bootstrap_i, n_train={n_train}")
    print(f"Truncation: first {MAX_LENGTH} residues")
    print(f"Train records: {len(train_dataset)}")
    if eval_dataset is None:
        print("Eval records: held-out records from the same dataset")
    else:
        print(f"Eval records: {len(eval_dataset)}")

    p_at_l = []
    p_at_l5 = []
    for bootstrap_i in range(n_bootstraps):
        train_indices, heldout_indices = make_bootstrap_split(
            len(train_dataset),
            bootstrap_i=bootstrap_i,
            n_train=n_train,
        )
        logreg = fit_bootstrap_model(
            extract_attention_features,
            train_dataset,
            train_indices,
            bootstrap_i=bootstrap_i,
        )
        if eval_dataset is None:
            current_eval_dataset = train_dataset
            eval_indices = heldout_indices
        else:
            current_eval_dataset = eval_dataset
            eval_indices = np.arange(len(eval_dataset), dtype=np.int64)
        result = evaluate_bootstrap_model(
            extract_attention_features,
            logreg,
            current_eval_dataset,
            eval_indices,
            extract_attention_scores=extract_attention_scores,
        )
        p_l = float(np.nanmean(result["P@L"]))
        p_l5 = float(np.nanmean(result["P@L/5"]))
        p_at_l.append(p_l)
        p_at_l5.append(p_l5)
        print(
            f"Bootstrap {bootstrap_i}: "
            f"eval={len(result['names'])}, skipped={len(result['skipped'])}, "
            f"P@L={p_l:.4f}, P@L/5={p_l5:.4f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    p_at_l_arr = np.array(p_at_l, dtype=np.float32)
    p_at_l5_arr = np.array(p_at_l5, dtype=np.float32)
    print("> Results")
    print(f"  P@L: {p_at_l_arr}")
    print(f"  P@L/5: {p_at_l5_arr}")
    print(f"  P@L mean/std: {p_at_l_arr.mean():.4f} +/- {p_at_l_arr.std():.4f}")
    print(f"  P@L/5 mean/std: {p_at_l5_arr.mean():.4f} +/- {p_at_l5_arr.std():.4f}")


def run_benchmark_for_split(
    extract_attention_features,
    root_path: str | Path,
    split_level: str,
    cv_partition: str,
    n_bootstraps: int,
    n_train: int,
    extract_attention_scores=None,
) -> None:
    train_dataset = ESMStructuralSplitDataset(
        split_level,
        cv_partition,
        split="train",
        root_path=root_path,
    )
    valid_dataset = ESMStructuralSplitDataset(
        split_level,
        cv_partition,
        split="valid",
        root_path=root_path,
    )
    run_contact_benchmark(
        extract_attention_features,
        train_dataset=train_dataset,
        n_bootstraps=n_bootstraps,
        n_train=n_train,
        eval_dataset=valid_dataset,
        benchmark_name=f"Rao2021 {split_level}/{cv_partition}",
        extract_attention_scores=extract_attention_scores,
    )


def run_benchmark(
    extract_attention_features,
    root_path: str | Path,
    n_bootstraps: int = N_BOOTSTRAPS,
    n_train: int = N_TRAIN,
    extract_attention_scores=None,
) -> None:
    for split_level in SPLIT_LEVELS:
        for cv_partition in CV_PARTITIONS:
            run_benchmark_for_split(
                extract_attention_features,
                root_path=root_path,
                split_level=split_level,
                cv_partition=cv_partition,
                n_bootstraps=n_bootstraps,
                n_train=n_train,
                extract_attention_scores=extract_attention_scores,
            )
