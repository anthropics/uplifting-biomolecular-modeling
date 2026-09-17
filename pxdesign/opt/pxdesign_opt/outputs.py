"""The upstream dump layout — read, never rewritten: `<out>/<task>/seed_<s>/predictions/<task>_sample_<i>.cif` plus `SUCCESS_FILE`
per (task, seed) (`pxdesign/runner/dumper.py:82-104,120,141`). Every mode writes exactly this layout;
the package adds `opt_manifest.json` (and `stock_env_proof.json` for mode off) beside it, listed never judged.

`count_designs` counts the design CIFs against tasks x seeds x N_sample after the run (the DONE / INCOMPLETE line and `opt_manifest.json`
"outputs"); nothing is digested. Task names come from the task file: a JSON task list's "name" fields, or — for a YAML input, which upstream
converts into one task (`utils/inputs.py:27-147,156-181`) — upstream's converted copy `<out>/input_tasks.json` once it exists (`task_names`).
A reused `<out>` is upstream's to handle: it skips any (task, seed) whose dump dir holds `SUCCESS_FILE` (`runner/inference.py:174-176`,
`dumper.py:125-128`) — its resume rule. The design verb takes the census of that rule BEFORE the run (`dumped_pairs`: the pairs already
dumped) and the count names it after: `skipped_existing` (this job's pairs upstream skipped as already dumped), `skipped_pairs`, `produced`
(design CIFs under pairs dumped THIS run) and `nothing_ran` (every pair of the job was already dumped: nothing was sampled in this process —
the verb then prints NOTHING RAN and leaves `opt_manifest.json` as the earlier run wrote it). The count covers this job's own (task, seed)
pairs when both are known (`scope=job`: the task names and the seed list — given, or the one upstream derived on the in-process route); only
on the console-script route without `--seeds` (the derived seed is upstream's, unseen) does it fall back to every design under `<out>`
(`scope=dir`, recorded in `opt_manifest.json` "outputs"; `nothing_ran` is never claimed there — the job's pairs are not known).
"""
import glob
import json
import os
from typing import List, Optional

SUCCESS_FILE = "SUCCESS_FILE"
PREDICTIONS_DIR = "predictions"
MANIFEST_NAME = "opt_manifest.json"
STOCK_PROOF_NAME = "stock_env_proof.json"
CONVERTED_TASKS = "input_tasks.json"                    # upstream's converted copy of the task list inside the dump directory (runner/inference.py:210-211)
LISTED_NOT_JUDGED = (SUCCESS_FILE, MANIFEST_NAME, STOCK_PROOF_NAME, "config.yaml", CONVERTED_TASKS, "ERR")
YAML_SUFFIXES = (".yaml",)                              # the input extensions upstream converts (utils/inputs.py:165-176: .json | .yaml)


def design_files(out_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, "*", "seed_*", PREDICTIONS_DIR, "*.cif")))


def success_files(out_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, "*", "seed_*", SUCCESS_FILE)))


def pair_key(task: str, seed) -> str:
    """``<task>/seed_<s>`` — the (task, seed) dump directory relative to the dump root (upstream's layout)."""
    return f"{task}/seed_{int(seed)}"


def _key_of(path: str, out_dir: str) -> str:
    rel = os.path.relpath(path, out_dir).split(os.sep)
    return "/".join(rel[:2])


def dumped_pairs(out_dir: str) -> List[str]:
    """The (task, seed) pairs already dumped under ``out_dir`` — the dump dirs holding `SUCCESS_FILE`, which is exactly upstream's resume rule
    (`dumper.check_completion`: it skips such a pair, "already dumped"). Taken BEFORE a run it is the one source of `skipped_existing`."""
    return sorted(_key_of(p, out_dir) for p in success_files(out_dir))


def read_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_yaml(path: Optional[str]) -> bool:
    return bool(path) and os.path.splitext(str(path))[1].lower() in YAML_SUFFIXES


def task_names(path: Optional[str], out_dir: Optional[str] = None) -> List[str]:
    """The task names of a task file: a JSON list's "name" fields (upstream requires the field, `data/infer_data_pipeline.py:442`; `sample_<i>`
    is this package's placeholder for an entry without one, so the count still closes); for a YAML input the names in upstream's converted
    copy `<out_dir>/input_tasks.json` when it exists, else [] (not yet known: upstream names it at conversion)."""
    if is_yaml(path):
        conv = os.path.join(out_dir, CONVERTED_TASKS) if out_dir else None
        if not conv or not os.path.isfile(conv):
            return []
        path = conv
    if not path or not os.path.isfile(path):
        return []
    t = read_tasks(path)
    t = t if isinstance(t, list) else [t]
    return [str(x.get("name") or f"sample_{i}") if isinstance(x, dict) else f"sample_{i}" for i, x in enumerate(t)]


def n_tasks_of(path: Optional[str]) -> int:
    """The number of tasks a task file declares before the run: a JSON list's length; 1 for a YAML input (one task after upstream's conversion)."""
    if is_yaml(path):
        return 1
    return len(task_names(path))


def count_designs(out_dir: str, tasks: List[str], seeds: Optional[List[int]], n_sample: int, n_tasks: Optional[int] = None,
                  existed: Optional[List[str]] = None) -> dict:
    """Design CIFs against tasks x seeds x N_sample: per (task, seed) dump dir when the names and seeds are known (scope job), else every design
    under ``out_dir`` (scope dir; an unknown seed list counts as one seed — upstream derives exactly one). ``existed`` = `dumped_pairs(out_dir)`
    taken before the run: the pairs upstream skipped as already dumped (`skipped_existing`, `skipped_pairs`), the designs under pairs dumped this
    run (`produced`), and `nothing_ran` — scope job, at least one pair, every pair already dumped (never claimed on scope dir)."""
    existed = set(existed or ())
    if tasks and seeds:
        ts = [(t, int(s)) for t in tasks for s in seeds]
        pairs = [pair_key(t, s) for t, s in ts]
        files = [p for t, s in ts for p in sorted(glob.glob(os.path.join(out_dir, t, f"seed_{s}", PREDICTIONS_DIR, "*.cif")))]
        scope = "job"
        skipped = [k for k in pairs if k in existed]
    else:
        files, scope, pairs = design_files(out_dir), "dir", None
        skipped = sorted(existed)                                          # the job's pairs unknown: every pre-existing dump under the root is inside the count, named as such
    n_t = len(tasks) if tasks else int(n_tasks or 0)
    expected = n_t * (len(seeds) if seeds else 1) * int(n_sample)
    produced = sum(1 for p in files if _key_of(p, out_dir) not in existed)
    return {"n_designs": len(files), "expected": expected, "complete": len(files) == expected, "scope": scope,
            "n_pairs": len(pairs) if pairs is not None else None, "skipped_existing": len(skipped), "skipped_pairs": skipped, "produced": produced,
            "nothing_ran": bool(pairs) and len(skipped) == len(pairs),
            "n_success_files": len(success_files(out_dir)), "cifs": [os.path.relpath(p, out_dir) for p in files]}


def summarize(out_dir: str, tasks_path: Optional[str], seeds: Optional[List[int]], n_sample: int, existed: Optional[List[str]] = None) -> dict:
    """The job's outputs record: task names, the seed list as known ([] = upstream's one derived seed, unseen), the count (`count_designs`, with
    the pre-run resume census ``existed``)."""
    names = task_names(tasks_path, out_dir)
    n_tasks = len(names) if names else (n_tasks_of(tasks_path) if tasks_path and os.path.exists(str(tasks_path)) else 0)
    seeds = [int(s) for s in (seeds or [])]
    c = count_designs(out_dir, names, seeds, n_sample, n_tasks=n_tasks, existed=existed)
    return {"tasks": names, "n_tasks": n_tasks, "seeds": seeds, "n_seeds": len(seeds) or 1, "N_sample": int(n_sample), **c}
