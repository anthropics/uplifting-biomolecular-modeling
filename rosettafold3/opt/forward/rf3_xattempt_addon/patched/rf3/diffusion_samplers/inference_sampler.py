import torch
from beartype.typing import Any, Literal
from jaxtyping import Float

from foundry.utils.ddp import RankedLogger
from foundry.utils.rigid import rot_vec_mul
from foundry.utils.rotation_augmentation import (
    centre_random_augmentation,
    uniform_random_rotation,
)

import rf3.graph_flags as GF  # [rf3_cudagraph] runtime flags (RF3_CUDAGRAPH=0 -> stock path)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class SampleDiffusion:
    """Algorithm 18"""

    def __init__(
        self,
        *,
        # Hyperparameters
        num_timesteps: int,  # AF-3: 200
        min_t: int,  # AF-3: 0
        max_t: int,  # AF-3: 1
        sigma_data: int,  # AF-3: 16
        s_min: float,  # AF-3: 4e-4
        s_max: int,  # AF-3: 160
        p: int,  # AF-3: 7
        gamma_0: float,  # AF-3: 0.8
        gamma_min: float,  # AF-3: 1.0,
        noise_scale: float,  # AF-3: 1.003,
        step_scale: float,  # AF-3: 1.5,
        solver: Literal["af3"],
    ):
        """Initialize the diffusion sampler, to perform a complete diffusion roll-out with the given recycling outputs.

        We do not use default values for the parameters to make the Hydra configuration the single source of truth and avoid silent failures.

        Args:
            num_timesteps (int): The number of timesteps for which the noise schedule is constructed. Default is 200, per AF3.
            min_t (float): The minimum value of t in the schedule. Default is 0, per AF3.
            max_t (float): The maximum value of t in the schedule. Default is 1, per AF3.
            sigma_data (int): A constant determined by the variance of the data. Default is 16, as defined in the AlphaFold 3 Supplement (Algorithm 20, Diffusion Module).
            s_min (float): The minimum value of the noise schedule. Default is 4e-4, per AF3.
            s_max (float): The maximum value of the noise schedule. Default is 160, per AF3.
            p (int): A constant that determines the shape of the noise schedule. Default is 7, per AF3.
            gamma_0 (float): The value of gamma when t > gamma_min. Default is 0.8, per AF3.
            solver (str): The solver to use for the diffusion process. Default is "af3".

            TODO: Continue documentation of the remaining parameters.
        """
        self.num_timesteps = num_timesteps
        self.min_t = min_t
        self.max_t = max_t
        self.sigma_data = sigma_data
        self.s_min = s_min
        self.s_max = s_max
        self.p = p
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.solver = solver

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        return t_hat

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
    ) -> torch.Tensor:
        """Sample initial point cloud from a normal distribution.

        Args:
            c0 (torch.Tensor): A scalar tensor that will be used to scale the initial point cloud. Effectively, the same as
                directly changing the standard deviation of the normal distribution. Derived from noise_schedule[0].
            D (int): The number of structures to sample.
            L (int): The number of atoms in the structure.
            coord_atom_lvl_to_be_noised (torch.Tensor): The atom-level coordinates to be noised (either completely or partially)
        """
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        X_L = noise + coord_atom_lvl_to_be_noised

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        S_inputs_I: Float[torch.Tensor, "I c_s_inputs"],
        S_trunk_I: Float[torch.Tensor, "I c_s"],
        Z_trunk_II: Float[torch.Tensor, "I I c_z"],
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
    ) -> dict[str, Any]:
        """Perform a complete diffusion roll-out with the given recycling outputs.

        [rf3_cudagraph] Dispatcher: RF3_CUDAGRAPH=0 -> the unchanged stock roll-out (_sample_diffusion_stock);
        RF3_CUDAGRAPH=replay -> pre-drawn RNG + eager replay; RF3_CUDAGRAPH=1 -> pre-drawn RNG + CUDA-graph replay.
        The non-stock paths compute the same function with the same draws (see _predraw / _step_body).

        Args:
            diffusion_module (torch.nn.Module): The diffusion module to use for denoising. If using EMA and performing validation or inference,
                this model should be the EMA model.
        """
        mode = GF.CUDAGRAPH_MODE
        GF.hoist_begin(n_calls=int(self.num_timesteps) - 1)  # [xattempt_hoist] per-roll-out cache of step-invariant sub-graphs (no-op unless RF3_HOIST=1); n_calls = denoiser calls per roll-out (the recompute switch checks the last one)
        try:
            if mode == "0" or not S_inputs_I.is_cuda:
                return self._sample_diffusion_stock(
                    S_inputs_I=S_inputs_I,
                    S_trunk_I=S_trunk_I,
                    Z_trunk_II=Z_trunk_II,
                    f=f,
                    diffusion_module=diffusion_module,
                    diffusion_batch_size=diffusion_batch_size,
                    coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                )
            return self._sample_diffusion_predrawn(
                S_inputs_I=S_inputs_I,
                S_trunk_I=S_trunk_I,
                Z_trunk_II=Z_trunk_II,
                f=f,
                diffusion_module=diffusion_module,
                diffusion_batch_size=diffusion_batch_size,
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                use_graph=(mode == "1"),
            )
        finally:
            GF.hoist_end()

    def _sample_diffusion_stock(
        self,
        *,
        S_inputs_I: Float[torch.Tensor, "I c_s_inputs"],
        S_trunk_I: Float[torch.Tensor, "I c_s"],
        Z_trunk_II: Float[torch.Tensor, "I I c_z"],
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
    ) -> dict[str, Any]:
        """The unchanged stock roll-out (verbatim)."""
        # Construct the noise schedule t_hat for inference on the appropriate device
        noise_schedule = self._construct_inference_noise_schedule(
            device=S_inputs_I.device
        )

        # Infer number of atoms from any atom-level feature
        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        # Initial X_L is drawn from a normal distribution with a mean vector of 0 and a
        # covariance matrix equal to the 3x3 identity matrix, scaled by the noise schedule
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
        )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        t_hats = []

        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            # (All predicted atoms exist)
            X_exists_L = torch.ones((D, L)).bool()  # (D, L)

            # Apply a random rotation and translation to the structure
            # TODO: Make s_trans a hyperparameter
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)

            # Update gamma
            gamma = self.gamma_0 if c_t > self.gamma_min else 0

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            X_denoised_L = diffusion_module(
                X_noisy_L=X_noisy_L,
                t=t_hat.tile(D),
                f=f,
                S_inputs_I=S_inputs_I,
                S_trunk_I=S_trunk_I,
                Z_trunk_II=Z_trunk_II,
            )

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + self.step_scale * d_t * delta_L

            X_noisy_L_scaled = (
                X_noisy_L
                / (torch.sqrt(t_hat[..., None, None] ** 2 + self.sigma_data**2))
            ) * self.sigma_data
            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
        )

    # ------------------------------------------------------------------------------------------------------------------
    # [rf3_cudagraph] pre-drawn RNG + (eager | CUDA-graph) replay of the SAME roll-out
    # ------------------------------------------------------------------------------------------------------------------
    # Stock per-step RNG order (see _sample_diffusion_stock / foundry.utils.rotation_augmentation):
    #   before the loop:  CUDA generator:  torch.normal(size=(D, L, 3), device=cuda)                 (initial noise)
    #   per step:         CPU generator:   torch.rand((D,)) x3 (theta_x, theta_y, theta_z)         (uniform_random_rotation)
    #                     CPU generator:   torch.normal(size=(D, 1, 3))                             (translation)
    #                     CUDA generator:  torch.normal(size=(D, L, 3), device=cuda)                 (epsilon)
    # _predraw consumes exactly these generators in exactly this order with exactly these shapes/dtypes and keeps the
    # draws; the CPU rotation matrices are built by the stock uniform_random_rotation() and moved to the device with the
    # stock .to(device). The sampler arithmetic of _step_body is the stock arithmetic with the stored draws substituted
    # for the in-loop calls, so the eager replay is bitwise identical to the stock roll-out.
    # The noise schedule and gamma/t_hat/d_t constants are computed ONCE on the
    # device exactly as stock computes them per step (same tensor ops on the same float32 schedule tensor), so the per-step
    # constants are the same bits; `gamma = gamma_0 if c_t > gamma_min else 0` is evaluated on the host for all steps at once
    # (the only host sync of the whole roll-out).

    def _predraw(self, noise_schedule, D, L, device):
        """Consume the generators in stock order; return the stored draws for every step."""
        T = noise_schedule.shape[0] - 1
        init_noise = torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device)  # CUDA gen (stock _get_initial_structure)
        rots, trans, eps = [], [], []
        for _ in range(T):
            rots.append(uniform_random_rotation((D,)))  # CPU gen: 3x torch.rand((D,)) -> [D,3,3] (stock get_random_augmentation)
            trans.append(torch.normal(mean=0, std=1, size=(D, 1, 3)))  # CPU gen (stock: s_trans * normal(...).to(device))
            eps.append(torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device))  # CUDA gen (stock epsilon_L)
        R = torch.stack([r.to(device) for r in rots])  # [T, D, 3, 3] (same .to(device) conversion as stock)
        Tr = torch.stack([t.to(device) for t in trans])  # [T, D, 1, 3]
        E = torch.stack(eps)  # [T, D, L, 3]
        return init_noise, R, Tr, E

    def _step_constants(self, noise_schedule):
        """Per-step constants t_hat, d_t, sqrt(t_hat^2 - c_{t-1}^2) as stock computes them (same ops on the same tensors)."""
        gamma_min = self.gamma_min
        c_prev = noise_schedule[:-1]
        c_t = noise_schedule[1:]
        # stock: `gamma = self.gamma_0 if c_t > self.gamma_min else 0` (python float), then t_hat = c_t_minus_1 * (gamma + 1)
        gam = (c_t > gamma_min).cpu().tolist()  # ONE host sync for the whole roll-out (stock: one per step)
        t_hat = torch.stack([c_prev[i] * ((self.gamma_0 if g else 0) + 1) for i, g in enumerate(gam)])
        d_t = torch.stack([c_t[i] - t_hat[i] for i in range(len(gam))])
        noise_std = torch.stack(
            [self.noise_scale * torch.sqrt(torch.square(t_hat[i]) - torch.square(c_prev[i])) for i in range(len(gam))]
        )
        return t_hat, d_t, noise_std

    @staticmethod
    def _centre(X_L):
        """Stock `centre()` for the all-atoms-exist case (X_exists_L = ones): X - mean over atoms; bitwise the same reduction."""
        # stock: X_L[X_exists_L] = X_L[X_exists_L] - torch.mean(X_L[X_exists_L], dim=-2, keepdim=True)
        # with X_exists_L all-True the boolean index yields the [D*L, 3] view, so the mean is over ALL D*L rows (dim=-2 of [D*L,3]).
        D, L, _ = X_L.shape
        flat = X_L.reshape(D * L, 3)
        return (flat - torch.mean(flat, dim=-2, keepdim=True)).reshape(D, L, 3)

    def _step_body(self, X_L, R, tr, eps, t_hat, d_t, noise_std, D, diffusion_module, f, S_inputs_I, S_trunk_I, Z_trunk_II):
        """One stock sampler step with the stored draws. Returns (X_L_next, X_noisy_L, X_denoised_L)."""
        X_L = self._centre(X_L)
        X_L = rot_vec_mul(R[:, None], X_L) + 1.0 * tr  # stock: rot_vec_mul(R[:, None], X_L) + s_trans * normal.to(device), s_trans = 1.0
        epsilon_L = noise_std * eps  # stock: noise_scale * sqrt(...) * normal(...)  (noise_std already = noise_scale*sqrt)
        X_noisy_L = X_L + epsilon_L
        X_denoised_L = diffusion_module(
            X_noisy_L=X_noisy_L,
            t=t_hat.tile(D),
            f=f,
            S_inputs_I=S_inputs_I,
            S_trunk_I=S_trunk_I,
            Z_trunk_II=Z_trunk_II,
        )
        delta_L = (X_noisy_L - X_denoised_L) / t_hat
        X_L_next = X_noisy_L + self.step_scale * d_t * delta_L
        return X_L_next, X_noisy_L, X_denoised_L

    def _sample_diffusion_predrawn(
        self,
        *,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        f,
        diffusion_module,
        diffusion_batch_size,
        coord_atom_lvl_to_be_noised,
        use_graph: bool,
    ) -> dict[str, Any]:
        device = S_inputs_I.device
        noise_schedule = self._construct_inference_noise_schedule(device=device)
        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        T = noise_schedule.shape[0] - 1
        if GF.GRAPH_SAFE_OPS:  # the host-known token count used by the capture-safe encoder must equal the stock tok_idx.max()+1 (one sync per fold)
            n_tok = int(f["atom_to_token_map"].max()) + 1
            assert n_tok == S_trunk_I.shape[-2], (n_tok, tuple(S_trunk_I.shape))

        init_noise, R_all, Tr_all, E_all = self._predraw(noise_schedule, D, L, device)
        X_L = noise_schedule[0] * init_noise + coord_atom_lvl_to_be_noised  # stock _get_initial_structure
        t_hat_all, d_t_all, std_all = self._step_constants(noise_schedule)

        X_noisy_L_traj, X_denoised_L_traj, t_hats = [], [], []
        step_cb = GF.STEP_CALLBACK

        if not use_graph:
            for i in range(T):
                if step_cb is not None:
                    step_cb(i)
                X_L, X_noisy_L, X_denoised_L = self._step_body(
                    X_L, R_all[i], Tr_all[i], E_all[i], t_hat_all[i], d_t_all[i], std_all[i], D,
                    diffusion_module, f, S_inputs_I, S_trunk_I, Z_trunk_II,
                )
                X_noisy_L_traj.append((X_noisy_L / (torch.sqrt(t_hat_all[i][..., None, None] ** 2 + self.sigma_data**2))) * self.sigma_data)
                X_denoised_L_traj.append(X_denoised_L)
                t_hats.append(t_hat_all[i])
            return dict(X_L=X_L, X_noisy_L_traj=X_noisy_L_traj, X_denoised_L_traj=X_denoised_L_traj, t_hats=t_hats)

        # ---------------- CUDA-graph replay: static buffers, capture one step, replay T times ----------------
        # Static inputs: X (state), R, tr, eps, t_hat, d_t, noise_std. Static outputs: X_next, X_noisy, X_denoised.
        # Each step copies the step's draws/constants into the static buffers (device-to-device copies inside the stream),
        # replays the graph and copies the outputs out (the trajectory lists keep clones).
        sX = X_L.clone()
        sR = R_all[0].clone()
        sTr = Tr_all[0].clone()
        sE = E_all[0].clone()
        s_t = t_hat_all[0].clone()
        s_dt = d_t_all[0].clone()
        s_std = std_all[0].clone()

        def _run():
            return self._step_body(sX, sR, sTr, sE, s_t, s_dt, s_std, D, diffusion_module, f, S_inputs_I, S_trunk_I, Z_trunk_II)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        n_warmup = GF.warmup_steps()  # [rf3_cudagraph] GRAPH_WARMUP, or GRAPH_WARMUP_REPEAT after the process's first capture under the `warm` lever
        with torch.cuda.stream(side):
            for _ in range(n_warmup):  # warm-up on the side stream (cuBLAS/cuDNN workspace + autotuning); results discarded
                _run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # capture_error_mode default ("global")
            oX, oNoisy, oDen = _run()
        t1.record()
        t1.synchronize()
        GF.LAST_CAPTURE.update(capture_ms=t0.elapsed_time(t1), n_atoms=int(L), D=int(D), T=int(T), warmup=n_warmup)
        GF.note_capture(n_warmup)  # [rf3_cudagraph] the capture count warmup_steps() keys on

        # the warm-up/capture consumed nothing from any generator (all draws are stored); the state buffer is re-initialised here
        sX.copy_(X_L)
        for i in range(T):
            if step_cb is not None:
                step_cb(i)
            sR.copy_(R_all[i]); sTr.copy_(Tr_all[i]); sE.copy_(E_all[i])
            s_t.copy_(t_hat_all[i]); s_dt.copy_(d_t_all[i]); s_std.copy_(std_all[i])
            graph.replay()
            X_noisy_L_traj.append((oNoisy / (torch.sqrt(t_hat_all[i][..., None, None] ** 2 + self.sigma_data**2))) * self.sigma_data)
            X_denoised_L_traj.append(oDen.clone())
            t_hats.append(t_hat_all[i])
            sX.copy_(oX)
        X_L = sX.clone()
        GF.LAST_CAPTURE["graph"] = None  # graph object freed below
        del graph

        return dict(X_L=X_L, X_noisy_L_traj=X_noisy_L_traj, X_denoised_L_traj=X_denoised_L_traj, t_hats=t_hats)


class SamplePartialDiffusion(SampleDiffusion):
    def __init__(self, partial_t: int, **kwargs):
        super().__init__(**kwargs)
        self.partial_t = partial_t

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference with partial t."""
        t_hat_full = super()._construct_inference_noise_schedule(device)

        assert (
            self.partial_t < self.num_timesteps
        ), f"Partial t ({self.partial_t}) must be less than num_timesteps ({self.num_timesteps})"
        ranked_logger.info(
            f"Using partial t index: {self.partial_t} [e.g., {t_hat_full[self.partial_t]:.4}], or {self.partial_t / (self.num_timesteps):.2%}, by index (100% is data, 0% is noise)"
        )

        return t_hat_full[self.partial_t :]
