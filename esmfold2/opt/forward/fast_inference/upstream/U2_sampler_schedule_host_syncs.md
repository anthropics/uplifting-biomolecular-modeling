# PR: Remove per-step host syncs from the EDM sampler schedule

sample() calls float(t.item()) three times per diffusion step on 0-d CUDA tensors of the noise schedule (204 device->host syncs per 68-step fold). Converting the schedule and gammas to Python lists once before the loop gives bit-identical float values (float32 -> Python float is exact either way) and removes the syncs; measured -4 to -6% sampler wall on H100 eager, and it is a prerequisite for CUDA-graph capture of the step.

Diff: `U2_sampler_schedule_host_syncs.diff` (against Biohub/transformers @ ef32577f55da19a4989cd7b22e004dc43a4998cb).
