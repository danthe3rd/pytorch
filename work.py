"""
python /data/home/dhaziza/pytorch/test/test_nn.py TestNNDeviceTypeCUDA.test_upsamplingBilinear2d_aa_correctness_memory_format0_cuda
PROFILE=1 /usr/local/cuda-12.1/bin/ncu --force-overwrite \
    -o profile.ncu-rep --import-source yes\
    --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
    --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --section WarpStateStats --section SpeedOfLight_RooflineChart --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables \
    python work.py
"""
import torch
import time
import os
from torch import nn

WARMUP_ITERS = 2
REPEAT_ITERS = 10
PROFILE = os.environ.get("PROFILE", "0") == "1"
print(f"profile mode: {PROFILE}")
if PROFILE:
    WARMUP_ITERS = 0
    REPEAT_ITERS = 1
def benchmark_fn(name, fn, *args, **kwargs):
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(REPEAT_ITERS):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    dt = (time.time() - begin)
    dt_us = int(dt * 1000000) / REPEAT_ITERS
    return dt_us
dtype = torch.bfloat16
device = torch.device("cuda")
l_embed_dim = [768, 1024, 1536, 4096]
if PROFILE:
    l_embed_dim = [l_embed_dim[0]]
l_grid_size = [16, 32, 48, 64]
l_output_size = [6, 7, 14, 16]
interpolation_mode = "bicubic"
antialias = True
all_results = []
print("[")
for embed_dim in l_embed_dim:
    for grid_size in l_grid_size:
        base_posemb_grid = torch.randn([embed_dim, grid_size, grid_size]).to(device=device, dtype=dtype)
        base_posemb_grid.requires_grad_(True)
        for output_size in l_output_size:
            def fw():  
                # interpolated should be [1, embed_dim, grid_size, grid_size]
                # ultimately we do patch_tokens += patch_tokens + interpolated
                # with patch_tokens size: (B, n_patches, embed_dim)
                # backward is also expensive
                return nn.functional.interpolate(
                    base_posemb_grid.unsqueeze(0),
                    size=(output_size, output_size),
                    mode=interpolation_mode,
                    antialias=antialias,
                )
            out = fw()
            def bw():
                base_posemb_grid.grad = None
                out.backward(out, retain_graph=True)
            bw()
            bw()
            name = f"embed_dim: {embed_dim}, grid_size: {grid_size}x{grid_size}, output_size: {output_size}x{output_size}"
            fw_time = benchmark_fn(name, fw)
            bw_time = benchmark_fn(name, bw)
            # roofline_time = benchmark_fn("read_both", lambda: (base_posemb_grid.sum(), out.sum()))
            print([embed_dim, grid_size, output_size, fw_time, bw_time], ",")
print("]")