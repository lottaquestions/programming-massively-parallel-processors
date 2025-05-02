from pathlib import Path
import torch
import torch.utils.cpp_extension
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

# Run this file in a python env with pytorch with CUDA support installed
# in it.

cuda_source = Path("rgb_to_grayscale.cu").read_text()
cpp_source = """
torch::Tensor rgb_to_grayscale(const torch::Tensor& input);
torch::Tensor rgb_to_grayscale_out(torch::Tensor output, const torch::Tensor& input);
"""
module = load_inline(
    name = "rgb_to_grayscale_ext",
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['rgb_to_grayscale', 'rgb_to_grayscale_out'],
    with_cuda = True,
    extra_cuda_cflags = ['--ptxas-options=-v'],
    verbose = True)
input_img_tensor = read_image('pexels-public-domain.jpg').contiguous().cuda()
output_img_tensor = module.rgb_to_grayscale(input_img_tensor) ; torch.cuda.synchronize()
# torch.unsqueeze Returns a new tensor with a dimension of size one inserted at the specified position.
# In the case below, write_png expects a 3D tensor, so squeeze converts my  2D tensor of (H,W) to
# (1, H, W)
write_png(output_img_tensor.cpu().unsqueeze(0), 'output.png')

import time
t0 = time.perf_counter_ns()
for i in range(10_000):
    module.rgb_to_grayscale_out(output_img_tensor, input_img_tensor)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()

print((t1-t0) / 10_000 / 1_000, "µs" )

with torch.profiler.profile() as prof:
    module.rgb_to_grayscale_out(output_img_tensor, input_img_tensor)
    torch.cuda.synchronize()

print(prof.key_averages().table())