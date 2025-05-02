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

# Get all image files in the current directory. Note!!: The images have
# to be of the same dimensions. This script does not resize images of
# differing dimensions
img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
list_of_paths = sorted([p for p in Path.cwd().iterdir() if p.suffix.lower() in img_extensions])

batched_input_imgs = torch.stack([read_image(str(img_path)).contiguous() for img_path in list_of_paths]).cuda()
output_img_tensor = module.rgb_to_grayscale(batched_input_imgs) ; torch.cuda.synchronize()

for i, single_img_tensor in enumerate(output_img_tensor):
    write_png(single_img_tensor.cpu().unsqueeze(0), f'output_{i}.png')

import time
t0 = time.perf_counter_ns()
for i in range(10_000):
    module.rgb_to_grayscale_out(output_img_tensor, batched_input_imgs)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()

print((t1-t0) / 10_000 / 1_000, "µs" )

with torch.profiler.profile() as prof:
    module.rgb_to_grayscale_out(output_img_tensor, batched_input_imgs)
    torch.cuda.synchronize()

print(prof.key_averages().table())