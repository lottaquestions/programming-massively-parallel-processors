from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

# Run this file in a python env with pytorch with CUDA support installed
# in it.

def compile_extension():
    cuda_source = Path('imageBlur.cu').read_text()
    cpp_source = 'torch::Tensor imageBlur(torch::Tensor image, int radius);'

    # Load the CUDA kernel as a Pytorch extension
    image_blur_extension = load_inline(
        name='image_blur_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['imageBlur'],
        with_cuda=True,
        extra_cuda_cflags=['-O2'],
    )
    return image_blur_extension

def main():
    """
    Use torch cpp inline extension function to compile the kernel in imageBlur.cu.
    Read input image, apply image blur custom cuda kernel and write result out into output.png
    """
    ext = compile_extension()

    x = read_image('pexels-public-domain.jpg').contiguous().cuda()
    assert x.dtype == torch.uint8
    print("Input image:", x.shape, x.dtype)
    y = ext.imageBlur(x,8)
    print('Output image: ', y.shape, y.dtype)
    write_png(y.cpu(), 'output.png')

if __name__ == '__main__':
    main()