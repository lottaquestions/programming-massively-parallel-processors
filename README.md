# programming-massively-parallel-processors
Code examples and solved exercises from the book Programming Massively Parallel Processors.

This repo is a work-in-progress, and so has two variants of kernels:

1. C++ CUDA kernels that are launched from within pytorch e.g see image blur (see its subdirectory) implementation in `chapter_3` or the rgb to gray scale conversion implementations in `chapter_4_and_5`.
2. Standalone kernels with no launching "infra" code and are replicated verbatim from the book.

The goal is to get as many of the kernels into the option 1 state above since pytorch allows for easy testing with images and other kinds of input data. Pytorch also allows for easy setup to process images in batches as can be seen in `chapter_4_and_5/batched_rgb_to_grayscale`.

To test the kernels launched from pytorch, simply run the respective python launcher scripts. Test data has been included in the respective directories of each kernel. Pytorch must be installed with CUDA capability, and an NVIDIA GPU must be present on the system for the scripts to work.
