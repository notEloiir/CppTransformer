# Low-Level Transformer Implementation in C++ and CUDA

## Overview
An implementation of the original transformer model as described in the 
"Attention is all you need" 2017 paper (for the most part*).
Implemented entirely in standard C++ and CUDA 
(with cuBLAS for optimized linear algebra operations).

**It's meant as a self-study project. The goal is to gain a deep understanding 
of the inner workings of Transformer architectures by building them from scratch, 
without relying on high-level libraries.**  
*This implementation may differ from the original paper 
as part of learning through experimentation.

*Note: This is a work-in-progress and intended primarily for educational purposes,
not for production use.*

## Features
- Full transformer architecture implemented from scratch
- 2D tensors handling data management, implemented from scratch
- CUDA-accelerated operations: utilizing CUDA for parallel processing and cuBLAS for optimized matrix multiplication
- Modular design
- Optimized for 1 CUDA-capable device. Does work on CPU only. 
Does work on multiple device systems but utilises only 1 GPU.

## Installation prerequisites
- C++20 or later
- CUDA toolkit 12.5 or later (even if you want to run it without using CUDA)
- CUDA-capable GPU (optional but recommended)
