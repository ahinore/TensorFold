# Notes on this fork

This branch is a personal fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) at v1.2.1. It is not an official release and is not supported by the upstream project.

It changes the native GLM-5.3-Flash engine on CUDA to serve Z.ai's official FP8 release (`zai-org/GLM-5.3-Flash`) on four DGX Sparks in a ring without a switch:

- `zig/src/families/glm/`: routed experts and dense layers are read as block FP8 with the existing `fp8g` kernels, and the `weight_scale_inv` scale name is accepted.
- `zig/src/families/glm/cuda_native.zig`: with `TF_GLM_PREFILL_REDUCE=ring`, the startup warm-up connects only neighbouring ranks.

On that ring the server also needs `NCCL_ALGO=Ring`, `TF_GLM_PREFILL_REDUCE=ring`, `TF_ROCE=0`, `NCCL_RUNTIME_CONNECT=1` and `NCCL_CUMEM_ENABLE=1`.

Licensed as upstream: Apache-2.0 (`LICENSE`, `NOTICE`).
