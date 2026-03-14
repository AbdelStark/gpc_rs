---
name: model-porting
description: Guide for porting PyTorch models to Burn/Tract — weight mapping, architecture translation, checkpoint loading, ONNX conversion. Activate when implementing models from the Python reference, loading pretrained weights, or converting between formats.
prerequisites: gpc-compat crate, tract-onnx
---

# Model Porting Guide

<purpose>
Systematic approach to porting the Python GPC reference implementation to Rust.
Covers architecture translation, weight loading, and validation.
</purpose>

<context>
— Reference implementation: https://github.com/han20192019/gpc_code (PyTorch)
— Two independent models to port: diffusion policy + world model.
— Burn for training-capable models, Tract for inference-only ONNX models.
— Weight format: PyTorch (.pt/.pth) → SafeTensors/ONNX → Burn/Tract.
</context>

<procedure>
1. **Read** the Python source for the target model. Note layer types, dimensions, activations.
2. **Map** PyTorch layers to Burn equivalents:

   | PyTorch               | Burn                          |
   |-----------------------|-------------------------------|
   | `nn.Linear`           | `nn::Linear`                  |
   | `nn.Conv2d`           | `nn::conv::Conv2d`            |
   | `nn.LayerNorm`        | `nn::LayerNorm`               |
   | `nn.BatchNorm2d`      | `nn::BatchNorm`               |
   | `nn.MultiheadAttention` | Custom (see burn attention)  |
   | `nn.TransformerEncoder` | Custom transformer blocks    |
   | `nn.Dropout`          | `nn::Dropout`                 |
   | `nn.GELU`             | `burn::tensor::activation::gelu` |
   | `nn.ReLU`             | `burn::tensor::activation::relu` |

3. **Implement** the Burn model following burn-patterns.md skill.
4. **Create** a weight mapping in gpc-compat for key remapping.
5. **Validate** by comparing outputs on identical inputs (use small test tensors).
6. **Test** with `cargo test` using NdArray backend.
</procedure>

<patterns>
<do>
— Start with the smallest submodule and work outward.
— Compare intermediate outputs (not just final output) during validation.
— Document the PyTorch-to-Burn key mapping for each model.
— Use `Tensor::from_floats` with known values for validation inputs.
</do>
<dont>
— Don't port the training loop first — port the model architecture, then training.
— Don't assume PyTorch default values match Burn defaults — check each parameter.
— Don't skip the validation step — silent numerical drift compounds.
— Don't try to port custom CUDA kernels — find Burn/ndarray equivalents.
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output values differ significantly | Weight key mismatch | Print and compare state_dict keys |
| Shape mismatch on load | Transposed weights | PyTorch Linear stores [out, in], check Burn convention |
| ONNX load fails in Tract | Unsupported op | Check tract supported ops, may need op decomposition |

</troubleshooting>

<references>
— gpc-compat/src/lib.rs: Checkpoint loading module
— Reference impl: https://github.com/han20192019/gpc_code
— Burn record format: https://burn.dev/docs
</references>
