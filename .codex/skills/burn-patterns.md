---
name: burn-patterns
description: Patterns for using the Burn ML framework — Module trait, Backend generics, tensor operations, model configuration. Activate when implementing neural network layers, defining model architectures, or working with tensors.
prerequisites: burn crate in dependencies
---

# Burn Framework Patterns

<purpose>
Guide for implementing ML models using Burn's backend-generic architecture.
Follows patterns established in the jepa-rs sister project.
</purpose>

<context>
— Burn 0.16 with NdArray (CPU) and WGPU (GPU) backends.
— All model code must be generic over `B: Backend`.
— Use `burn-ndarray` backend in tests for determinism.
— Config structs use `#[derive(Config)]` from Burn.
</context>

<procedure>
1. Define config struct with `#[derive(Config)]`.
2. Define model struct with `#[derive(Module, Debug)]`.
3. Implement `init` method on config to construct model.
4. Implement `forward` method on model (not a trait — convention).
5. Write tests using `burn_ndarray::NdArray` backend.
</procedure>

<patterns>
<do>

```rust
use burn::prelude::*;

// Config for hyperparameters
#[derive(Config)]
pub struct MyLayerConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

// Model struct — generic over Backend
#[derive(Module, Debug)]
pub struct MyLayer<B: Backend> {
    linear: nn::Linear<B>,
    norm: nn::LayerNorm<B>,
    dropout: nn::Dropout,
}

impl MyLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyLayer<B> {
        MyLayer {
            linear: nn::LinearConfig::new(self.d_model, self.d_model).init(device),
            norm: nn::LayerNormConfig::new(self.d_model).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> MyLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.linear.forward(x.clone());
        let h = self.dropout.forward(h);
        self.norm.forward(h + x)
    }
}
```

— Always pass `device: &B::Device` to init methods.
— Use `Tensor<B, N>` where N is the rank (number of dimensions).
— Clone tensors explicitly when used multiple times (Burn tensors are move-by-default).

</do>
<dont>
— Don't hardcode backend: `MyLayer<NdArray>` — always use generic `B: Backend`.
— Don't use `Tensor::from_data` without specifying device — use `.to_device(device)`.
— Don't implement `Module` manually — use `#[derive(Module)]`.
— Don't store `Device` in module structs — pass it to methods that need it.
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `the trait Backend is not implemented` | Missing generic parameter | Add `<B: Backend>` to function/struct |
| `cannot move out of borrowed content` | Tensor consumed twice | Clone the tensor before second use |
| `expected Tensor<B, 3>, found Tensor<B, 2>` | Shape rank mismatch | Use `.unsqueeze()` or `.reshape()` |
| Module fields not serializable | Missing `#[derive(Module)]` | Add derive macro to struct |

</troubleshooting>

<references>
— gpc-core/src/lib.rs: Core traits and error types
— Burn docs: https://burn.dev/docs
— jepa-rs: https://github.com/AbdelStark/jepa-rs (canonical patterns)
</references>
