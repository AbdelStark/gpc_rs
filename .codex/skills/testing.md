---
name: testing
description: Testing strategies for Rust ML code — unit tests with NdArray backend, float comparisons with approx, property-based testing with proptest. Activate when writing tests, debugging test failures, or setting up test infrastructure.
prerequisites: approx, proptest, burn-ndarray in dev-dependencies
---

# Testing Strategies

<purpose>
Testing patterns for ML code in Rust. Covers unit tests, tensor validation,
property-based testing, and common pitfalls with floating point and backends.
</purpose>

<context>
— Use `burn-ndarray` (NdArray) backend in all tests — deterministic, no GPU.
— Use `approx::assert_relative_eq!` for float comparisons.
— Use `proptest` for testing invariants across random inputs.
— Tests live in `#[cfg(test)] mod tests` at bottom of each source file.
</context>

<procedure>
1. Write the test function with `#[test]` attribute.
2. Create a device: `let device = burn_ndarray::NdArrayDevice::Cpu;`
3. Initialize model/layer using config.
4. Create input tensors with known values.
5. Run forward pass and assert outputs.
6. Use `approx` for floating point assertions.
</procedure>

<patterns>
<do>

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn test_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn test_layer_output_shape() {
        let device = test_device();
        let config = MyLayerConfig::new(64, 8);
        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_values_in_range() {
        let device = test_device();
        let data = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let result = data.mean().into_scalar();
        assert_relative_eq!(result, 2.0, epsilon = 1e-6);
    }
}
```

— Define `type TestBackend = NdArray<f32>` at top of test module.
— Create a `test_device()` helper for consistent device creation.
— Test shapes first, then values — shape bugs are most common.
— Use `into_scalar()` to extract single values for assertion.

</do>
<dont>
— Don't use WGPU backend in tests — non-deterministic, requires GPU.
— Don't compare floats with `==` — use `approx` crate.
— Don't skip shape assertions — they catch the most bugs.
— Don't create large tensors in unit tests — use small dimensions (2-8).
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| Test hangs forever | WGPU backend waiting for GPU | Switch to NdArray backend |
| Float assertion fails | Accumulated FP error | Increase epsilon in assert_relative_eq |
| `cannot infer type` in test | Missing backend annotation | Add `::<TestBackend, N>` to tensor creation |

</troubleshooting>

<references>
— approx docs: https://docs.rs/approx
— proptest docs: https://docs.rs/proptest
</references>
