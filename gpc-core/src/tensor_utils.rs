//! Tensor utility functions for common operations.

use burn::tensor::{Tensor, backend::Backend};

/// Flatten the last two dimensions of a 3D tensor into a 2D tensor.
///
/// `[batch, seq, dim]` → `[batch, seq * dim]`
pub fn flatten_last_two<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch, seq, dim] = tensor.dims();
    tensor.reshape([batch, seq * dim])
}

/// Unflatten a 2D tensor into a 3D tensor.
///
/// `[batch, seq * dim]` → `[batch, seq, dim]`
pub fn unflatten<B: Backend>(tensor: Tensor<B, 2>, seq: usize, dim: usize) -> Tensor<B, 3> {
    let [batch, _] = tensor.dims();
    tensor.reshape([batch, seq, dim])
}

/// Compute the mean squared error between two tensors.
pub fn mse_loss<B: Backend, const D: usize>(
    prediction: &Tensor<B, D>,
    target: &Tensor<B, D>,
) -> Tensor<B, 1> {
    let diff = prediction.clone() - target.clone();
    let sq = diff.clone() * diff;
    sq.mean()
}

/// Repeat a tensor along the batch dimension K times.
///
/// `[batch, ...]` → `[batch * K, ...]` by repeating each sample K times.
pub fn repeat_batch<B: Backend>(tensor: &Tensor<B, 3>, k: usize) -> Tensor<B, 3> {
    let [batch, seq, dim] = tensor.dims();
    // Expand [batch, seq, dim] to [batch, k, seq, dim] then reshape
    let expanded = tensor
        .clone()
        .reshape([batch, 1, seq, dim])
        .repeat_dim(1, k);
    expanded.reshape([batch * k, seq, dim])
}

/// Repeat a 2D tensor along the batch dimension K times.
pub fn repeat_batch_2d<B: Backend>(tensor: &Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    let [batch, dim] = tensor.dims();
    let expanded = tensor.clone().reshape([batch, 1, dim]).repeat_dim(1, k);
    expanded.reshape([batch * k, dim])
}

/// Create a sinusoidal timestep embedding.
///
/// Maps integer timesteps to `embed_dim`-dimensional vectors using
/// sinusoidal positional encoding (same as Transformer PE).
pub fn timestep_embedding<B: Backend>(
    timesteps: &Tensor<B, 1>,
    embed_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half_dim = embed_dim / 2;
    let [batch] = timesteps.dims();

    // log(10000) / (half_dim - 1)
    let emb_scale = -(10000.0_f32.ln()) / (half_dim as f32 - 1.0).max(1.0);

    // Create frequency indices [0, 1, ..., half_dim-1]
    let freq_indices: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let freq_data = Tensor::<B, 1>::from_floats(freq_indices.as_slice(), device);

    // exp(i * emb_scale) for each frequency index
    let freqs = (freq_data * emb_scale).exp();

    // [batch, 1] * [1, half_dim] -> [batch, half_dim]
    let timesteps_f: Tensor<B, 2> = timesteps.clone().reshape([batch, 1]);
    let freqs_2d = freqs.reshape([1, half_dim]);
    let angles = timesteps_f * freqs_2d;

    // Concatenate sin and cos
    let sin_emb = angles.clone().sin();
    let cos_emb = angles.cos();

    Tensor::cat(vec![sin_emb, cos_emb], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let tensor = Tensor::<TestBackend, 3>::ones([2, 4, 8], &device);
        let flat = flatten_last_two(tensor.clone());
        assert_eq!(flat.dims(), [2, 32]);
        let unflat = unflatten(flat, 4, 8);
        assert_eq!(unflat.dims(), [2, 4, 8]);
    }

    #[test]
    fn test_mse_loss_zero_for_identical() {
        let device = <TestBackend as Backend>::Device::default();
        let a = Tensor::<TestBackend, 2>::ones([4, 8], &device);
        let b = a.clone();
        let loss = mse_loss(&a, &b);
        let val = loss.into_scalar();
        assert!(
            val < 1e-6,
            "MSE of identical tensors should be 0, got {val}"
        );
    }

    #[test]
    fn test_repeat_batch() {
        let device = <TestBackend as Backend>::Device::default();
        let tensor = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let repeated = repeat_batch(&tensor, 5);
        assert_eq!(repeated.dims(), [10, 3, 4]);
    }

    #[test]
    fn test_timestep_embedding_shape() {
        let device = <TestBackend as Backend>::Device::default();
        let timesteps = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 50.0, 99.0], &device);
        let emb = timestep_embedding(&timesteps, 64, &device);
        assert_eq!(emb.dims(), [4, 64]);
    }

    #[test]
    fn test_timestep_embedding_different_timesteps_produce_different_embeddings() {
        let device = <TestBackend as Backend>::Device::default();
        let t1 = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let t2 = Tensor::<TestBackend, 1>::from_floats([50.0], &device);
        let emb1 = timestep_embedding(&t1, 32, &device);
        let emb2 = timestep_embedding(&t2, 32, &device);
        let diff = (emb1 - emb2).abs().sum().into_scalar();
        assert!(
            diff > 1.0,
            "Different timesteps should produce different embeddings"
        );
    }
}
