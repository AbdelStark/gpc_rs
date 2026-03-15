//! Conditional denoising network for diffusion policy.
//!
//! A residual MLP that predicts noise given noisy actions,
//! observation conditioning, and timestep embedding.

use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

/// Configuration for the denoising network.
#[derive(Config, Debug)]
pub struct DenoisingNetworkConfig {
    /// Input dimension (pred_horizon * action_dim).
    pub input_dim: usize,
    /// Conditioning dimension (obs_horizon * obs_dim).
    pub cond_dim: usize,
    /// Hidden layer dimension.
    #[config(default = 256)]
    pub hidden_dim: usize,
    /// Timestep embedding dimension.
    #[config(default = 128)]
    pub time_embed_dim: usize,
    /// Number of residual blocks.
    #[config(default = 3)]
    pub num_blocks: usize,
}

/// A single residual block with conditioning.
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    cond_proj: Linear<B>,
    time_proj: Linear<B>,
    activation: Gelu,
}

/// Configuration for a residual block.
#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    pub hidden_dim: usize,
    pub cond_dim: usize,
    pub time_dim: usize,
}

impl ResidualBlockConfig {
    /// Initialize a residual block.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            linear1: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            cond_proj: LinearConfig::new(self.cond_dim, self.hidden_dim).init(device),
            time_proj: LinearConfig::new(self.time_dim, self.hidden_dim).init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    /// Forward pass through the residual block.
    ///
    /// # Arguments
    /// * `x` - Input `[batch, hidden_dim]`
    /// * `cond` - Conditioning `[batch, cond_dim]`
    /// * `time_emb` - Timestep embedding `[batch, time_dim]`
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        cond: &Tensor<B, 2>,
        time_emb: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let h = self.linear1.forward(x.clone());
        let h = h + self.cond_proj.forward(cond.clone());
        let h = h + self.time_proj.forward(time_emb.clone());
        let h = self.activation.forward(h);
        let h = self.linear2.forward(h);
        // Residual connection
        x + h
    }
}

/// Conditional denoising network for the diffusion policy.
///
/// Architecture: input projection → N residual blocks → output projection.
/// Each block receives observation conditioning and timestep embedding.
#[derive(Module, Debug)]
pub struct DenoisingNetwork<B: Backend> {
    input_proj: Linear<B>,
    blocks: Vec<ResidualBlock<B>>,
    output_proj: Linear<B>,
    time_mlp: Linear<B>,
    time_mlp2: Linear<B>,
    time_activation: Gelu,
}

impl DenoisingNetworkConfig {
    /// Initialize the denoising network.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DenoisingNetwork<B> {
        let blocks = (0..self.num_blocks)
            .map(|_| {
                ResidualBlockConfig {
                    hidden_dim: self.hidden_dim,
                    cond_dim: self.hidden_dim,
                    time_dim: self.time_embed_dim,
                }
                .init(device)
            })
            .collect();

        DenoisingNetwork {
            input_proj: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            blocks,
            output_proj: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            time_mlp: LinearConfig::new(self.time_embed_dim, self.time_embed_dim).init(device),
            time_mlp2: LinearConfig::new(self.cond_dim, self.hidden_dim).init(device),
            time_activation: Gelu::new(),
        }
    }
}

impl<B: Backend> DenoisingNetwork<B> {
    /// Predict noise in the noisy action sequence.
    ///
    /// # Arguments
    /// * `noisy_actions` - Noisy flattened actions `[batch, input_dim]`
    /// * `cond` - Observation conditioning `[batch, cond_dim]`
    /// * `time_emb` - Sinusoidal timestep embedding `[batch, time_embed_dim]`
    ///
    /// # Returns
    /// Predicted noise `[batch, input_dim]`
    pub fn forward(
        &self,
        noisy_actions: Tensor<B, 2>,
        cond: Tensor<B, 2>,
        time_emb: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Project timestep embedding
        let time_emb = self.time_mlp.forward(time_emb);
        let time_emb = self.time_activation.forward(time_emb);

        // Project conditioning to hidden dim
        let cond_proj = self.time_mlp2.forward(cond);

        // Project input
        let mut h = self.input_proj.forward(noisy_actions);

        // Residual blocks
        for block in &self.blocks {
            h = block.forward(h, &cond_proj, &time_emb);
        }

        // Output projection
        self.output_proj.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_denoising_network_output_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DenoisingNetworkConfig {
            input_dim: 32, // 16 steps * 2 action_dim
            cond_dim: 40,  // 2 obs_horizon * 20 obs_dim
            hidden_dim: 64,
            time_embed_dim: 32,
            num_blocks: 2,
        };

        let network = config.init::<TestBackend>(&device);

        let batch_size = 4;
        let noisy = Tensor::<TestBackend, 2>::zeros([batch_size, 32], &device);
        let cond = Tensor::<TestBackend, 2>::zeros([batch_size, 40], &device);
        let time = Tensor::<TestBackend, 2>::zeros([batch_size, 32], &device);

        let output = network.forward(noisy, cond, time);
        assert_eq!(output.dims(), [batch_size, 32]);
    }

    #[test]
    fn test_residual_block_preserves_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let block = ResidualBlockConfig {
            hidden_dim: 64,
            cond_dim: 64,
            time_dim: 32,
        }
        .init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 2>::zeros([2, 64], &device);
        let cond = Tensor::<TestBackend, 2>::zeros([2, 64], &device);
        let time = Tensor::<TestBackend, 2>::zeros([2, 32], &device);

        let out = block.forward(x, &cond, &time);
        assert_eq!(out.dims(), [2, 64]);
    }
}
