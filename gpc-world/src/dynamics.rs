//! Dynamics network for single-step state prediction.

use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

/// Configuration for the dynamics network.
#[derive(Config, Debug)]
pub struct DynamicsNetworkConfig {
    /// State dimensionality.
    pub state_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Hidden layer dimension.
    #[config(default = 256)]
    pub hidden_dim: usize,
    /// Number of residual hidden layers.
    #[config(default = 4)]
    pub num_layers: usize,
}

/// A residual MLP block for the dynamics network.
#[derive(Module, Debug)]
pub struct DynamicsResBlock<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Gelu,
}

/// Configuration for a dynamics residual block.
#[derive(Config, Debug)]
pub struct DynamicsResBlockConfig {
    pub hidden_dim: usize,
}

impl DynamicsResBlockConfig {
    /// Initialize a residual block.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicsResBlock<B> {
        DynamicsResBlock {
            linear1: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> DynamicsResBlock<B> {
    /// Forward pass with residual connection.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear1.forward(x.clone());
        let h = self.activation.forward(h);
        let h = self.linear2.forward(h);
        x + h
    }
}

/// MLP-based dynamics network for state prediction.
///
/// Predicts state deltas: `next_state = state + f(state, action)`.
/// Uses residual learning for stable multi-step rollouts.
#[derive(Module, Debug)]
pub struct DynamicsNetwork<B: Backend> {
    encoder: Linear<B>,
    blocks: Vec<DynamicsResBlock<B>>,
    decoder: Linear<B>,
    activation: Gelu,
}

impl DynamicsNetworkConfig {
    /// Initialize the dynamics network.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicsNetwork<B> {
        let input_dim = self.state_dim + self.action_dim;

        let blocks = (0..self.num_layers)
            .map(|_| {
                DynamicsResBlockConfig {
                    hidden_dim: self.hidden_dim,
                }
                .init(device)
            })
            .collect();

        DynamicsNetwork {
            encoder: LinearConfig::new(input_dim, self.hidden_dim).init(device),
            blocks,
            decoder: LinearConfig::new(self.hidden_dim, self.state_dim).init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> DynamicsNetwork<B> {
    /// Predict state delta given current state and action.
    ///
    /// # Arguments
    /// * `state` - Current state `[batch, state_dim]`
    /// * `action` - Action `[batch, action_dim]`
    ///
    /// # Returns
    /// State delta `[batch, state_dim]` (add to current state for next state)
    pub fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate state and action
        let input = Tensor::cat(vec![state, action], 1);

        // Encode
        let mut h = self.encoder.forward(input);
        h = self.activation.forward(h);

        // Residual blocks
        for block in &self.blocks {
            h = block.forward(h);
        }

        // Decode to state delta
        self.decoder.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_dynamics_network_output_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DynamicsNetworkConfig {
            state_dim: 20,
            action_dim: 2,
            hidden_dim: 64,
            num_layers: 2,
        };

        let network = config.init::<TestBackend>(&device);

        let state = Tensor::<TestBackend, 2>::zeros([4, 20], &device);
        let action = Tensor::<TestBackend, 2>::zeros([4, 2], &device);

        let delta = network.forward(state, action);
        assert_eq!(delta.dims(), [4, 20]);
    }

    #[test]
    fn test_dynamics_zero_input_finite_output() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DynamicsNetworkConfig {
            state_dim: 10,
            action_dim: 2,
            hidden_dim: 32,
            num_layers: 2,
        };

        let network = config.init::<TestBackend>(&device);

        let state = Tensor::<TestBackend, 2>::zeros([1, 10], &device);
        let action = Tensor::<TestBackend, 2>::zeros([1, 2], &device);

        let delta = network.forward(state, action);
        let max_val = delta.abs().max().into_scalar();
        assert!(max_val.is_finite(), "Output should be finite");
    }
}
