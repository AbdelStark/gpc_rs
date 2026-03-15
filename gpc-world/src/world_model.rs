//! State-based world model with single-step and multi-step rollout.

use burn::prelude::*;

use gpc_core::traits::WorldModel;

use crate::dynamics::{DynamicsNetwork, DynamicsNetworkConfig};

/// State-based world model.
///
/// Uses a dynamics network to predict state deltas, then applies
/// residual prediction: `next_state = state + delta`.
#[derive(Module, Debug)]
pub struct StateWorldModel<B: Backend> {
    dynamics: DynamicsNetwork<B>,
}

/// Configuration for the state world model.
#[derive(Config, Debug)]
pub struct StateWorldModelConfig {
    /// State dimensionality.
    pub state_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Hidden dimension for dynamics network.
    #[config(default = 256)]
    pub hidden_dim: usize,
    /// Number of residual layers.
    #[config(default = 4)]
    pub num_layers: usize,
}

impl StateWorldModelConfig {
    /// Initialize the world model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StateWorldModel<B> {
        let dynamics_config = DynamicsNetworkConfig {
            state_dim: self.state_dim,
            action_dim: self.action_dim,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
        };

        StateWorldModel {
            dynamics: dynamics_config.init(device),
        }
    }

    /// Create from a [`gpc_core::config::WorldModelConfig`].
    pub fn from_world_model_config(config: &gpc_core::config::WorldModelConfig) -> Self {
        Self {
            state_dim: config.state_dim,
            action_dim: config.action_dim,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
        }
    }
}

impl<B: Backend> StateWorldModel<B> {
    /// Predict single-step with residual connection.
    pub fn predict_step(&self, state: &Tensor<B, 2>, action: &Tensor<B, 2>) -> Tensor<B, 2> {
        let delta = self.dynamics.forward(state.clone(), action.clone());
        state.clone() + delta
    }

    /// Compute the dynamics network output (delta) for training loss.
    pub fn predict_delta(&self, state: &Tensor<B, 2>, action: &Tensor<B, 2>) -> Tensor<B, 2> {
        self.dynamics.forward(state.clone(), action.clone())
    }
}

impl<B: Backend> WorldModel<B> for StateWorldModel<B> {
    fn predict_next_state(
        &self,
        state: &Tensor<B, 2>,
        action: &Tensor<B, 2>,
    ) -> gpc_core::Result<Tensor<B, 2>> {
        Ok(self.predict_step(state, action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use gpc_core::traits::WorldModel;

    type TestBackend = NdArray;

    #[test]
    fn test_world_model_single_step() {
        let device = <TestBackend as Backend>::Device::default();

        let config = StateWorldModelConfig {
            state_dim: 10,
            action_dim: 2,
            hidden_dim: 32,
            num_layers: 2,
        };

        let model = config.init::<TestBackend>(&device);

        let state = Tensor::<TestBackend, 2>::zeros([2, 10], &device);
        let action = Tensor::<TestBackend, 2>::zeros([2, 2], &device);

        let next_state = model.predict_next_state(&state, &action).unwrap();
        assert_eq!(next_state.dims(), [2, 10]);
    }

    #[test]
    fn test_world_model_rollout() {
        let device = <TestBackend as Backend>::Device::default();

        let config = StateWorldModelConfig {
            state_dim: 10,
            action_dim: 2,
            hidden_dim: 32,
            num_layers: 2,
        };

        let model = config.init::<TestBackend>(&device);

        let state = Tensor::<TestBackend, 2>::zeros([2, 10], &device);
        let actions = Tensor::<TestBackend, 3>::zeros([2, 8, 2], &device);

        let trajectory = model.rollout(&state, &actions).unwrap();
        assert_eq!(trajectory.dims(), [2, 8, 10]);
    }

    #[test]
    fn test_predict_delta_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = StateWorldModelConfig {
            state_dim: 5,
            action_dim: 3,
            hidden_dim: 16,
            num_layers: 1,
        };

        let model = config.init::<TestBackend>(&device);

        let state = Tensor::<TestBackend, 2>::zeros([4, 5], &device);
        let action = Tensor::<TestBackend, 2>::zeros([4, 3], &device);

        let delta = model.predict_delta(&state, &action);
        assert_eq!(delta.dims(), [4, 5]);
    }
}
