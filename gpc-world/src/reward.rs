//! Reward functions for trajectory evaluation.

use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

use gpc_core::traits::RewardFunction;

/// L2 distance-based reward function.
///
/// Computes negative L2 distance between predicted final state and
/// a goal state. Higher reward = closer to goal.
#[derive(Module, Debug)]
pub struct L2RewardFunction<B: Backend> {
    /// Goal state `[state_dim]`.
    goal: Tensor<B, 1>,
}

/// Configuration for the L2 reward function.
#[derive(Config, Debug)]
pub struct L2RewardFunctionConfig {
    /// State dimensionality.
    pub state_dim: usize,
}

impl L2RewardFunctionConfig {
    /// Initialize with a zero goal (to be set later).
    pub fn init<B: Backend>(&self, device: &B::Device) -> L2RewardFunction<B> {
        L2RewardFunction {
            goal: Tensor::zeros([self.state_dim], device),
        }
    }
}

impl<B: Backend> L2RewardFunction<B> {
    /// Set the goal state.
    pub fn set_goal(&mut self, goal: Tensor<B, 1>) {
        self.goal = goal;
    }

    /// Create with a specific goal.
    pub fn with_goal(mut self, goal: Tensor<B, 1>) -> Self {
        self.goal = goal;
        self
    }
}

impl<B: Backend> RewardFunction<B> for L2RewardFunction<B> {
    fn compute_reward(&self, predicted_states: &Tensor<B, 3>) -> gpc_core::Result<Tensor<B, 1>> {
        let [batch_size, horizon, state_dim] = predicted_states.dims();

        // Extract final predicted state
        let final_state = predicted_states
            .clone()
            .slice([0..batch_size, (horizon - 1)..horizon, 0..state_dim])
            .reshape([batch_size, state_dim]);

        // Expand goal to batch size
        let goal = self
            .goal
            .clone()
            .reshape([1, state_dim])
            .repeat_dim(0, batch_size);

        // Negative L2 distance (higher = closer to goal)
        let diff = final_state - goal;
        let sq_dist = (diff.clone() * diff).sum_dim(1).squeeze_dim(1);
        let reward = sq_dist.sqrt().neg(); // -sqrt(sum((s - g)^2))

        Ok(reward)
    }
}

/// Learned MLP-based reward predictor.
///
/// A neural network that maps state sequences to scalar rewards,
/// trained on task-specific reward labels.
#[derive(Module, Debug)]
pub struct LearnedRewardFunction<B: Backend> {
    encoder: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Gelu,
    #[module(skip)]
    state_dim: usize,
}

/// Configuration for the learned reward function.
#[derive(Config, Debug)]
pub struct LearnedRewardFunctionConfig {
    /// State dimensionality.
    pub state_dim: usize,
    /// Prediction horizon (number of timesteps).
    pub horizon: usize,
    /// Hidden layer dimension.
    #[config(default = 128)]
    pub hidden_dim: usize,
}

impl LearnedRewardFunctionConfig {
    /// Initialize the learned reward function.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LearnedRewardFunction<B> {
        let input_dim = self.state_dim * self.horizon;

        LearnedRewardFunction {
            encoder: LinearConfig::new(input_dim, self.hidden_dim).init(device),
            hidden: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            output: LinearConfig::new(self.hidden_dim, 1).init(device),
            activation: Gelu::new(),
            state_dim: self.state_dim,
        }
    }
}

impl<B: Backend> LearnedRewardFunction<B> {
    /// Forward pass computing reward from state sequence.
    pub fn forward(&self, predicted_states: &Tensor<B, 3>) -> Tensor<B, 1> {
        let [batch_size, horizon, _] = predicted_states.dims();

        // Flatten state sequence
        let flat = predicted_states
            .clone()
            .reshape([batch_size, horizon * self.state_dim]);

        let h = self.encoder.forward(flat);
        let h = self.activation.forward(h);
        let h = self.hidden.forward(h);
        let h = self.activation.forward(h);
        let out = self.output.forward(h);

        out.reshape([batch_size])
    }
}

impl<B: Backend> RewardFunction<B> for LearnedRewardFunction<B> {
    fn compute_reward(&self, predicted_states: &Tensor<B, 3>) -> gpc_core::Result<Tensor<B, 1>> {
        Ok(self.forward(predicted_states))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_l2_reward_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = L2RewardFunctionConfig { state_dim: 10 };
        let reward_fn = config.init::<TestBackend>(&device);

        let states = Tensor::<TestBackend, 3>::zeros([4, 8, 10], &device);
        let rewards = reward_fn.compute_reward(&states).unwrap();
        assert_eq!(rewards.dims(), [4]);
    }

    #[test]
    fn test_l2_reward_closer_is_higher() {
        let device = <TestBackend as Backend>::Device::default();

        let config = L2RewardFunctionConfig { state_dim: 2 };
        let reward_fn = config.init::<TestBackend>(&device);

        // State close to goal (0, 0)
        let close_states = Tensor::<TestBackend, 3>::from_floats([[[0.1, 0.1]]], &device);
        // State far from goal
        let far_states = Tensor::<TestBackend, 3>::from_floats([[[10.0, 10.0]]], &device);

        let close_reward = reward_fn.compute_reward(&close_states).unwrap();
        let far_reward = reward_fn.compute_reward(&far_states).unwrap();

        let close_val = close_reward.into_scalar();
        let far_val = far_reward.into_scalar();

        assert!(
            close_val > far_val,
            "Closer state should have higher reward: close={close_val}, far={far_val}"
        );
    }

    #[test]
    fn test_learned_reward_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = LearnedRewardFunctionConfig {
            state_dim: 10,
            horizon: 8,
            hidden_dim: 32,
        };

        let reward_fn = config.init::<TestBackend>(&device);

        let states = Tensor::<TestBackend, 3>::zeros([4, 8, 10], &device);
        let rewards = reward_fn.compute_reward(&states).unwrap();
        assert_eq!(rewards.dims(), [4]);
    }
}
