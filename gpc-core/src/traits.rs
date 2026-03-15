//! Core traits defining the GPC component interfaces.

use burn::tensor::{Tensor, backend::Backend};

use crate::Result;

/// A generative action policy that produces candidate action sequences.
///
/// Given observation history, the policy generates action sequences
/// using a diffusion-based generative model (DDPM).
pub trait Policy<B: Backend> {
    /// Generate a single action sequence from observations.
    ///
    /// # Arguments
    /// * `obs_history` - Observation history `[batch_size, obs_horizon, obs_dim]`
    ///
    /// # Returns
    /// Action sequence `[batch_size, pred_horizon, action_dim]`
    fn sample(&self, obs_history: &Tensor<B, 3>, device: &B::Device) -> Result<Tensor<B, 3>>;

    /// Generate K candidate action sequences from observations.
    ///
    /// # Arguments
    /// * `obs_history` - Observation history `[batch_size, obs_horizon, obs_dim]`
    /// * `num_candidates` - Number of candidate trajectories K
    ///
    /// # Returns
    /// Candidates `[batch_size * num_candidates, pred_horizon, action_dim]`
    fn sample_k(
        &self,
        obs_history: &Tensor<B, 3>,
        num_candidates: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>>;
}

/// A predictive world model that forecasts future states.
///
/// Given current state and an action, predicts the next state.
/// Applied recursively for multi-step trajectory evaluation.
pub trait WorldModel<B: Backend> {
    /// Predict the next state given current state and action.
    ///
    /// # Arguments
    /// * `state` - Current state `[batch_size, state_dim]`
    /// * `action` - Action to apply `[batch_size, action_dim]`
    ///
    /// # Returns
    /// Predicted next state `[batch_size, state_dim]`
    fn predict_next_state(
        &self,
        state: &Tensor<B, 2>,
        action: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 2>>;

    /// Roll out the world model for a full action sequence.
    ///
    /// # Arguments
    /// * `initial_state` - Starting state `[batch_size, state_dim]`
    /// * `actions` - Action sequence `[batch_size, horizon, action_dim]`
    ///
    /// # Returns
    /// Predicted state sequence `[batch_size, horizon, state_dim]`
    fn rollout(
        &self,
        initial_state: &Tensor<B, 2>,
        actions: &Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let [batch_size, horizon, _action_dim] = actions.dims();
        let state_dim = initial_state.dims()[1];

        let mut states = Vec::with_capacity(horizon);
        let mut current_state = initial_state.clone();

        for t in 0..horizon {
            let action_t = actions
                .clone()
                .slice([0..batch_size, t..t + 1, 0.._action_dim]);
            let action_t = action_t.reshape([batch_size, _action_dim]);
            let next_state = self.predict_next_state(&current_state, &action_t)?;
            states.push(next_state.clone().reshape([batch_size, 1, state_dim]));
            current_state = next_state;
        }

        let result = Tensor::cat(states, 1);
        debug_assert_eq!(result.dims(), [batch_size, horizon, state_dim]);
        Ok(result)
    }
}

/// A reward function that scores predicted state trajectories.
pub trait RewardFunction<B: Backend> {
    /// Compute reward for predicted state sequences.
    ///
    /// # Arguments
    /// * `predicted_states` - State sequence `[batch_size, horizon, state_dim]`
    ///
    /// # Returns
    /// Reward scores `[batch_size]`
    fn compute_reward(&self, predicted_states: &Tensor<B, 3>) -> Result<Tensor<B, 1>>;
}

/// An evaluation strategy that combines policy + world model for action selection.
pub trait Evaluator<B: Backend> {
    /// Select the best action sequence given current observations and state.
    ///
    /// # Arguments
    /// * `obs_history` - Observation history `[1, obs_horizon, obs_dim]`
    /// * `current_state` - Current state `[1, state_dim]`
    ///
    /// # Returns
    /// Best action sequence `[1, pred_horizon, action_dim]`
    fn select_action(
        &self,
        obs_history: &Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    struct DummyWorldModel;

    impl WorldModel<TestBackend> for DummyWorldModel {
        fn predict_next_state(
            &self,
            state: &Tensor<TestBackend, 2>,
            action: &Tensor<TestBackend, 2>,
        ) -> Result<Tensor<TestBackend, 2>> {
            // Simple additive dynamics: next_state = state + action (padded/truncated)
            let [_batch_size, state_dim] = state.dims();
            let [_, action_dim] = action.dims();

            if action_dim == state_dim {
                Ok(state.clone() + action.clone())
            } else {
                // Just return state unchanged for dimension mismatch
                Ok(state.clone())
            }
        }
    }

    #[test]
    fn test_world_model_rollout() {
        let device = <TestBackend as Backend>::Device::default();
        let model = DummyWorldModel;

        let batch_size = 2;
        let state_dim = 3;
        let horizon = 4;
        let action_dim = 3;

        let initial_state = Tensor::<TestBackend, 2>::zeros([batch_size, state_dim], &device);
        let actions = Tensor::<TestBackend, 3>::ones([batch_size, horizon, action_dim], &device);

        let result = model.rollout(&initial_state, &actions).unwrap();
        assert_eq!(result.dims(), [batch_size, horizon, state_dim]);

        // After 4 steps of adding 1, final state should be [4, 4, 4]
        let final_state = result.clone().slice([0..1, 3..4, 0..3]).reshape([1, 3]);
        let expected = Tensor::<TestBackend, 2>::ones([1, state_dim], &device) * 4.0;
        let diff = (final_state - expected).abs().max().into_scalar();
        assert!(diff < 1e-5, "Expected final state ≈ 4.0, got diff={diff}");
    }
}
