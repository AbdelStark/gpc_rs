//! GPC-OPT: Gradient-based trajectory optimization.
//!
//! Warm-starts from a policy sample and iteratively refines the action
//! sequence using gradients of the reward through the world model.

use burn::prelude::*;

use gpc_core::Result;
use gpc_core::traits::{Evaluator, Policy, RewardFunction, WorldModel};

/// GPC-OPT evaluator.
///
/// Optimizes action sequences by computing reward gradients through
/// the differentiable world model. Uses Adam-style updates.
pub struct GpcOpt<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_opt_steps: usize,
    learning_rate: f32,
    _backend: core::marker::PhantomData<B>,
}

/// Builder for GPC-OPT.
pub struct GpcOptBuilder<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_opt_steps: usize,
    learning_rate: f32,
    _backend: core::marker::PhantomData<B>,
}

impl<B, P, W, R> GpcOptBuilder<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    /// Create a new GPC-OPT builder.
    pub fn new(policy: P, world_model: W, reward_fn: R) -> Self {
        Self {
            policy,
            world_model,
            reward_fn,
            num_opt_steps: 25,
            learning_rate: 1e-2,
            _backend: core::marker::PhantomData,
        }
    }

    /// Set the number of optimization steps.
    pub fn num_opt_steps(mut self, steps: usize) -> Self {
        self.num_opt_steps = steps;
        self
    }

    /// Set the optimization learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Build the GPC-OPT evaluator.
    pub fn build(self) -> GpcOpt<B, P, W, R> {
        GpcOpt {
            policy: self.policy,
            world_model: self.world_model,
            reward_fn: self.reward_fn,
            num_opt_steps: self.num_opt_steps,
            learning_rate: self.learning_rate,
            _backend: core::marker::PhantomData,
        }
    }
}

impl<B, P, W, R> GpcOpt<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    /// Optimize an action sequence using finite-difference gradient estimation.
    ///
    /// Since Burn's NdArray backend doesn't support autodiff natively for
    /// arbitrary operations, we use finite differences as a fallback for
    /// gradient estimation. When using the Autodiff backend, this can be
    /// replaced with true backpropagation.
    fn optimize_actions(
        &self,
        initial_actions: Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>> {
        let [batch_size, horizon, action_dim] = initial_actions.dims();
        let mut actions = initial_actions;

        let epsilon = 1e-3_f32;

        for _step in 0..self.num_opt_steps {
            // Compute current reward
            let predicted_states = self.world_model.rollout(current_state, &actions)?;
            let current_reward = self.reward_fn.compute_reward(&predicted_states)?;
            let current_reward_f32: f32 = current_reward.clone().into_scalar().elem();

            // Estimate gradient for each action dimension using finite differences
            let actions_data: Vec<f32> = actions
                .clone()
                .reshape([batch_size * horizon * action_dim])
                .into_data()
                .to_vec()
                .map_err(|e| {
                    gpc_core::GpcError::Evaluation(format!(
                        "failed to extract action tensor data: {e:?}"
                    ))
                })?;
            let mut grad_data = vec![0.0_f32; actions_data.len()];

            for i in 0..actions_data.len() {
                let mut perturbed = actions_data.clone();
                perturbed[i] += epsilon;

                let device = actions.device();
                let perturbed_tensor = Tensor::<B, 1>::from_floats(perturbed.as_slice(), &device)
                    .reshape([batch_size, horizon, action_dim]);

                let perturbed_states =
                    self.world_model.rollout(current_state, &perturbed_tensor)?;
                let perturbed_reward = self.reward_fn.compute_reward(&perturbed_states)?;
                let perturbed_f32: f32 = perturbed_reward.into_scalar().elem();

                grad_data[i] = (perturbed_f32 - current_reward_f32) / epsilon;
            }

            // Update actions: a = a + lr * grad (gradient ascent)
            let device = actions.device();
            let grad_tensor = Tensor::<B, 1>::from_floats(grad_data.as_slice(), &device)
                .reshape([batch_size, horizon, action_dim]);

            actions = actions + grad_tensor * self.learning_rate;
        }

        Ok(actions)
    }
}

impl<B, P, W, R> Evaluator<B> for GpcOpt<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    fn select_action(
        &self,
        obs_history: &Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>> {
        // 1. Warm-start: sample initial trajectory from policy
        let initial_actions = self.policy.sample(obs_history, device)?;

        // 2. Optimize via gradient ascent through world model
        let optimized = self.optimize_actions(initial_actions, current_state)?;

        Ok(optimized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    struct MockPolicy {
        action_dim: usize,
        pred_horizon: usize,
    }

    impl Policy<TestBackend> for MockPolicy {
        fn sample(
            &self,
            _obs: &Tensor<TestBackend, 3>,
            device: &<TestBackend as Backend>::Device,
        ) -> Result<Tensor<TestBackend, 3>> {
            Ok(Tensor::zeros(
                [1, self.pred_horizon, self.action_dim],
                device,
            ))
        }

        fn sample_k(
            &self,
            _obs: &Tensor<TestBackend, 3>,
            k: usize,
            device: &<TestBackend as Backend>::Device,
        ) -> Result<Tensor<TestBackend, 3>> {
            Ok(Tensor::zeros(
                [k, self.pred_horizon, self.action_dim],
                device,
            ))
        }
    }

    struct MockWorldModel;

    impl WorldModel<TestBackend> for MockWorldModel {
        fn predict_next_state(
            &self,
            state: &Tensor<TestBackend, 2>,
            _action: &Tensor<TestBackend, 2>,
        ) -> Result<Tensor<TestBackend, 2>> {
            Ok(state.clone())
        }
    }

    struct MockReward;

    impl RewardFunction<TestBackend> for MockReward {
        fn compute_reward(
            &self,
            predicted_states: &Tensor<TestBackend, 3>,
        ) -> Result<Tensor<TestBackend, 1>> {
            let [batch, _, _] = predicted_states.dims();
            let device = predicted_states.device();
            Ok(Tensor::zeros([batch], &device))
        }
    }

    #[test]
    fn test_gpc_opt_output_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let policy = MockPolicy {
            action_dim: 2,
            pred_horizon: 4,
        };
        let world_model = MockWorldModel;
        let reward_fn = MockReward;

        let evaluator = GpcOptBuilder::new(policy, world_model, reward_fn)
            .num_opt_steps(2)
            .learning_rate(0.01)
            .build();

        let obs = Tensor::<TestBackend, 3>::zeros([1, 2, 5], &device);
        let state = Tensor::<TestBackend, 2>::zeros([1, 5], &device);

        let best_action = evaluator.select_action(&obs, &state, &device).unwrap();
        assert_eq!(best_action.dims(), [1, 4, 2]);
    }
}
