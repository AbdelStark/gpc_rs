//! GPC-OPT: Gradient-based trajectory optimization.
//!
//! Warm-starts from a policy sample and iteratively refines the action
//! sequence using gradients of the reward through the world model.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use gpc_core::Result;
use gpc_core::traits::{Evaluator, Policy, RewardFunction, WorldModel};

/// Trace for a single optimization step in GPC-OPT.
pub struct GpcOptStepTrace<B: Backend> {
    /// Zero-based optimization step index.
    pub step_index: usize,
    /// Action sequence evaluated at this step.
    pub actions: Tensor<B, 3>,
    /// Predicted rollout for the current actions.
    pub predicted_states: Tensor<B, 3>,
    /// Reward assigned to the predicted rollout.
    pub reward: Tensor<B, 1>,
    /// Gradient estimate used for the update.
    pub gradient: Tensor<B, 3>,
    /// Action sequence after the update is applied.
    pub updated_actions: Tensor<B, 3>,
}

/// Full trace for a GPC-OPT evaluation run.
pub struct GpcOptTrace<B: Backend> {
    /// Initial policy sample used to warm-start optimization.
    pub initial_actions: Tensor<B, 3>,
    /// Final optimized action sequence.
    pub optimized_actions: Tensor<B, 3>,
    /// Per-step optimization trace.
    pub step_traces: Vec<GpcOptStepTrace<B>>,
    /// Number of optimization steps configured for this run.
    pub num_opt_steps: usize,
    /// Learning rate used for updates.
    pub learning_rate: f32,
    /// Finite-difference epsilon used for gradient estimation.
    ///
    /// Exact autodiff evaluators set this to `0.0`.
    pub epsilon: f32,
}

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

/// Autodiff-enabled GPC-OPT.
///
/// This variant performs gradient ascent using true backpropagation through the
/// differentiable world model and reward function.
pub struct AutodiffGpcOpt<AD, P, W, R>
where
    AD: AutodiffBackend,
    P: Policy<AD>,
    W: WorldModel<AD>,
    R: RewardFunction<AD>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_opt_steps: usize,
    learning_rate: f32,
    _backend: core::marker::PhantomData<AD>,
}

/// Builder for autodiff GPC-OPT.
pub struct AutodiffGpcOptBuilder<AD, P, W, R>
where
    AD: AutodiffBackend,
    P: Policy<AD>,
    W: WorldModel<AD>,
    R: RewardFunction<AD>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_opt_steps: usize,
    learning_rate: f32,
    _backend: core::marker::PhantomData<AD>,
}

impl<AD, P, W, R> AutodiffGpcOptBuilder<AD, P, W, R>
where
    AD: AutodiffBackend,
    P: Policy<AD>,
    W: WorldModel<AD>,
    R: RewardFunction<AD>,
{
    /// Create a new autodiff GPC-OPT builder.
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

    /// Build the autodiff GPC-OPT evaluator.
    pub fn build(self) -> AutodiffGpcOpt<AD, P, W, R> {
        AutodiffGpcOpt {
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
    /// Select an action sequence and return the full inspection trace.
    pub fn select_action_with_trace(
        &self,
        obs_history: &Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Result<GpcOptTrace<B>> {
        // 1. Warm-start: sample initial trajectory from policy.
        let initial_actions = self.policy.sample(obs_history, device)?;

        // 2. Optimize via gradient ascent through world model.
        self.optimize_actions_with_trace(initial_actions, current_state)
    }

    /// Optimize an action sequence and return the full inspection trace.
    ///
    /// Since Burn's NdArray backend doesn't support autodiff natively for
    /// arbitrary operations, we use finite differences as a fallback for
    /// gradient estimation. When using the Autodiff backend, this can be
    /// replaced with true backpropagation.
    fn optimize_actions_with_trace(
        &self,
        initial_actions: Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
    ) -> Result<GpcOptTrace<B>> {
        let [batch_size, horizon, action_dim] = initial_actions.dims();
        let mut actions = initial_actions.clone();
        let mut step_traces = Vec::with_capacity(self.num_opt_steps);

        let epsilon = 1e-3_f32;

        for step_index in 0..self.num_opt_steps {
            let actions_before_update = actions.clone();

            // Compute current reward.
            let predicted_states = self
                .world_model
                .rollout(current_state, &actions_before_update)?;
            let current_reward = self.reward_fn.compute_reward(&predicted_states)?;
            let current_reward_f32: f32 = current_reward.clone().into_scalar().elem();

            // Estimate gradient for each action dimension using finite differences.
            let actions_data: Vec<f32> = actions_before_update
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

                let device = actions_before_update.device();
                let perturbed_tensor = Tensor::<B, 1>::from_floats(perturbed.as_slice(), &device)
                    .reshape([batch_size, horizon, action_dim]);

                let perturbed_states =
                    self.world_model.rollout(current_state, &perturbed_tensor)?;
                let perturbed_reward = self.reward_fn.compute_reward(&perturbed_states)?;
                let perturbed_f32: f32 = perturbed_reward.into_scalar().elem();

                grad_data[i] = (perturbed_f32 - current_reward_f32) / epsilon;
            }

            // Update actions: a = a + lr * grad (gradient ascent).
            let device = actions_before_update.device();
            let grad_tensor = Tensor::<B, 1>::from_floats(grad_data.as_slice(), &device)
                .reshape([batch_size, horizon, action_dim]);
            let updated_actions =
                actions_before_update.clone() + grad_tensor.clone() * self.learning_rate;

            step_traces.push(GpcOptStepTrace {
                step_index,
                actions: actions_before_update,
                predicted_states,
                reward: current_reward,
                gradient: grad_tensor,
                updated_actions: updated_actions.clone(),
            });

            actions = updated_actions;
        }

        Ok(GpcOptTrace {
            initial_actions,
            optimized_actions: actions,
            step_traces,
            num_opt_steps: self.num_opt_steps,
            learning_rate: self.learning_rate,
            epsilon,
        })
    }

    /// Optimize an action sequence using finite-difference gradient estimation.
    pub fn optimize_actions(
        &self,
        initial_actions: Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>> {
        Ok(self
            .optimize_actions_with_trace(initial_actions, current_state)?
            .optimized_actions)
    }

    /// Optimize an action sequence and return the full inspection trace.
    pub fn optimize_actions_trace(
        &self,
        initial_actions: Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
    ) -> Result<GpcOptTrace<B>> {
        self.optimize_actions_with_trace(initial_actions, current_state)
    }
}

impl<AD, P, W, R> AutodiffGpcOpt<AD, P, W, R>
where
    AD: AutodiffBackend,
    P: Policy<AD>,
    W: WorldModel<AD>,
    R: RewardFunction<AD>,
{
    /// Select an action sequence and return the full inspection trace.
    pub fn select_action_with_trace(
        &self,
        obs_history: &Tensor<AD, 3>,
        current_state: &Tensor<AD, 2>,
        device: &AD::Device,
    ) -> Result<GpcOptTrace<AD>> {
        let initial_actions = self.policy.sample(obs_history, device)?;
        self.optimize_actions_with_trace(initial_actions, current_state)
    }

    /// Optimize an action sequence and return the full inspection trace.
    fn optimize_actions_with_trace(
        &self,
        initial_actions: Tensor<AD, 3>,
        current_state: &Tensor<AD, 2>,
    ) -> Result<GpcOptTrace<AD>> {
        let mut actions_inner = initial_actions.inner();
        let mut step_traces = Vec::with_capacity(self.num_opt_steps);

        let current_state_inner = current_state.clone().inner();
        let initial_actions = Tensor::<AD, 3>::from_inner(actions_inner.clone());

        for step_index in 0..self.num_opt_steps {
            let actions_before_update = actions_inner.clone();
            let tracked_actions =
                Tensor::<AD, 3>::from_inner(actions_before_update.clone()).require_grad();
            let current_state_ad = Tensor::<AD, 2>::from_inner(current_state_inner.clone());

            let predicted_states = self
                .world_model
                .rollout(&current_state_ad, &tracked_actions)?;
            let current_reward = self.reward_fn.compute_reward(&predicted_states)?;

            let objective = current_reward.clone().sum();
            let grads = objective.backward();
            let action_grad_inner = tracked_actions.grad(&grads).ok_or_else(|| {
                gpc_core::GpcError::Evaluation(
                    "failed to extract action gradients from autodiff pass".to_string(),
                )
            })?;

            let action_grad = Tensor::<AD, 3>::from_inner(action_grad_inner.clone());
            let updated_actions_inner =
                actions_inner.clone() + action_grad_inner * self.learning_rate;

            step_traces.push(GpcOptStepTrace {
                step_index,
                actions: Tensor::<AD, 3>::from_inner(actions_before_update.clone()),
                predicted_states,
                reward: current_reward,
                gradient: action_grad,
                updated_actions: Tensor::<AD, 3>::from_inner(updated_actions_inner.clone()),
            });

            actions_inner = updated_actions_inner;
        }

        Ok(GpcOptTrace {
            initial_actions,
            optimized_actions: Tensor::<AD, 3>::from_inner(actions_inner),
            step_traces,
            num_opt_steps: self.num_opt_steps,
            learning_rate: self.learning_rate,
            epsilon: 0.0,
        })
    }

    /// Optimize an action sequence using backprop gradients.
    pub fn optimize_actions(
        &self,
        initial_actions: Tensor<AD, 3>,
        current_state: &Tensor<AD, 2>,
    ) -> Result<Tensor<AD, 3>> {
        Ok(self
            .optimize_actions_with_trace(initial_actions, current_state)?
            .optimized_actions)
    }

    /// Optimize an action sequence and return the full inspection trace.
    pub fn optimize_actions_trace(
        &self,
        initial_actions: Tensor<AD, 3>,
        current_state: &Tensor<AD, 2>,
    ) -> Result<GpcOptTrace<AD>> {
        self.optimize_actions_with_trace(initial_actions, current_state)
    }
}

impl<AD, P, W, R> Evaluator<AD> for AutodiffGpcOpt<AD, P, W, R>
where
    AD: AutodiffBackend,
    P: Policy<AD>,
    W: WorldModel<AD>,
    R: RewardFunction<AD>,
{
    fn select_action(
        &self,
        obs_history: &Tensor<AD, 3>,
        current_state: &Tensor<AD, 2>,
        device: &AD::Device,
    ) -> Result<Tensor<AD, 3>> {
        Ok(self
            .select_action_with_trace(obs_history, current_state, device)?
            .optimized_actions)
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
        let optimized = self.select_action_with_trace(obs_history, current_state, device)?;

        Ok(optimized.optimized_actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;
    type AutoBackend = Autodiff<TestBackend>;

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

    fn scalar_to_f32<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
        tensor.clone().into_scalar().elem()
    }

    #[derive(Debug, Clone)]
    struct DifferentiablePolicy {
        action_dim: usize,
        pred_horizon: usize,
    }

    impl<B: Backend> Policy<B> for DifferentiablePolicy {
        fn sample(
            &self,
            _obs: &Tensor<B, 3>,
            device: &<B as Backend>::Device,
        ) -> Result<Tensor<B, 3>> {
            Ok(Tensor::ones(
                [1, self.pred_horizon, self.action_dim],
                device,
            ))
        }

        fn sample_k(
            &self,
            _obs: &Tensor<B, 3>,
            k: usize,
            device: &<B as Backend>::Device,
        ) -> Result<Tensor<B, 3>> {
            Ok(Tensor::ones(
                [k, self.pred_horizon, self.action_dim],
                device,
            ))
        }
    }

    #[derive(Debug, Clone)]
    struct SummedNextStateWorldModel;

    impl<B: Backend> WorldModel<B> for SummedNextStateWorldModel {
        fn predict_next_state(
            &self,
            state: &Tensor<B, 2>,
            action: &Tensor<B, 2>,
        ) -> Result<Tensor<B, 2>> {
            Ok(state.clone() + action.clone())
        }
    }

    #[derive(Debug, Clone)]
    struct FinalStateSumReward;

    impl<B: Backend> RewardFunction<B> for FinalStateSumReward {
        fn compute_reward(&self, predicted_states: &Tensor<B, 3>) -> Result<Tensor<B, 1>> {
            let [batch_size, horizon, _state_dim] = predicted_states.dims();
            let final_state = predicted_states
                .clone()
                .slice([0..batch_size, (horizon - 1)..horizon, 0..1])
                .reshape([batch_size, 1]);
            Ok(final_state.sum_dim(1).squeeze_dim(1))
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

        let trace = evaluator
            .select_action_with_trace(&obs, &state, &device)
            .unwrap();
        assert_eq!(trace.initial_actions.dims(), [1, 4, 2]);
        assert_eq!(trace.optimized_actions.dims(), [1, 4, 2]);
        assert_eq!(trace.step_traces.len(), 2);
        assert_eq!(trace.num_opt_steps, 2);
        assert_eq!(trace.learning_rate, 0.01);
        assert_eq!(trace.epsilon, 1e-3_f32);
        assert_eq!(trace.step_traces[0].step_index, 0);
        assert_eq!(trace.step_traces[0].actions.dims(), [1, 4, 2]);
        assert_eq!(trace.step_traces[0].predicted_states.dims(), [1, 4, 5]);
        assert_eq!(trace.step_traces[0].reward.dims(), [1]);
        assert_eq!(trace.step_traces[0].gradient.dims(), [1, 4, 2]);
        assert_eq!(trace.step_traces[0].updated_actions.dims(), [1, 4, 2]);
    }

    #[test]
    fn test_autodiff_gpc_opt_improves_reward() {
        let device = <AutoBackend as Backend>::Device::default();

        let policy = DifferentiablePolicy {
            action_dim: 1,
            pred_horizon: 3,
        };
        let world_model = SummedNextStateWorldModel;
        let reward_fn = FinalStateSumReward;

        let evaluator =
            AutodiffGpcOptBuilder::new(policy.clone(), world_model.clone(), reward_fn.clone())
                .num_opt_steps(4)
                .learning_rate(0.15)
                .build();

        let obs = Tensor::<AutoBackend, 3>::zeros([1, 2, 1], &device);
        let state = Tensor::<AutoBackend, 2>::zeros([1, 1], &device);

        let trace = evaluator
            .select_action_with_trace(&obs, &state, &device)
            .unwrap();
        let initial_reward = scalar_to_f32(&trace.step_traces[0].reward);
        let final_reward = scalar_to_f32(
            &reward_fn
                .compute_reward(
                    &world_model
                        .rollout(&state, &trace.optimized_actions)
                        .unwrap(),
                )
                .unwrap(),
        );

        assert_eq!(trace.initial_actions.dims(), [1, 3, 1]);
        assert_eq!(trace.num_opt_steps, 4);
        assert_eq!(trace.step_traces.len(), 4);
        assert!(
            trace
                .step_traces
                .iter()
                .all(|step| step.gradient.dims() == [1, 3, 1]),
            "gradient shape should match actions"
        );
        assert_eq!(trace.step_traces[0].reward.dims(), [1]);
        assert!(
            final_reward > initial_reward,
            "optimization should improve reward"
        );
    }
}
