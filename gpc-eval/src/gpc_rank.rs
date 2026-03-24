//! GPC-RANK: Trajectory ranking via world model scoring.

use burn::prelude::*;

use gpc_core::Result;
use gpc_core::traits::{Evaluator, Policy, RewardFunction, WorldModel};

/// Detailed trace for a single GPC-RANK evaluation.
pub struct GpcRankTrace<B: Backend> {
    /// Candidate action sequences sampled from the policy.
    pub candidate_actions: Tensor<B, 3>,
    /// Predicted rollouts for each candidate action sequence.
    pub predicted_states: Tensor<B, 3>,
    /// Scalar reward for each candidate trajectory.
    pub rewards: Tensor<B, 1>,
    /// Index of the winning candidate.
    pub best_index: usize,
    /// Winning action sequence.
    pub best_action: Tensor<B, 3>,
    /// Number of candidates evaluated.
    pub num_candidates: usize,
}

/// GPC-RANK evaluator.
///
/// Samples K candidate trajectories from the policy, scores them
/// using the world model + reward function, and selects the best.
pub struct GpcRank<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_candidates: usize,
    _backend: core::marker::PhantomData<B>,
}

/// Builder for GPC-RANK.
pub struct GpcRankBuilder<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    policy: P,
    world_model: W,
    reward_fn: R,
    num_candidates: usize,
    _backend: core::marker::PhantomData<B>,
}

impl<B, P, W, R> GpcRankBuilder<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    /// Create a new GPC-RANK builder.
    pub fn new(policy: P, world_model: W, reward_fn: R) -> Self {
        Self {
            policy,
            world_model,
            reward_fn,
            num_candidates: 100,
            _backend: core::marker::PhantomData,
        }
    }

    /// Set the number of candidate trajectories.
    pub fn num_candidates(mut self, k: usize) -> Self {
        self.num_candidates = k;
        self
    }

    /// Build the GPC-RANK evaluator.
    pub fn build(self) -> GpcRank<B, P, W, R> {
        GpcRank {
            policy: self.policy,
            world_model: self.world_model,
            reward_fn: self.reward_fn,
            num_candidates: self.num_candidates,
            _backend: core::marker::PhantomData,
        }
    }
}

impl<B, P, W, R> GpcRank<B, P, W, R>
where
    B: Backend,
    P: Policy<B>,
    W: WorldModel<B>,
    R: RewardFunction<B>,
{
    /// Evaluate candidates and return the full inspection trace.
    pub fn select_action_with_trace(
        &self,
        obs_history: &Tensor<B, 3>,
        current_state: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Result<GpcRankTrace<B>> {
        let k = self.num_candidates;

        // 1. Sample K candidate action sequences
        let candidate_actions = self.policy.sample_k(obs_history, k, device)?;
        let [_, horizon, action_dim] = candidate_actions.dims();

        // 2. Repeat current state K times
        let states_repeated = gpc_core::tensor_utils::repeat_batch_2d(current_state, k);

        // 3. Roll out world model for each candidate
        let predicted_states = self
            .world_model
            .rollout(&states_repeated, &candidate_actions)?;

        // 4. Score each trajectory
        let rewards = self.reward_fn.compute_reward(&predicted_states)?;

        // 5. Find the best trajectory (argmax over rewards)
        let best_idx: i64 = rewards.clone().argmax(0).into_scalar().elem();
        let best_index = best_idx as usize;

        // 6. Extract the best action sequence
        let best_action = candidate_actions.clone().slice([
            best_index..best_index + 1,
            0..horizon,
            0..action_dim,
        ]);

        Ok(GpcRankTrace {
            candidate_actions,
            predicted_states,
            rewards,
            best_index,
            best_action,
            num_candidates: k,
        })
    }
}

impl<B, P, W, R> Evaluator<B> for GpcRank<B, P, W, R>
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
        Ok(self
            .select_action_with_trace(obs_history, current_state, device)?
            .best_action)
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
    fn test_gpc_rank_output_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let policy = MockPolicy {
            action_dim: 2,
            pred_horizon: 8,
        };
        let world_model = MockWorldModel;
        let reward_fn = MockReward;

        let evaluator = GpcRankBuilder::new(policy, world_model, reward_fn)
            .num_candidates(10)
            .build();

        let obs = Tensor::<TestBackend, 3>::zeros([1, 2, 10], &device);
        let state = Tensor::<TestBackend, 2>::zeros([1, 10], &device);

        let best_action = evaluator.select_action(&obs, &state, &device).unwrap();
        assert_eq!(best_action.dims(), [1, 8, 2]);

        let trace = evaluator
            .select_action_with_trace(&obs, &state, &device)
            .unwrap();
        assert_eq!(trace.num_candidates, 10);
        assert_eq!(trace.candidate_actions.dims(), [10, 8, 2]);
        assert_eq!(trace.predicted_states.dims(), [10, 8, 10]);
        assert_eq!(trace.rewards.dims(), [10]);
        assert_eq!(trace.best_index, 0);
        assert_eq!(trace.best_action.dims(), [1, 8, 2]);
    }
}
