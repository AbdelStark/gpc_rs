mod arena;
mod dataset;
mod types;

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

use gpc_core::config::{NoiseScheduleConfig, PolicyConfig};
use gpc_core::noise::DdpmSchedule;
use gpc_core::tensor_utils;
use gpc_core::traits::{Policy, RewardFunction, WorldModel};
use gpc_policy::{DiffusionPolicy, DiffusionPolicyConfig};
use gpc_world::world_model::StateWorldModelConfig;

use crate::arena::{
    ACTION_DIM, EXECUTION_STRIDE, OBS_HORIZON, PRED_HORIZON, RobotState, STATE_DIM, apply_action,
    build_training_dataset, effector_from_slice, forward_kinematics, goal_distance_from_slice,
    min_clearance_from_slice, mission_state, preset_missions, state_to_vec,
};
use crate::dataset::DemoDataset;
use crate::types::{
    CandidateSummary, MissionPlayback, MissionSpec, MissionSummary, PlanningFrame, RuntimeOverview,
    RuntimeSnapshot, Vec2,
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

type TrainBackend = Autodiff<NdArray>;
type InferBackend = NdArray;

const TOP_CANDIDATES: usize = 7;
const GOAL_SUCCESS_THRESHOLD: f32 = 0.1;

#[derive(Clone, Debug)]
struct BuildConfig {
    dataset_seed: u64,
    dataset_episodes: usize,
    episode_length: usize,
    world_phase1_epochs: usize,
    world_phase2_epochs: usize,
    policy_epochs: usize,
    batch_size: usize,
    recommended_candidates: usize,
    recommended_opt_steps: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            dataset_seed: 42,
            dataset_episodes: 72,
            episode_length: 14,
            world_phase1_epochs: 12,
            world_phase2_epochs: 8,
            policy_epochs: 16,
            batch_size: 24,
            recommended_candidates: 18,
            recommended_opt_steps: 2,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ArenaReward;

impl<B: Backend> RewardFunction<B> for ArenaReward {
    fn compute_reward(&self, predicted_states: &Tensor<B, 3>) -> gpc_core::Result<Tensor<B, 1>> {
        let [batch_size, horizon, state_dim] = predicted_states.dims();
        let device = predicted_states.device();
        let data = tensor_to_vec(predicted_states.clone())?;
        let mut rewards = Vec::with_capacity(batch_size);

        for batch in 0..batch_size {
            let offset = batch * horizon * state_dim;
            let trajectory = &data[offset..offset + horizon * state_dim];

            let first = &trajectory[0..state_dim];
            let last = &trajectory[(horizon - 1) * state_dim..horizon * state_dim];

            let mut min_clearance = f32::INFINITY;

            for step in 0..horizon {
                let slice = &trajectory[step * state_dim..(step + 1) * state_dim];
                min_clearance = min_clearance.min(min_clearance_from_slice(slice));
            }

            let progress = goal_distance_from_slice(first) - goal_distance_from_slice(last);
            let final_distance = goal_distance_from_slice(last);

            let collision_penalty = if min_clearance < 0.0 {
                1.2 + min_clearance.abs() * 8.0
            } else if min_clearance < 0.08 {
                (0.08 - min_clearance) * 2.0
            } else {
                0.0
            };

            let reward = progress * 1.8 - final_distance * 3.3 - collision_penalty * 4.4
                + min_clearance.min(0.24) * 0.45;
            rewards.push(reward);
        }

        Ok(Tensor::<B, 1>::from_floats(rewards.as_slice(), &device))
    }
}

#[derive(Clone, Debug)]
struct Engine {
    policy: DiffusionPolicy<InferBackend>,
    world_model: gpc_world::StateWorldModel<InferBackend>,
    missions: Vec<MissionSpec>,
    overview: RuntimeOverview,
    opt_steps: usize,
}

impl Engine {
    fn bootstrap(config: BuildConfig) -> gpc_core::Result<Self> {
        let started = Instant::now();
        let train_device = <TrainBackend as Backend>::Device::default();
        let dataset = build_training_dataset(
            config.dataset_seed,
            config.dataset_episodes,
            config.episode_length,
        );

        let (world_model, world_loss_curve) =
            train_world_model(&dataset.dataset, &train_device, &config)?;
        let (policy, policy_loss_curve) = train_policy(&dataset.dataset, &train_device, &config)?;

        let overview = RuntimeOverview {
            dataset_episodes: dataset.episodes,
            dataset_transitions: dataset.transitions,
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            obs_horizon: OBS_HORIZON,
            pred_horizon: PRED_HORIZON,
            bootstrap_ms: started.elapsed().as_millis(),
            world_loss_curve,
            policy_loss_curve,
            recommended_candidates: config.recommended_candidates,
            recommended_opt_steps: config.recommended_opt_steps,
        };

        Ok(Self {
            policy,
            world_model,
            missions: preset_missions(),
            overview,
            opt_steps: config.recommended_opt_steps,
        })
    }

    fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            overview: self.overview.clone(),
            missions: self.missions.clone(),
        }
    }

    fn simulate_mission(
        &self,
        mission_id: &str,
        mode: &str,
        num_candidates: usize,
    ) -> gpc_core::Result<MissionPlayback> {
        let mission = self
            .missions
            .iter()
            .find(|mission| mission.id == mission_id)
            .cloned()
            .ok_or_else(|| {
                gpc_core::GpcError::Evaluation(format!("unknown mission: {mission_id}"))
            })?;
        let device = <InferBackend as Backend>::Device::default();
        let reward = ArenaReward;

        let mut current = mission_state(&mission);
        let mut obs_history = vec![state_to_vec(&current); OBS_HORIZON];
        let mut executed_path = vec![forward_kinematics(current.theta1, current.theta2).effector];
        let mut frames = Vec::with_capacity(mission.max_steps);
        let mut min_clearance = f32::INFINITY;
        let mut world_error_sum = 0.0_f32;

        for step in 0..mission.max_steps {
            let obs_tensor = tensor_from_history(&obs_history, &device);
            let current_tensor = tensor_from_state(&current, &device);

            let policy_actions = self.policy.sample(&obs_tensor, &device)?;
            let policy_rollout = self.world_model.rollout(&current_tensor, &policy_actions)?;
            let policy_path = trajectory_path(&policy_rollout)?;

            let candidate_actions = self.policy.sample_k(&obs_tensor, num_candidates, &device)?;
            let repeated_state =
                gpc_core::tensor_utils::repeat_batch_2d(&current_tensor, num_candidates);
            let candidate_rollouts = self
                .world_model
                .rollout(&repeated_state, &candidate_actions)?;
            let reward_tensor = reward.compute_reward(&candidate_rollouts)?;

            let candidate_summaries =
                summarize_candidates(&candidate_rollouts, &reward_tensor, TOP_CANDIDATES)?;
            let rank_actions =
                select_rank_actions(&candidate_actions, &reward_tensor, num_candidates)?;
            let ranked_rollout = self.world_model.rollout(&current_tensor, &rank_actions)?;
            let ranked_path = trajectory_path(&ranked_rollout)?;

            let optimized_actions = optimize_actions(
                &self.world_model,
                &reward,
                &policy_actions,
                &current_tensor,
                self.opt_steps,
            )?;
            let optimized_rollout = self
                .world_model
                .rollout(&current_tensor, &optimized_actions)?;
            let optimized_path = trajectory_path(&optimized_rollout)?;

            let selected_actions = if mode == "opt" {
                optimized_actions
            } else {
                rank_actions
            };

            let selected_first_action = first_action(&selected_actions)?;
            let selected_rollout = if mode == "opt" {
                optimized_rollout
            } else {
                ranked_rollout
            };
            let model_next_state = first_predicted_state(&selected_rollout)?;

            for _ in 0..EXECUTION_STRIDE {
                current = apply_action(&current, selected_first_action);
            }

            let actual_state_vec = state_to_vec(&current);
            let actual_effector = effector_from_slice(&actual_state_vec);
            executed_path.push(actual_effector);

            let world_model_error =
                actual_effector.distance(effector_from_slice(&model_next_state));
            world_error_sum += world_model_error;

            let goal_distance = goal_distance_from_slice(&actual_state_vec);
            min_clearance = min_clearance.min(min_clearance_from_slice(&actual_state_vec));

            let rewards = tensor_to_vec(reward_tensor)?;
            let reward_best = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let reward_mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
            let reward_low = rewards.iter().copied().fold(f32::INFINITY, f32::min);
            let reward_spread = reward_best - reward_low;

            frames.push(PlanningFrame {
                step,
                pose: forward_kinematics(current.theta1, current.theta2),
                executed_path: executed_path.clone(),
                policy_path,
                ranked_path,
                optimized_path,
                candidates: candidate_summaries,
                selected_action: selected_first_action,
                goal_distance,
                min_clearance,
                world_model_error,
                reward_mean,
                reward_best,
                reward_spread,
            });

            obs_history.remove(0);
            obs_history.push(actual_state_vec);

            if goal_distance <= GOAL_SUCCESS_THRESHOLD {
                break;
            }
        }

        let final_goal_distance = obs_history
            .last()
            .map(|values| goal_distance_from_slice(values))
            .unwrap_or_default();
        let average_world_error = if frames.is_empty() {
            0.0
        } else {
            world_error_sum / frames.len() as f32
        };

        Ok(MissionPlayback {
            mission,
            summary: MissionSummary {
                success: final_goal_distance <= GOAL_SUCCESS_THRESHOLD,
                final_goal_distance,
                min_clearance,
                average_world_error,
                executed_steps: frames.len(),
                mode: mode.to_string(),
            },
            frames,
        })
    }
}

#[wasm_bindgen]
pub struct DemoRuntime {
    engine: Engine,
}

#[wasm_bindgen]
impl DemoRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<DemoRuntime, JsValue> {
        let engine = Engine::bootstrap(BuildConfig::default())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self { engine })
    }

    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        to_value(&self.engine.snapshot()).map_err(into_js_error)
    }

    pub fn simulate_mission(
        &self,
        mission_id: &str,
        mode: &str,
        num_candidates: usize,
    ) -> Result<JsValue, JsValue> {
        let playback = self
            .engine
            .simulate_mission(mission_id, mode, num_candidates)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        to_value(&playback).map_err(into_js_error)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn init() {
    console_error_panic_hook::set_once();
}

fn train_world_model(
    dataset: &DemoDataset,
    device: &<TrainBackend as Backend>::Device,
    config: &BuildConfig,
) -> gpc_core::Result<(gpc_world::StateWorldModel<InferBackend>, Vec<f32>)> {
    let model_config = StateWorldModelConfig {
        state_dim: STATE_DIM,
        action_dim: ACTION_DIM,
        hidden_dim: 64,
        num_layers: 3,
    };
    let mut model = model_config.init::<TrainBackend>(device);
    let optimizer_config = AdamWConfig::new().with_weight_decay(1e-6);
    let mut optimizer = optimizer_config.init();
    let mut losses = Vec::with_capacity(config.world_phase1_epochs + config.world_phase2_epochs);

    let samples = dataset.world_model_samples();
    let batch_size = config.batch_size.min(samples.len());

    for _epoch in 0..config.world_phase1_epochs {
        let mut epoch_loss = 0.0_f32;
        let mut num_batches = 0;

        for chunk in samples.chunks(batch_size) {
            let batch_size = chunk.len();
            let states_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(state, _, _)| state.iter().copied())
                .collect();
            let actions_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(_, action, _)| action.iter().copied())
                .collect();
            let next_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(_, _, next_state)| next_state.iter().copied())
                .collect();

            let states = Tensor::<TrainBackend, 1>::from_floats(states_flat.as_slice(), device)
                .reshape([batch_size, STATE_DIM]);
            let actions = Tensor::<TrainBackend, 1>::from_floats(actions_flat.as_slice(), device)
                .reshape([batch_size, ACTION_DIM]);
            let next_states = Tensor::<TrainBackend, 1>::from_floats(next_flat.as_slice(), device)
                .reshape([batch_size, STATE_DIM]);

            let predicted_delta = model.predict_delta(&states, &actions);
            let target_delta = next_states.clone() - states.clone();
            let loss = tensor_utils::mse_loss(&predicted_delta, &target_delta);
            let loss_val: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(3e-3, model, grads);

            epoch_loss += loss_val;
            num_batches += 1;
        }

        losses.push(epoch_loss / num_batches as f32);
    }

    let sequences = dataset.world_model_sequences(PRED_HORIZON);
    let state_dim = STATE_DIM;
    let action_dim = ACTION_DIM;
    let batch_size = config.batch_size.min(sequences.len());

    for _epoch in 0..config.world_phase2_epochs {
        let mut epoch_loss = 0.0_f32;
        let mut num_batches = 0;

        for chunk in sequences.chunks(batch_size) {
            let batch_size = chunk.len();
            let initial_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(initial_state, _, _)| initial_state.iter().copied())
                .collect();
            let actions_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(_, actions, _)| {
                    actions.iter().flat_map(|action| action.iter().copied())
                })
                .collect();
            let target_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(_, _, target_states)| {
                    target_states.iter().flat_map(|state| state.iter().copied())
                })
                .collect();

            let initial_states =
                Tensor::<TrainBackend, 1>::from_floats(initial_flat.as_slice(), device)
                    .reshape([batch_size, state_dim]);
            let actions = Tensor::<TrainBackend, 1>::from_floats(actions_flat.as_slice(), device)
                .reshape([batch_size, PRED_HORIZON, action_dim]);
            let targets = Tensor::<TrainBackend, 1>::from_floats(target_flat.as_slice(), device)
                .reshape([batch_size, PRED_HORIZON, state_dim]);

            let predictions = model.rollout(&initial_states, &actions)?;
            let loss = tensor_utils::mse_loss(&predictions, &targets);
            let loss_val: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(1.6e-3, model, grads);

            epoch_loss += loss_val;
            num_batches += 1;
        }

        losses.push(epoch_loss / num_batches as f32);
    }

    Ok((model.valid(), losses))
}

fn train_policy(
    dataset: &DemoDataset,
    device: &<TrainBackend as Backend>::Device,
    config: &BuildConfig,
) -> gpc_core::Result<(DiffusionPolicy<InferBackend>, Vec<f32>)> {
    let schedule_config = NoiseScheduleConfig {
        num_timesteps: 24,
        beta_start: 1e-4,
        beta_end: 0.02,
    };
    let policy_config = PolicyConfig {
        obs_dim: STATE_DIM,
        action_dim: ACTION_DIM,
        obs_horizon: OBS_HORIZON,
        pred_horizon: PRED_HORIZON,
        action_horizon: EXECUTION_STRIDE,
        hidden_dim: 64,
        num_res_blocks: 1,
        noise_schedule: schedule_config.clone(),
    };
    let model_config = DiffusionPolicyConfig::from_policy_config(&policy_config);
    let mut model = model_config.init::<TrainBackend>(device);
    let schedule = DdpmSchedule::new(&schedule_config);

    let optimizer_config = AdamWConfig::new().with_weight_decay(1e-6);
    let mut optimizer = optimizer_config.init();

    let samples = dataset.policy_samples();
    let batch_size = config.batch_size.min(samples.len());
    let mut losses = Vec::with_capacity(config.policy_epochs);

    for _epoch in 0..config.policy_epochs {
        let mut epoch_loss = 0.0_f32;
        let mut num_batches = 0;

        for chunk in samples.chunks(batch_size) {
            let batch_size = chunk.len();
            let obs_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(observations, _)| {
                    observations
                        .iter()
                        .flat_map(|observation| observation.iter().copied())
                })
                .collect();
            let actions_flat: Vec<f32> = chunk
                .iter()
                .flat_map(|(_, actions)| actions.iter().flat_map(|action| action.iter().copied()))
                .collect();

            let observations = Tensor::<TrainBackend, 1>::from_floats(obs_flat.as_slice(), device)
                .reshape([batch_size, OBS_HORIZON, STATE_DIM]);
            let actions = Tensor::<TrainBackend, 1>::from_floats(actions_flat.as_slice(), device)
                .reshape([batch_size, PRED_HORIZON, ACTION_DIM]);
            let actions_flat = tensor_utils::flatten_last_two(actions);
            let obs_flat = tensor_utils::flatten_last_two(observations);

            let timesteps: Vec<f32> = (0..batch_size)
                .map(|_| rand::random::<f32>() * (schedule.num_timesteps - 1) as f32)
                .collect();
            let timestep_tensor =
                Tensor::<TrainBackend, 1>::from_floats(timesteps.as_slice(), device);

            let noise = Tensor::<TrainBackend, 2>::random(
                [batch_size, PRED_HORIZON * ACTION_DIM],
                Distribution::Normal(0.0, 1.0),
                device,
            );
            let t_index = timesteps[0].floor() as usize;
            let noisy_actions = schedule.add_noise(&actions_flat, &noise, t_index);
            let prediction = model.predict_noise(noisy_actions, obs_flat, timestep_tensor);
            let loss = tensor_utils::mse_loss(&prediction, &noise);
            let loss_val: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(1.8e-3, model, grads);

            epoch_loss += loss_val;
            num_batches += 1;
        }

        losses.push(epoch_loss / num_batches as f32);
    }

    Ok((model.valid(), losses))
}

fn summarize_candidates(
    rollouts: &Tensor<InferBackend, 3>,
    rewards: &Tensor<InferBackend, 1>,
    limit: usize,
) -> gpc_core::Result<Vec<CandidateSummary>> {
    let [batch_size, horizon, state_dim] = rollouts.dims();
    let rollout_data = tensor_to_vec(rollouts.clone())?;
    let reward_data = tensor_to_vec(rewards.clone())?;

    let mut candidates = Vec::with_capacity(batch_size);

    for (index, reward) in reward_data.iter().copied().enumerate().take(batch_size) {
        let start = index * horizon * state_dim;
        let trajectory = &rollout_data[start..start + horizon * state_dim];
        let path = (0..horizon)
            .map(|step| {
                let slice = &trajectory[step * state_dim..(step + 1) * state_dim];
                effector_from_slice(slice)
            })
            .collect::<Vec<_>>();

        let min_clearance = (0..horizon)
            .map(|step| {
                let slice = &trajectory[step * state_dim..(step + 1) * state_dim];
                min_clearance_from_slice(slice)
            })
            .fold(f32::INFINITY, f32::min);

        let terminal_distance = trajectory
            .chunks_exact(state_dim)
            .last()
            .map(goal_distance_from_slice)
            .unwrap_or_default();

        candidates.push(CandidateSummary {
            rank: index + 1,
            reward,
            clearance: min_clearance,
            terminal_distance,
            effector_path: path,
        });
    }

    candidates.sort_by(|left, right| right.reward.total_cmp(&left.reward));
    for (rank, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = rank + 1;
    }
    candidates.truncate(limit);

    Ok(candidates)
}

fn select_rank_actions(
    candidate_actions: &Tensor<InferBackend, 3>,
    rewards: &Tensor<InferBackend, 1>,
    num_candidates: usize,
) -> gpc_core::Result<Tensor<InferBackend, 3>> {
    let best_index: i64 = rewards.clone().argmax(0).into_scalar().elem();
    let [_, horizon, action_dim] = candidate_actions.dims();
    let best_index = (best_index as usize).min(num_candidates.saturating_sub(1));

    Ok(candidate_actions
        .clone()
        .slice([best_index..best_index + 1, 0..horizon, 0..action_dim]))
}

fn optimize_actions(
    world_model: &gpc_world::StateWorldModel<InferBackend>,
    reward_fn: &ArenaReward,
    initial_actions: &Tensor<InferBackend, 3>,
    current_state: &Tensor<InferBackend, 2>,
    num_steps: usize,
) -> gpc_core::Result<Tensor<InferBackend, 3>> {
    let [batch_size, horizon, action_dim] = initial_actions.dims();
    let mut actions = initial_actions.clone();
    let epsilon = 1e-3_f32;

    for _ in 0..num_steps {
        let predicted_states = world_model.rollout(current_state, &actions)?;
        let reward = reward_fn.compute_reward(&predicted_states)?;
        let base_reward: f32 = reward.into_scalar().elem();
        let flat_actions =
            tensor_to_vec(actions.clone().reshape([batch_size * horizon * action_dim]))?;
        let mut gradient = vec![0.0_f32; flat_actions.len()];

        for index in 0..flat_actions.len() {
            let mut perturbed = flat_actions.clone();
            perturbed[index] += epsilon;
            let device = actions.device();
            let perturbed_actions =
                Tensor::<InferBackend, 1>::from_floats(perturbed.as_slice(), &device)
                    .reshape([batch_size, horizon, action_dim]);
            let perturbed_states = world_model.rollout(current_state, &perturbed_actions)?;
            let perturbed_reward = reward_fn.compute_reward(&perturbed_states)?;
            let reward_value: f32 = perturbed_reward.into_scalar().elem();
            gradient[index] = (reward_value - base_reward) / epsilon;
        }

        let device = actions.device();
        let gradient_tensor = Tensor::<InferBackend, 1>::from_floats(gradient.as_slice(), &device)
            .reshape([batch_size, horizon, action_dim]);
        actions = actions + gradient_tensor * 0.015;
    }

    Ok(actions)
}

fn tensor_from_history(
    history: &[Vec<f32>],
    device: &<InferBackend as Backend>::Device,
) -> Tensor<InferBackend, 3> {
    let flat: Vec<f32> = history
        .iter()
        .flat_map(|entry| entry.iter().copied())
        .collect();
    Tensor::<InferBackend, 1>::from_floats(flat.as_slice(), device).reshape([
        1,
        OBS_HORIZON,
        STATE_DIM,
    ])
}

fn tensor_from_state(
    state: &RobotState,
    device: &<InferBackend as Backend>::Device,
) -> Tensor<InferBackend, 2> {
    Tensor::<InferBackend, 1>::from_floats(state_to_vec(state).as_slice(), device)
        .reshape([1, STATE_DIM])
}

fn trajectory_path(trajectory: &Tensor<InferBackend, 3>) -> gpc_core::Result<Vec<Vec2>> {
    let [_, horizon, state_dim] = trajectory.dims();
    let data = tensor_to_vec(trajectory.clone())?;

    Ok((0..horizon)
        .map(|step| {
            let slice = &data[step * state_dim..(step + 1) * state_dim];
            effector_from_slice(slice)
        })
        .collect())
}

fn first_action(actions: &Tensor<InferBackend, 3>) -> gpc_core::Result<[f32; 2]> {
    let [_, _, action_dim] = actions.dims();
    let slice = actions
        .clone()
        .slice([0..1, 0..1, 0..action_dim])
        .reshape([action_dim]);
    let data = tensor_to_vec(slice)?;

    Ok([data[0], data[1]])
}

fn first_predicted_state(trajectory: &Tensor<InferBackend, 3>) -> gpc_core::Result<Vec<f32>> {
    let [_, _, state_dim] = trajectory.dims();
    let slice = trajectory
        .clone()
        .slice([0..1, 0..1, 0..state_dim])
        .reshape([state_dim]);
    tensor_to_vec(slice)
}

fn tensor_to_vec<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> gpc_core::Result<Vec<f32>> {
    tensor.into_data().to_vec().map_err(|error| {
        gpc_core::GpcError::Evaluation(format!("failed to extract tensor data: {error:?}"))
    })
}

fn into_js_error(error: serde_wasm_bindgen::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}
