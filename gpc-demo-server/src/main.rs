mod arena;
mod dataset;
mod types;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use serde::Deserialize;
use tower_http::cors::CorsLayer;

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
    CandidateSummary, MissionPlayback, MissionSpec, MissionSummary, PlanningFrame,
    RuntimeBuildConfig, RuntimeOverview, RuntimeSnapshot, Vec2,
};

type TrainBackend = Autodiff<NdArray>;
type InferBackend = NdArray;

const TOP_CANDIDATES: usize = 7;
const GOAL_SUCCESS_THRESHOLD: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlannerMode {
    Policy,
    Rank,
    Opt,
}

impl PlannerMode {
    fn parse(mode: &str) -> gpc_core::Result<Self> {
        match mode {
            "policy" => Ok(Self::Policy),
            "rank" => Ok(Self::Rank),
            "opt" => Ok(Self::Opt),
            other => Err(gpc_core::GpcError::Config(format!(
                "unsupported planner mode: {other} (expected policy, rank, or opt)"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Policy => "policy",
            Self::Rank => "rank",
            Self::Opt => "opt",
        }
    }
}

fn select_mode_outputs(
    mode: PlannerMode,
    policy_actions: Tensor<InferBackend, 3>,
    policy_rollout: Tensor<InferBackend, 3>,
    rank_actions: Tensor<InferBackend, 3>,
    ranked_rollout: Tensor<InferBackend, 3>,
    optimized_actions: Tensor<InferBackend, 3>,
    optimized_rollout: Tensor<InferBackend, 3>,
) -> (Tensor<InferBackend, 3>, Tensor<InferBackend, 3>) {
    match mode {
        PlannerMode::Policy => (policy_actions, policy_rollout),
        PlannerMode::Rank => (rank_actions, ranked_rollout),
        PlannerMode::Opt => (optimized_actions, optimized_rollout),
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
    fn bootstrap(config: RuntimeBuildConfig) -> gpc_core::Result<Self> {
        config.validate()?;
        let started = Instant::now();
        let train_device = <TrainBackend as Backend>::Device::default();
        let dataset = build_training_dataset(
            config.dataset_seed,
            config.dataset_episodes,
            config.episode_length,
        );

        tracing::info!(
            episodes = dataset.episodes,
            transitions = dataset.transitions,
            "dataset generated, training world model…"
        );

        let (world_model, world_loss_curve) =
            train_world_model(&dataset.dataset, &train_device, &config)?;

        tracing::info!("world model trained, training diffusion policy…");

        let (policy, policy_loss_curve) = train_policy(&dataset.dataset, &train_device, &config)?;

        let bootstrap_ms = started.elapsed().as_millis();
        tracing::info!(bootstrap_ms, "bootstrap complete");

        let overview = RuntimeOverview {
            dataset_episodes: dataset.episodes,
            dataset_transitions: dataset.transitions,
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            obs_horizon: OBS_HORIZON,
            pred_horizon: PRED_HORIZON,
            bootstrap_ms,
            world_loss_curve,
            policy_loss_curve,
            recommended_candidates: config.recommended_candidates,
            recommended_opt_steps: config.recommended_opt_steps,
            build_config: config.clone(),
        };
        let opt_steps = config.recommended_opt_steps;

        Ok(Self {
            policy,
            world_model,
            missions: preset_missions(),
            overview,
            opt_steps,
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
        let mode = PlannerMode::parse(mode)?;
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

            let (
                selected_actions,
                selected_rollout,
                ranked_path,
                optimized_path,
                candidates,
                reward_mean,
                reward_best,
                reward_spread,
            ) = if mode == PlannerMode::Policy {
                (
                    policy_actions,
                    policy_rollout,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    0.0,
                    0.0,
                    0.0,
                )
            } else {
                let candidate_actions =
                    self.policy.sample_k(&obs_tensor, num_candidates, &device)?;
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

                let rewards = tensor_to_vec(reward_tensor)?;
                let reward_best = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let reward_mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
                let reward_low = rewards.iter().copied().fold(f32::INFINITY, f32::min);
                let reward_spread = reward_best - reward_low;

                let (selected_actions, selected_rollout) = select_mode_outputs(
                    mode,
                    policy_actions,
                    policy_rollout,
                    rank_actions,
                    ranked_rollout,
                    optimized_actions,
                    optimized_rollout,
                );

                (
                    selected_actions,
                    selected_rollout,
                    ranked_path,
                    optimized_path,
                    candidate_summaries,
                    reward_mean,
                    reward_best,
                    reward_spread,
                )
            };

            let selected_first_action = first_action(&selected_actions)?;
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

            frames.push(PlanningFrame {
                step,
                pose: forward_kinematics(current.theta1, current.theta2),
                executed_path: executed_path.clone(),
                policy_path,
                ranked_path,
                optimized_path,
                candidates,
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

        let executed_steps = frames.len();
        Ok(MissionPlayback {
            mission,
            frames,
            summary: MissionSummary {
                success: final_goal_distance <= GOAL_SUCCESS_THRESHOLD,
                final_goal_distance,
                min_clearance,
                average_world_error,
                executed_steps,
                mode: mode.as_str().to_string(),
            },
        })
    }
}

// -- HTTP handlers ----------------------------------------------------------

type AppState = Arc<Mutex<Engine>>;

async fn handle_snapshot(State(engine): State<AppState>) -> Json<RuntimeSnapshot> {
    let engine = engine.lock().expect("engine lock poisoned");
    Json(engine.snapshot())
}

#[derive(Deserialize)]
struct SimulateRequest {
    mission_id: String,
    mode: String,
    num_candidates: usize,
}

#[derive(Deserialize)]
struct RebuildRequest {
    config: RuntimeBuildConfig,
}

async fn handle_simulate(
    State(engine): State<AppState>,
    Json(request): Json<SimulateRequest>,
) -> Result<Json<MissionPlayback>, (StatusCode, String)> {
    let engine = engine.lock().expect("engine lock poisoned");
    let result = engine
        .simulate_mission(&request.mission_id, &request.mode, request.num_candidates)
        .map_err(|error| {
            let status = match error {
                gpc_core::GpcError::Config(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, format!("simulation error: {error}"))
        })?;

    Ok(Json(result))
}

async fn handle_rebuild(
    State(engine): State<AppState>,
    Json(request): Json<RebuildRequest>,
) -> Result<Json<RuntimeSnapshot>, (StatusCode, String)> {
    let new_engine = Engine::bootstrap(request.config).map_err(|error| {
        let status = match error {
            gpc_core::GpcError::Config(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, format!("rebuild error: {error}"))
    })?;

    let snapshot = new_engine.snapshot();
    let mut engine = engine.lock().expect("engine lock poisoned");
    *engine = new_engine;

    Ok(Json(snapshot))
}

async fn handle_health() -> &'static str {
    "ok"
}

// -- main -------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gpc_demo_server=info".parse().unwrap()),
        )
        .init();

    tracing::info!("bootstrapping engine (training models)…");
    let engine = Engine::bootstrap(RuntimeBuildConfig::default()).expect("engine bootstrap failed");
    tracing::info!("engine ready, starting HTTP server");

    let state: AppState = Arc::new(Mutex::new(engine));

    let app = Router::new()
        .route("/api/health", get(handle_health))
        .route("/api/snapshot", get(handle_snapshot))
        .route("/api/rebuild", post(handle_rebuild))
        .route("/api/simulate", post(handle_simulate))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3100")
        .await
        .expect("failed to bind to port 3100");
    tracing::info!("listening on http://127.0.0.1:3100");
    axum::serve(listener, app).await.expect("server error");
}

// -- training ---------------------------------------------------------------

fn train_world_model(
    dataset: &DemoDataset,
    device: &<TrainBackend as Backend>::Device,
    config: &RuntimeBuildConfig,
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
    if samples.is_empty() {
        return Err(gpc_core::GpcError::Training(
            "world model dataset is empty".to_string(),
        ));
    }
    let batch_size = config.batch_size.min(samples.len());

    for epoch in 0..config.world_phase1_epochs {
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

        let avg = epoch_loss / num_batches as f32;
        losses.push(avg);
        tracing::debug!(epoch, phase = 1, loss = avg, "world model");
    }

    let sequences = dataset.world_model_sequences(PRED_HORIZON);
    let state_dim = STATE_DIM;
    let action_dim = ACTION_DIM;
    let batch_size = config.batch_size.min(sequences.len());

    for epoch in 0..config.world_phase2_epochs {
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

        let avg = epoch_loss / num_batches as f32;
        losses.push(avg);
        tracing::debug!(epoch, phase = 2, loss = avg, "world model");
    }

    Ok((model.valid(), losses))
}

fn train_policy(
    dataset: &DemoDataset,
    device: &<TrainBackend as Backend>::Device,
    config: &RuntimeBuildConfig,
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
    if samples.is_empty() {
        return Err(gpc_core::GpcError::Training(
            "policy dataset is empty".to_string(),
        ));
    }
    let batch_size = config.batch_size.min(samples.len());
    let mut losses = Vec::with_capacity(config.policy_epochs);

    for epoch in 0..config.policy_epochs {
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

            let timestep_indices: Vec<usize> = (0..batch_size)
                .map(|_| {
                    (rand::random::<f32>() * schedule.num_timesteps as f32)
                        .floor()
                        .min((schedule.num_timesteps - 1) as f32) as usize
                })
                .collect();
            let timestep_tensor = Tensor::<TrainBackend, 1>::from_floats(
                timestep_indices
                    .iter()
                    .map(|&t| t as f32)
                    .collect::<Vec<_>>()
                    .as_slice(),
                device,
            );

            let noise = Tensor::<TrainBackend, 2>::random(
                [batch_size, PRED_HORIZON * ACTION_DIM],
                Distribution::Normal(0.0, 1.0),
                device,
            );
            let noisy_actions =
                schedule.add_noise_batch(&actions_flat, &noise, timestep_indices.as_slice());
            let prediction = model.predict_noise(noisy_actions, obs_flat, timestep_tensor);
            let loss = tensor_utils::mse_loss(&prediction, &noise);
            let loss_val: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(1.8e-3, model, grads);

            epoch_loss += loss_val;
            num_batches += 1;
        }

        let avg = epoch_loss / num_batches as f32;
        losses.push(avg);
        tracing::debug!(epoch, loss = avg, "policy");
    }

    Ok((model.valid(), losses))
}

impl RuntimeBuildConfig {
    fn validate(&self) -> gpc_core::Result<()> {
        if self.dataset_episodes == 0 {
            return Err(gpc_core::GpcError::Config(
                "dataset_episodes must be greater than zero".to_string(),
            ));
        }
        if self.episode_length == 0 {
            return Err(gpc_core::GpcError::Config(
                "episode_length must be greater than zero".to_string(),
            ));
        }
        if self.episode_length <= PRED_HORIZON {
            return Err(gpc_core::GpcError::Config(format!(
                "episode_length must be greater than PRED_HORIZON ({PRED_HORIZON})"
            )));
        }
        if self.world_phase1_epochs == 0 {
            return Err(gpc_core::GpcError::Config(
                "world_phase1_epochs must be greater than zero".to_string(),
            ));
        }
        if self.world_phase2_epochs == 0 {
            return Err(gpc_core::GpcError::Config(
                "world_phase2_epochs must be greater than zero".to_string(),
            ));
        }
        if self.policy_epochs == 0 {
            return Err(gpc_core::GpcError::Config(
                "policy_epochs must be greater than zero".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(gpc_core::GpcError::Config(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        if self.recommended_candidates == 0 {
            return Err(gpc_core::GpcError::Config(
                "recommended_candidates must be greater than zero".to_string(),
            ));
        }
        if self.recommended_opt_steps == 0 {
            return Err(gpc_core::GpcError::Config(
                "recommended_opt_steps must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

// -- tensor helpers ---------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> RuntimeBuildConfig {
        RuntimeBuildConfig {
            dataset_seed: 42,
            dataset_episodes: 4,
            episode_length: 8,
            world_phase1_epochs: 1,
            world_phase2_epochs: 1,
            policy_epochs: 1,
            batch_size: 2,
            recommended_candidates: 3,
            recommended_opt_steps: 2,
        }
    }

    #[test]
    fn config_validation_rejects_zero_values() {
        let mut config = sample_config();
        config.batch_size = 0;

        let error = config.validate().expect_err("expected validation failure");
        assert!(matches!(error, gpc_core::GpcError::Config(_)));
    }

    #[test]
    fn config_validation_rejects_episode_length_at_or_below_horizon() {
        let mut config = sample_config();
        config.episode_length = PRED_HORIZON;

        let error = config.validate().expect_err("expected validation failure");
        assert!(
            matches!(error, gpc_core::GpcError::Config(message) if message.contains("PRED_HORIZON"))
        );
    }

    #[test]
    fn bootstrap_snapshot_reflects_config() {
        let config = sample_config();
        let engine = Engine::bootstrap(config.clone()).expect("bootstrap should succeed");

        assert_eq!(engine.snapshot().overview.build_config, config);
    }

    #[test]
    fn select_mode_outputs_prefers_the_expected_branch() {
        let device = <InferBackend as Backend>::Device::default();
        let policy_actions = Tensor::<InferBackend, 3>::from_floats([[[1.0, 2.0]]], &device);
        let policy_rollout = Tensor::<InferBackend, 3>::from_floats([[[3.0, 4.0]]], &device);
        let rank_actions = Tensor::<InferBackend, 3>::from_floats([[[5.0, 6.0]]], &device);
        let ranked_rollout = Tensor::<InferBackend, 3>::from_floats([[[7.0, 8.0]]], &device);
        let optimized_actions = Tensor::<InferBackend, 3>::from_floats([[[9.0, 10.0]]], &device);
        let optimized_rollout = Tensor::<InferBackend, 3>::from_floats([[[11.0, 12.0]]], &device);

        let (selected_actions, selected_rollout) = select_mode_outputs(
            PlannerMode::Policy,
            policy_actions.clone(),
            policy_rollout.clone(),
            rank_actions.clone(),
            ranked_rollout.clone(),
            optimized_actions.clone(),
            optimized_rollout.clone(),
        );
        assert_eq!(tensor_to_vec(selected_actions).unwrap(), vec![1.0, 2.0]);
        assert_eq!(tensor_to_vec(selected_rollout).unwrap(), vec![3.0, 4.0]);

        let (selected_actions, selected_rollout) = select_mode_outputs(
            PlannerMode::Rank,
            policy_actions.clone(),
            policy_rollout.clone(),
            rank_actions.clone(),
            ranked_rollout.clone(),
            optimized_actions.clone(),
            optimized_rollout.clone(),
        );
        assert_eq!(tensor_to_vec(selected_actions).unwrap(), vec![5.0, 6.0]);
        assert_eq!(tensor_to_vec(selected_rollout).unwrap(), vec![7.0, 8.0]);

        let (selected_actions, selected_rollout) = select_mode_outputs(
            PlannerMode::Opt,
            policy_actions,
            policy_rollout,
            rank_actions,
            ranked_rollout,
            optimized_actions,
            optimized_rollout,
        );
        assert_eq!(tensor_to_vec(selected_actions).unwrap(), vec![9.0, 10.0]);
        assert_eq!(tensor_to_vec(selected_rollout).unwrap(), vec![11.0, 12.0]);
    }

    #[test]
    fn simulate_mission_supports_policy_mode_and_rejects_invalid_modes() {
        let engine = Engine::bootstrap(sample_config()).expect("bootstrap should succeed");
        let mission = engine.missions[0].clone();
        let policy_playback = engine
            .simulate_mission(&mission.id, "policy", 4)
            .expect("policy playback should succeed");
        assert_eq!(policy_playback.summary.mode, "policy");
        assert!(!policy_playback.frames.is_empty());
        let frame = &policy_playback.frames[0];
        assert!(frame.candidates.is_empty());
        assert!(frame.ranked_path.is_empty());
        assert!(frame.optimized_path.is_empty());
        assert_eq!(frame.reward_mean, 0.0);
        assert_eq!(frame.reward_best, 0.0);
        assert_eq!(frame.reward_spread, 0.0);

        let error = engine
            .simulate_mission(&mission.id, "invalid", 4)
            .expect_err("invalid mode should fail");
        assert!(
            matches!(error, gpc_core::GpcError::Config(message) if message.contains("unsupported planner mode"))
        );
    }
}
