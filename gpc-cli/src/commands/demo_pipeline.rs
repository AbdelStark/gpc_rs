//! Shared end-to-end demo pipeline for the CLI demo command.

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use gpc_core::config::{PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_core::traits::Evaluator;
use gpc_eval::GpcRankBuilder;
use gpc_train::data::{GpcDataset, GpcDatasetConfig};
use gpc_train::{PolicyTrainer, WorldModelTrainer};
use gpc_world::reward::L2RewardFunctionConfig;

use super::DemoArgs;

type DemoBackend = Autodiff<NdArray>;

const OBS_HORIZON: usize = 2;
const PRED_HORIZON: usize = 4;

/// Summary of a completed training stage.
#[derive(Debug, Clone)]
pub(crate) struct StageSummary {
    pub(crate) final_epoch: Option<usize>,
    pub(crate) final_loss: Option<f32>,
}

/// Summary of the evaluation step.
#[derive(Debug, Clone)]
pub(crate) struct EvaluationSummary {
    pub(crate) selected_action_shape: [usize; 3],
    pub(crate) selected_action_values: Vec<f32>,
}

/// Summary of the full demo pipeline.
#[derive(Debug, Clone)]
pub(crate) struct DemoRunSummary {
    pub(crate) dataset_episodes: usize,
    pub(crate) dataset_transitions: usize,
    pub(crate) world_model_phase1: StageSummary,
    pub(crate) world_model_phase2: StageSummary,
    pub(crate) policy: StageSummary,
    pub(crate) evaluation: EvaluationSummary,
}

/// Execute the full demo pipeline with synthetic data and trained models.
pub(crate) fn run_demo_pipeline(args: &DemoArgs) -> Result<DemoRunSummary> {
    tracing::info!("Step 1: Generating synthetic dataset...");

    let device = <DemoBackend as Backend>::Device::default();
    let dataset_config = GpcDatasetConfig {
        data_dir: "demo".to_string(),
        state_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_dim: args.state_dim,
        obs_horizon: OBS_HORIZON,
        pred_horizon: PRED_HORIZON,
    };
    let dataset = GpcDataset::generate_synthetic(
        dataset_config,
        args.episodes,
        args.episode_length,
        args.seed,
    );

    tracing::info!(
        "  Generated {} episodes, {} transitions",
        dataset.num_episodes(),
        dataset.num_transitions()
    );

    let training_config = TrainingConfig {
        num_epochs: args.epochs,
        batch_size: 16,
        learning_rate: 1e-3,
        log_every: 1,
        ..Default::default()
    };

    let world_model_config = WorldModelConfig {
        state_dim: args.state_dim,
        action_dim: args.action_dim,
        hidden_dim: 32,
        num_layers: 2,
        ..Default::default()
    };

    tracing::info!("Step 2: Training world model (Phase 1)...");
    let world_trainer = WorldModelTrainer::new(training_config.clone(), world_model_config);
    let phase1 = world_trainer.train_phase1_with_summary::<DemoBackend>(&dataset, &device);

    tracing::info!("Step 3: Refining world model (Phase 2)...");
    let phase2 = world_trainer.train_phase2_with_summary::<DemoBackend>(
        &dataset,
        phase1.model,
        PRED_HORIZON,
        &device,
    );

    let policy_config = PolicyConfig {
        obs_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_horizon: OBS_HORIZON,
        pred_horizon: PRED_HORIZON,
        hidden_dim: 32,
        num_res_blocks: 1,
        ..Default::default()
    };

    tracing::info!("Step 4: Training diffusion policy...");
    let policy_trainer = PolicyTrainer::new(training_config, policy_config);
    let policy = policy_trainer.train_with_summary::<DemoBackend>(&dataset, &device);

    tracing::info!("Step 5: Running GPC-RANK evaluation on trained models...");
    let evaluation = run_evaluation(
        &dataset,
        policy.model,
        phase2.model,
        args.num_candidates,
        &device,
    )?;

    Ok(DemoRunSummary {
        dataset_episodes: dataset.num_episodes(),
        dataset_transitions: dataset.num_transitions(),
        world_model_phase1: StageSummary {
            final_epoch: phase1.final_epoch,
            final_loss: phase1.final_loss,
        },
        world_model_phase2: StageSummary {
            final_epoch: phase2.final_epoch,
            final_loss: phase2.final_loss,
        },
        policy: StageSummary {
            final_epoch: policy.final_epoch,
            final_loss: policy.final_loss,
        },
        evaluation,
    })
}

fn run_evaluation(
    dataset: &GpcDataset,
    policy: gpc_policy::DiffusionPolicy<DemoBackend>,
    world_model: gpc_world::StateWorldModel<DemoBackend>,
    num_candidates: usize,
    device: &<DemoBackend as Backend>::Device,
) -> Result<EvaluationSummary> {
    let policy_samples = dataset.policy_samples();
    let world_samples = dataset.world_model_samples();

    let (obs_history, _) = policy_samples
        .first()
        .context("missing policy sample for evaluation")?;
    let (current_state, _action, goal_state) = world_samples
        .first()
        .context("missing world model sample for evaluation")?;

    let obs_tensor = observation_history_tensor(obs_history, device)?;
    let current_state_tensor =
        vector_tensor(current_state, device).reshape([1, current_state.len()]);
    let goal_tensor = vector_tensor(goal_state, device);

    let reward_fn = L2RewardFunctionConfig {
        state_dim: goal_state.len(),
    }
    .init::<DemoBackend>(device)
    .with_goal(goal_tensor);

    let evaluator = GpcRankBuilder::new(policy, world_model, reward_fn)
        .num_candidates(num_candidates)
        .build();

    let selected_action = evaluator.select_action(&obs_tensor, &current_state_tensor, device)?;
    let selected_action_shape = selected_action.dims();
    let selected_action_flat_len =
        selected_action_shape[0] * selected_action_shape[1] * selected_action_shape[2];
    let selected_action_values =
        tensor_to_vec(selected_action.reshape([selected_action_flat_len]))?;

    Ok(EvaluationSummary {
        selected_action_shape,
        selected_action_values,
    })
}

fn observation_history_tensor(
    history: &[Vec<f32>],
    device: &<DemoBackend as Backend>::Device,
) -> Result<Tensor<DemoBackend, 3>> {
    let obs_horizon = history.len();
    let obs_dim = history
        .first()
        .map(|row| row.len())
        .context("observation history cannot be empty")?;
    let flat = history
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();

    Ok(
        Tensor::<DemoBackend, 1>::from_floats(flat.as_slice(), device).reshape([
            1,
            obs_horizon,
            obs_dim,
        ]),
    )
}

fn vector_tensor(
    values: &[f32],
    device: &<DemoBackend as Backend>::Device,
) -> Tensor<DemoBackend, 1> {
    Tensor::<DemoBackend, 1>::from_floats(values, device)
}

fn tensor_to_vec(tensor: Tensor<DemoBackend, 1>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .to_vec()
        .map_err(|error| anyhow::anyhow!("failed to extract tensor data: {error:?}"))
}
