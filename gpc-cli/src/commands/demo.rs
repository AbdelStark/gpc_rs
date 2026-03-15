//! Demo command: end-to-end pipeline with synthetic data.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use clap::Args;

use gpc_core::config::{PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_core::traits::Evaluator;
use gpc_eval::GpcRankBuilder;
use gpc_policy::DiffusionPolicyConfig;
use gpc_train::data::{GpcDataset, GpcDatasetConfig};
use gpc_train::{PolicyTrainer, WorldModelTrainer};
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

type TrainBackend = Autodiff<NdArray>;
type InferBackend = NdArray;

/// Arguments for the demo command.
#[derive(Args, Debug)]
pub struct DemoArgs {
    /// Number of training epochs (keep small for demo).
    #[arg(long, default_value = "5")]
    epochs: usize,

    /// Number of GPC-RANK candidates.
    #[arg(long, default_value = "10")]
    num_candidates: usize,

    /// State/observation dimensionality.
    #[arg(long, default_value = "4")]
    state_dim: usize,

    /// Action dimensionality.
    #[arg(long, default_value = "2")]
    action_dim: usize,
}

/// Run the end-to-end demo.
pub fn run_demo(args: DemoArgs) -> Result<()> {
    tracing::info!("=== GPC End-to-End Demo ===");
    tracing::info!(
        "Config: state_dim={}, action_dim={}, epochs={}, candidates={}",
        args.state_dim,
        args.action_dim,
        args.epochs,
        args.num_candidates
    );

    let device = <TrainBackend as Backend>::Device::default();

    // 1. Generate synthetic dataset
    tracing::info!("Step 1: Generating synthetic dataset...");
    let dataset_config = GpcDatasetConfig {
        data_dir: "demo".to_string(),
        state_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_dim: args.state_dim,
        obs_horizon: 2,
        pred_horizon: 4,
    };
    let dataset = GpcDataset::generate_synthetic(dataset_config, 10, 30, 42);
    tracing::info!(
        "  Generated {} episodes, {} transitions",
        dataset.num_episodes(),
        dataset.num_transitions()
    );

    // 2. Train world model (Phase 1 only for demo)
    tracing::info!("Step 2: Training world model (Phase 1)...");
    let training_config = TrainingConfig {
        num_epochs: args.epochs,
        batch_size: 32,
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
    let wm_trainer = WorldModelTrainer::new(training_config.clone(), world_model_config);
    let _world_model = wm_trainer.train_phase1::<TrainBackend>(&dataset, &device);

    // 3. Train diffusion policy
    tracing::info!("Step 3: Training diffusion policy...");
    let policy_config = PolicyConfig {
        obs_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_horizon: 2,
        pred_horizon: 4,
        hidden_dim: 32,
        num_res_blocks: 1,
        ..Default::default()
    };
    let policy_trainer = PolicyTrainer::new(training_config, policy_config);
    let _policy = policy_trainer.train::<TrainBackend>(&dataset, &device);

    // 4. Convert to inference backend
    tracing::info!("Step 4: Running GPC-RANK evaluation...");
    let infer_device = <InferBackend as Backend>::Device::default();

    // Create fresh inference models (in production, you'd load from checkpoint)
    let policy_config = DiffusionPolicyConfig {
        obs_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_horizon: 2,
        pred_horizon: 4,
        hidden_dim: 32,
        time_embed_dim: 16,
        num_res_blocks: 1,
    };
    let infer_policy = policy_config.init::<InferBackend>(&infer_device);

    let wm_config = StateWorldModelConfig {
        state_dim: args.state_dim,
        action_dim: args.action_dim,
        hidden_dim: 32,
        num_layers: 2,
    };
    let infer_wm = wm_config.init::<InferBackend>(&infer_device);

    let reward_config = L2RewardFunctionConfig {
        state_dim: args.state_dim,
    };
    let reward_fn = reward_config.init::<InferBackend>(&infer_device);

    // 5. GPC-RANK evaluation
    let evaluator = GpcRankBuilder::new(infer_policy, infer_wm, reward_fn)
        .num_candidates(args.num_candidates)
        .build();

    let obs = Tensor::<InferBackend, 3>::zeros([1, 2, args.state_dim], &infer_device);
    let state = Tensor::<InferBackend, 2>::zeros([1, args.state_dim], &infer_device);

    let best_action = evaluator.select_action(&obs, &state, &infer_device)?;
    let [_, horizon, action_dim] = best_action.dims();

    tracing::info!("Selected action sequence: [{horizon} steps, {action_dim} dims]");

    // Print first action
    let first_action = best_action
        .clone()
        .slice([0..1, 0..1, 0..action_dim])
        .reshape([action_dim]);
    let action_data: Vec<f32> = first_action.into_data().to_vec().unwrap();
    tracing::info!("First action: {:?}", action_data);

    tracing::info!("=== Demo Complete ===");
    Ok(())
}
