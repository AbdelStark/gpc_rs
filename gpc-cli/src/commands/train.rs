//! Train command implementation.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use clap::Args;

use gpc_core::config::{GpcConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_train::data::{GpcDataset, GpcDatasetConfig};
use gpc_train::{PolicyTrainer, WorldModelTrainer};

type TrainBackend = Autodiff<NdArray>;

/// Arguments for the train command.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to configuration file (JSON).
    #[arg(short, long)]
    config: Option<String>,

    /// Path to training data directory.
    #[arg(short, long)]
    data: Option<String>,

    /// Component to train: "policy", "world-model", or "all".
    #[arg(long, default_value = "all")]
    component: String,

    /// Number of training epochs (overrides config).
    #[arg(long)]
    epochs: Option<usize>,

    /// Use synthetic data for testing.
    #[arg(long)]
    synthetic: bool,

    /// Output directory for checkpoints.
    #[arg(short, long, default_value = "checkpoints")]
    output: String,

    /// World model prediction horizon for phase 2.
    #[arg(long, default_value = "8")]
    horizon: usize,
}

/// Run the train command.
pub fn run_train(args: TrainArgs) -> Result<()> {
    let config = if let Some(config_path) = &args.config {
        let data = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&data)?
    } else {
        GpcConfig::default()
    };

    let mut training_config = config.training.clone();
    if let Some(epochs) = args.epochs {
        training_config.num_epochs = epochs;
    }

    let device = <TrainBackend as burn::prelude::Backend>::Device::default();

    // Load or generate dataset
    let dataset_config = GpcDatasetConfig {
        data_dir: args.data.clone().unwrap_or_else(|| "data".to_string()),
        state_dim: config.world_model.state_dim,
        action_dim: config.policy.action_dim,
        obs_dim: config.policy.obs_dim,
        obs_horizon: config.policy.obs_horizon,
        pred_horizon: config.policy.pred_horizon,
    };

    let dataset = if args.synthetic {
        tracing::info!("Generating synthetic dataset");
        GpcDataset::generate_synthetic(dataset_config, 50, 100, training_config.seed)
    } else if let Some(data_path) = &args.data {
        let json_path = format!("{data_path}/episodes.json");
        GpcDataset::from_json(&json_path, dataset_config)?
    } else {
        tracing::info!("No data path specified, using synthetic data");
        GpcDataset::generate_synthetic(dataset_config, 50, 100, training_config.seed)
    };

    tracing::info!(
        "Dataset loaded: {} episodes, {} transitions",
        dataset.num_episodes(),
        dataset.num_transitions()
    );

    std::fs::create_dir_all(&args.output)?;

    match args.component.as_str() {
        "world-model" | "wm" => {
            train_world_model(
                &training_config,
                &config.world_model,
                &dataset,
                &device,
                &args,
            )?;
        }
        "policy" => {
            train_policy(&training_config, &config.policy, &dataset, &device)?;
        }
        "all" => {
            train_world_model(
                &training_config,
                &config.world_model,
                &dataset,
                &device,
                &args,
            )?;
            train_policy(&training_config, &config.policy, &dataset, &device)?;
        }
        other => {
            anyhow::bail!("Unknown component: {other}. Use 'policy', 'world-model', or 'all'.");
        }
    }

    tracing::info!("Training complete!");
    Ok(())
}

fn train_world_model(
    training_config: &TrainingConfig,
    world_model_config: &WorldModelConfig,
    dataset: &GpcDataset,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    args: &TrainArgs,
) -> Result<()> {
    let trainer = WorldModelTrainer::new(training_config.clone(), world_model_config.clone());

    tracing::info!("=== Phase 1: Single-step world model training ===");
    let model = trainer.train_phase1::<TrainBackend>(dataset, device);

    tracing::info!("=== Phase 2: Multi-step world model training ===");
    let _model = trainer.train_phase2::<TrainBackend>(dataset, model, args.horizon, device);

    tracing::info!("World model training complete");
    Ok(())
}

fn train_policy(
    training_config: &TrainingConfig,
    policy_config: &PolicyConfig,
    dataset: &GpcDataset,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
) -> Result<()> {
    let trainer = PolicyTrainer::new(training_config.clone(), policy_config.clone());

    tracing::info!("=== Diffusion policy training ===");
    let _model = trainer.train::<TrainBackend>(dataset, device);

    tracing::info!("Policy training complete");
    Ok(())
}
