//! Train command implementation.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use clap::Args;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use gpc_core::config::{GpcConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_train::data::{GpcDataset, GpcDatasetConfig};
use gpc_train::{PolicyTrainer, PolicyTrainingResult, WorldModelTrainer, WorldModelTrainingResult};

use gpc_compat::{
    CheckpointArtifact, CheckpointFormat, CheckpointKind, CheckpointMetadata,
    load_world_model_checkpoint, save_policy_checkpoint, save_world_model_checkpoint,
};

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

    let artifacts = match args.component.as_str() {
        "world-model" | "wm" => train_world_model(
            &training_config,
            &config.world_model,
            &dataset,
            &device,
            &args,
        )?,
        "policy" => train_policy(
            &training_config,
            &config.policy,
            &dataset,
            &device,
            &args.output,
        )?,
        "all" => {
            let mut artifacts = train_world_model(
                &training_config,
                &config.world_model,
                &dataset,
                &device,
                &args,
            )?;
            artifacts.extend(train_policy(
                &training_config,
                &config.policy,
                &dataset,
                &device,
                &args.output,
            )?);
            artifacts
        }
        other => {
            anyhow::bail!("Unknown component: {other}. Use 'policy', 'world-model', or 'all'.");
        }
    };

    log_artifacts(&artifacts);

    tracing::info!("Training complete!");
    Ok(())
}

fn train_world_model(
    training_config: &TrainingConfig,
    world_model_config: &WorldModelConfig,
    dataset: &GpcDataset,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    args: &TrainArgs,
) -> Result<Vec<CheckpointArtifact>> {
    let trainer = WorldModelTrainer::new(training_config.clone(), world_model_config.clone());

    tracing::info!("=== Phase 1: Single-step world model training ===");
    let phase1_result = trainer.train_phase1_with_summary::<TrainBackend>(dataset, device);
    let phase1_artifact = save_world_model_stage(
        phase1_result,
        world_model_config,
        "world_model_phase1",
        &args.output,
    )?;

    let phase1_model =
        load_world_model_checkpoint::<TrainBackend>(&phase1_artifact.checkpoint_path, device)?;

    tracing::info!("=== Phase 2: Multi-step world model training ===");
    let phase2_result = trainer.train_phase2_with_summary::<TrainBackend>(
        dataset,
        phase1_model,
        args.horizon,
        device,
    );
    let phase2_artifact = save_world_model_stage(
        phase2_result,
        world_model_config,
        "world_model_final",
        &args.output,
    )?;

    tracing::info!("World model training complete");
    Ok(vec![phase1_artifact, phase2_artifact])
}

fn train_policy(
    training_config: &TrainingConfig,
    policy_config: &PolicyConfig,
    dataset: &GpcDataset,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    output_dir: &str,
) -> Result<Vec<CheckpointArtifact>> {
    let trainer = PolicyTrainer::new(training_config.clone(), policy_config.clone());

    tracing::info!("=== Diffusion policy training ===");
    let result = trainer.train_with_summary::<TrainBackend>(dataset, device);
    let artifact = save_policy_stage(result, policy_config, "policy_final", output_dir)?;

    tracing::info!("Policy training complete");
    Ok(vec![artifact])
}

fn save_policy_stage(
    result: PolicyTrainingResult<TrainBackend>,
    policy_config: &PolicyConfig,
    name: &str,
    output_dir: &str,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        CheckpointKind::Policy,
        result.final_epoch,
        result.final_loss,
        serde_json::to_string_pretty(policy_config)?,
    );
    let checkpoint = save_policy_checkpoint::<TrainBackend>(
        result.model,
        &metadata,
        Path::new(output_dir).join(name),
        CheckpointFormat::Bin,
    )?;
    tracing::info!(
        "Saved policy checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    Ok(checkpoint)
}

fn save_world_model_stage(
    result: WorldModelTrainingResult<TrainBackend>,
    world_model_config: &WorldModelConfig,
    name: &str,
    output_dir: &str,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        CheckpointKind::WorldModel,
        result.final_epoch,
        result.final_loss,
        serde_json::to_string_pretty(world_model_config)?,
    );
    let checkpoint = save_world_model_checkpoint::<TrainBackend>(
        result.model,
        &metadata,
        Path::new(output_dir).join(name),
        CheckpointFormat::Bin,
    )?;
    tracing::info!(
        "Saved world-model checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    Ok(checkpoint)
}

fn checkpoint_metadata(
    model_type: CheckpointKind,
    epoch: Option<usize>,
    loss: Option<f32>,
    config_json: String,
) -> CheckpointMetadata {
    CheckpointMetadata {
        model_type: model_type.as_str().to_string(),
        epoch: epoch.unwrap_or_default(),
        loss: loss.map(f64::from).unwrap_or_default(),
        timestamp: current_timestamp(),
        config_json,
    }
}

fn current_timestamp() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format!("{}Z", duration.as_secs()),
        Err(_) => "0Z".to_string(),
    }
}

fn log_artifacts(artifacts: &[CheckpointArtifact]) {
    if artifacts.is_empty() {
        tracing::warn!("No checkpoint artifacts were written");
        return;
    }

    tracing::info!("Artifacts written:");
    for artifact in artifacts {
        tracing::info!("  - {}", artifact.checkpoint_path.display());
        tracing::info!("    metadata: {}", artifact.metadata_path.display());
    }
}
