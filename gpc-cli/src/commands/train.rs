//! Train command implementation.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use clap::Args;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use gpc_core::config::{GpcConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_train::data::{GpcDataset, GpcDatasetConfig};
use gpc_train::{
    PolicyTrainer, PolicyValidationSummary, WorldModelTrainer, WorldModelValidationSummary,
};

use gpc_compat::{
    CheckpointArtifact, CheckpointFormat, CheckpointKind, CheckpointMetadata,
    save_policy_checkpoint, save_world_model_checkpoint,
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

    /// Fraction of episodes to hold out for validation.
    #[arg(long, default_value = "0.0")]
    validation_split: f32,
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

    let dataset_split = if args.validation_split > 0.0 {
        Some(dataset.split(args.validation_split, training_config.seed)?)
    } else {
        None
    };
    let train_dataset = dataset_split
        .as_ref()
        .map(|split| &split.train)
        .unwrap_or(&dataset);
    let validation_dataset = dataset_split.as_ref().and_then(|split| {
        if split.validation.num_episodes() > 0 {
            Some(&split.validation)
        } else {
            None
        }
    });

    if let Some(split) = &dataset_split {
        tracing::info!(
            train_episodes = split.train.num_episodes(),
            validation_episodes = split.validation.num_episodes(),
            validation_split = args.validation_split,
            "dataset split into train and validation subsets"
        );
    }

    let artifacts = match args.component.as_str() {
        "world-model" | "wm" => train_world_model(
            &training_config,
            &config.world_model,
            train_dataset,
            validation_dataset,
            &device,
            &args,
        )?,
        "policy" => train_policy(
            &training_config,
            &config.policy,
            train_dataset,
            validation_dataset,
            &device,
            &args.output,
        )?,
        "all" => {
            let mut artifacts = train_world_model(
                &training_config,
                &config.world_model,
                train_dataset,
                validation_dataset,
                &device,
                &args,
            )?;
            artifacts.extend(train_policy(
                &training_config,
                &config.policy,
                train_dataset,
                validation_dataset,
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
    validation_dataset: Option<&GpcDataset>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    args: &TrainArgs,
) -> Result<Vec<CheckpointArtifact>> {
    let trainer = WorldModelTrainer::new(training_config.clone(), world_model_config.clone());

    tracing::info!("=== Phase 1: Single-step world model training ===");
    let phase1_summary = trainer.train_phase1_with_validation_summary::<TrainBackend>(
        dataset,
        validation_dataset,
        device,
    );
    let phase1_artifacts = save_world_model_validation_stages(
        &phase1_summary,
        world_model_config,
        "world_model_phase1",
        &args.output,
    )?;

    let phase1_model = if validation_dataset.is_some() {
        phase1_summary.best_model.clone()
    } else {
        phase1_summary.training.model.clone()
    };

    tracing::info!("=== Phase 2: Multi-step world model training ===");
    let phase2_summary = trainer.train_phase2_with_validation_summary::<TrainBackend>(
        dataset,
        phase1_model,
        args.horizon,
        validation_dataset,
        device,
    );
    let phase2_artifacts = save_world_model_validation_stages(
        &phase2_summary,
        world_model_config,
        "world_model_final",
        &args.output,
    )?;

    tracing::info!("World model training complete");
    let mut artifacts = phase1_artifacts;
    artifacts.extend(phase2_artifacts);
    Ok(artifacts)
}

fn train_policy(
    training_config: &TrainingConfig,
    policy_config: &PolicyConfig,
    dataset: &GpcDataset,
    validation_dataset: Option<&GpcDataset>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    output_dir: &str,
) -> Result<Vec<CheckpointArtifact>> {
    let trainer = PolicyTrainer::new(training_config.clone(), policy_config.clone());

    tracing::info!("=== Diffusion policy training ===");
    let summary =
        trainer.train_with_validation_summary::<TrainBackend>(dataset, validation_dataset, device);
    let artifacts = save_policy_validation_stages(&summary, policy_config, output_dir)?;

    tracing::info!("Policy training complete");
    Ok(artifacts)
}

fn save_policy_validation_stages(
    summary: &PolicyValidationSummary<TrainBackend>,
    policy_config: &PolicyConfig,
    output_dir: &str,
) -> Result<Vec<CheckpointArtifact>> {
    let mut artifacts = Vec::new();
    artifacts.push(save_policy_stage(
        CheckpointKind::Policy,
        &summary.training.model,
        policy_config,
        "policy_final",
        output_dir,
        summary.training.final_epoch,
        summary.training.final_loss,
    )?);

    if summary.best_epoch.is_some() {
        artifacts.push(save_policy_stage(
            CheckpointKind::Policy,
            &summary.best_model,
            policy_config,
            "policy_best",
            output_dir,
            summary.best_epoch,
            summary.best_validation_loss.or(summary.training.final_loss),
        )?);
    }

    Ok(artifacts)
}

fn save_world_model_validation_stages(
    summary: &WorldModelValidationSummary<TrainBackend>,
    world_model_config: &WorldModelConfig,
    base_name: &str,
    output_dir: &str,
) -> Result<Vec<CheckpointArtifact>> {
    let mut artifacts = Vec::new();
    artifacts.push(save_world_model_stage(
        CheckpointKind::WorldModel,
        &summary.training.model,
        world_model_config,
        base_name,
        output_dir,
        summary.training.final_epoch,
        summary.training.final_loss,
    )?);

    if summary.best_epoch.is_some() {
        let best_name = if base_name.ends_with("_phase1") {
            "world_model_phase1_best"
        } else {
            "world_model_best"
        };
        artifacts.push(save_world_model_stage(
            CheckpointKind::WorldModel,
            &summary.best_model,
            world_model_config,
            best_name,
            output_dir,
            summary.best_epoch,
            summary.best_validation_loss.or(summary.training.final_loss),
        )?);
    }

    Ok(artifacts)
}

fn save_policy_stage(
    model_type: CheckpointKind,
    model: &gpc_policy::DiffusionPolicy<TrainBackend>,
    policy_config: &PolicyConfig,
    name: &str,
    output_dir: &str,
    epoch: Option<usize>,
    loss: Option<f32>,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        model_type,
        epoch,
        loss,
        serde_json::to_string_pretty(policy_config)?,
    );
    let checkpoint = save_policy_checkpoint::<TrainBackend>(
        model.clone(),
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
    model_type: CheckpointKind,
    model: &gpc_world::StateWorldModel<TrainBackend>,
    world_model_config: &WorldModelConfig,
    name: &str,
    output_dir: &str,
    epoch: Option<usize>,
    loss: Option<f32>,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        model_type,
        epoch,
        loss,
        serde_json::to_string_pretty(world_model_config)?,
    );
    let checkpoint = save_world_model_checkpoint::<TrainBackend>(
        model.clone(),
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
