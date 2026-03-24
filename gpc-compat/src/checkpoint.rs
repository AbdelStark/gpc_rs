//! Checkpoint save/load for Burn models.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder};
use serde::{Deserialize, Serialize};

use gpc_core::config::{PolicyConfig, WorldModelConfig};

/// Supported checkpoint formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// Bincode `.bin` checkpoints.
    Bin,
    /// Named MessagePack `.mpk` checkpoints.
    Mpk,
}

impl CheckpointFormat {
    /// Extension used by the format.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Bin => "bin",
            Self::Mpk => "mpk",
        }
    }

    /// Infer the format from a file path.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "bin" => Some(Self::Bin),
            "mpk" => Some(Self::Mpk),
            _ => None,
        }
    }

    /// Swap to the other supported format.
    pub fn opposite(self) -> Self {
        match self {
            Self::Bin => Self::Mpk,
            Self::Mpk => Self::Bin,
        }
    }
}

/// Known checkpoint model kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointKind {
    /// Diffusion policy checkpoint.
    Policy,
    /// State world model checkpoint.
    WorldModel,
}

impl CheckpointKind {
    /// Parse a model kind from metadata.
    pub fn from_metadata(metadata: &CheckpointMetadata) -> gpc_core::Result<Self> {
        match metadata.model_type.trim().to_ascii_lowercase().as_str() {
            "policy" | "diffusion_policy" | "diffusion-policy" => Ok(Self::Policy),
            "world_model" | "world-model" | "state_world_model" | "state-world-model" => {
                Ok(Self::WorldModel)
            }
            other => Err(gpc_core::GpcError::Config(format!(
                "unsupported checkpoint model_type: {other}"
            ))),
        }
    }

    /// Canonical string used in metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Policy => "policy",
            Self::WorldModel => "world_model",
        }
    }
}

/// Metadata stored alongside a model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model type identifier.
    pub model_type: String,
    /// Training epoch at which checkpoint was saved.
    pub epoch: usize,
    /// Training loss at checkpoint time.
    pub loss: f64,
    /// Timestamp (ISO 8601 or equivalent human-readable time string).
    pub timestamp: String,
    /// Configuration used for training (JSON).
    pub config_json: String,
}

/// Saved checkpoint artifact paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointArtifact {
    /// Model checkpoint path.
    pub checkpoint_path: PathBuf,
    /// Metadata sidecar path.
    pub metadata_path: PathBuf,
}

impl CheckpointArtifact {
    fn new(checkpoint_path: PathBuf) -> Self {
        let metadata_path = metadata_path_for(&checkpoint_path);
        Self {
            checkpoint_path,
            metadata_path,
        }
    }
}

/// Save model weights as raw bytes and metadata as JSON.
pub fn save_checkpoint(
    weights: &[u8],
    metadata: &CheckpointMetadata,
    dir: &Path,
    name: &str,
) -> gpc_core::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;

    let weights_path = dir.join(format!("{name}.bin"));
    std::fs::write(&weights_path, weights)?;

    let meta_path = metadata_path_for(&weights_path);
    let meta_json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(&meta_path, meta_json)?;

    tracing::info!("Checkpoint saved to {}", weights_path.display());
    Ok(weights_path)
}

/// Save a diffusion policy checkpoint.
pub fn save_policy_checkpoint<B: Backend>(
    model: gpc_policy::DiffusionPolicy<B>,
    metadata: &CheckpointMetadata,
    path: impl AsRef<Path>,
    format: CheckpointFormat,
) -> gpc_core::Result<CheckpointArtifact> {
    save_module(model, metadata, path.as_ref(), format)
}

/// Save a world model checkpoint.
pub fn save_world_model_checkpoint<B: Backend>(
    model: gpc_world::StateWorldModel<B>,
    metadata: &CheckpointMetadata,
    path: impl AsRef<Path>,
    format: CheckpointFormat,
) -> gpc_core::Result<CheckpointArtifact> {
    save_module(model, metadata, path.as_ref(), format)
}

/// Convert a diffusion policy checkpoint between supported formats.
pub fn convert_policy_checkpoint<B: Backend>(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    output_format: CheckpointFormat,
    device: &B::Device,
) -> gpc_core::Result<CheckpointArtifact> {
    convert_module::<B, gpc_policy::DiffusionPolicy<B>, PolicyConfig>(
        input_path.as_ref(),
        output_path.as_ref(),
        output_format,
        device,
        |config| {
            let config = gpc_policy::DiffusionPolicyConfig::from_policy_config(&config);
            config.init::<B>(device)
        },
    )
}

/// Convert a world model checkpoint between supported formats.
pub fn convert_world_model_checkpoint<B: Backend>(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    output_format: CheckpointFormat,
    device: &B::Device,
) -> gpc_core::Result<CheckpointArtifact> {
    convert_module::<B, gpc_world::StateWorldModel<B>, WorldModelConfig>(
        input_path.as_ref(),
        output_path.as_ref(),
        output_format,
        device,
        |config| {
            let config =
                gpc_world::world_model::StateWorldModelConfig::from_world_model_config(&config);
            config.init::<B>(device)
        },
    )
}

/// Load a diffusion policy checkpoint from a file.
pub fn load_policy_checkpoint<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> gpc_core::Result<gpc_policy::DiffusionPolicy<B>> {
    load_module_from_file::<B, gpc_policy::DiffusionPolicy<B>, PolicyConfig>(
        path.as_ref(),
        device,
        |config| {
            let config = gpc_policy::DiffusionPolicyConfig::from_policy_config(&config);
            config.init::<B>(device)
        },
    )
}

/// Load a world model checkpoint from a file.
pub fn load_world_model_checkpoint<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> gpc_core::Result<gpc_world::StateWorldModel<B>> {
    load_module_from_file::<B, gpc_world::StateWorldModel<B>, WorldModelConfig>(
        path.as_ref(),
        device,
        |config| {
            let config =
                gpc_world::world_model::StateWorldModelConfig::from_world_model_config(&config);
            config.init::<B>(device)
        },
    )
}

/// Load raw checkpoint bytes from a file.
pub fn load_checkpoint_bytes(path: &Path) -> gpc_core::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    tracing::info!(
        "Checkpoint loaded from {} ({} bytes)",
        path.display(),
        bytes.len()
    );
    Ok(bytes)
}

/// Load checkpoint metadata.
pub fn load_metadata(path: &Path) -> gpc_core::Result<CheckpointMetadata> {
    let meta_path = metadata_path_for(&checkpoint_path(path));
    let data = std::fs::read_to_string(&meta_path)?;
    let metadata: CheckpointMetadata = serde_json::from_str(&data)?;
    Ok(metadata)
}

/// Normalize a metadata path or checkpoint path to the checkpoint path.
pub fn checkpoint_path(path: &Path) -> PathBuf {
    if is_metadata_path(path) {
        let stem = strip_metadata_suffix(path);
        let bin_path = stem.with_extension(CheckpointFormat::Bin.extension());
        if bin_path.exists() {
            return bin_path;
        }

        let mpk_path = stem.with_extension(CheckpointFormat::Mpk.extension());
        if mpk_path.exists() {
            return mpk_path;
        }

        stem
    } else {
        path.to_path_buf()
    }
}

/// Compute the metadata sidecar path for a checkpoint path.
pub fn metadata_path_for(path: &Path) -> PathBuf {
    let checkpoint_path = checkpoint_path(path);
    checkpoint_path.with_extension("meta.json")
}

fn save_module<B, M>(
    model: M,
    metadata: &CheckpointMetadata,
    path: &Path,
    format: CheckpointFormat,
) -> gpc_core::Result<CheckpointArtifact>
where
    B: Backend,
    M: Module<B>,
{
    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;

    let checkpoint_path = match format {
        CheckpointFormat::Bin => {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            model
                .save_file(path.to_path_buf(), &recorder)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?;
            path.with_extension(format.extension())
        }
        CheckpointFormat::Mpk => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            model
                .save_file(path.to_path_buf(), &recorder)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?;
            path.with_extension(format.extension())
        }
    };

    let meta_path = metadata_path_for(&checkpoint_path);
    let meta_json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(&meta_path, meta_json)?;

    tracing::info!("Checkpoint saved to {}", checkpoint_path.display());
    tracing::info!("Checkpoint metadata saved to {}", meta_path.display());

    Ok(CheckpointArtifact::new(checkpoint_path))
}

fn convert_module<B, M, C>(
    input_path: &Path,
    output_path: &Path,
    output_format: CheckpointFormat,
    device: &B::Device,
    init: impl FnOnce(C) -> M,
) -> gpc_core::Result<CheckpointArtifact>
where
    B: Backend,
    M: Module<B>,
    C: serde::de::DeserializeOwned,
{
    let checkpoint_path = checkpoint_path(input_path);
    let metadata = load_metadata(&checkpoint_path)?;
    let config: C = serde_json::from_str(&metadata.config_json)?;
    let template = init(config);
    let input_format = CheckpointFormat::from_path(&checkpoint_path)
        .ok_or_else(|| gpc_core::GpcError::Config("unsupported checkpoint extension".into()))?;

    let loaded = match input_format {
        CheckpointFormat::Bin => {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            template
                .load_file(checkpoint_path.clone(), &recorder, device)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?
        }
        CheckpointFormat::Mpk => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            template
                .load_file(checkpoint_path.clone(), &recorder, device)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?
        }
    };

    save_module::<B, M>(loaded, &metadata, output_path, output_format)
}

fn load_module_from_file<B, M, C>(
    input_path: &Path,
    device: &B::Device,
    init: impl FnOnce(C) -> M,
) -> gpc_core::Result<M>
where
    B: Backend,
    M: Module<B>,
    C: serde::de::DeserializeOwned,
{
    let checkpoint_path = checkpoint_path(input_path);
    let metadata = load_metadata(&checkpoint_path)?;
    let config: C = serde_json::from_str(&metadata.config_json)?;
    let template = init(config);
    let input_format = CheckpointFormat::from_path(&checkpoint_path)
        .ok_or_else(|| gpc_core::GpcError::Config("unsupported checkpoint extension".into()))?;

    let loaded = match input_format {
        CheckpointFormat::Bin => {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            template
                .load_file(checkpoint_path, &recorder, device)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?
        }
        CheckpointFormat::Mpk => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            template
                .load_file(checkpoint_path, &recorder, device)
                .map_err(|err| gpc_core::GpcError::Checkpoint(err.to_string()))?
        }
    };

    Ok(loaded)
}

fn is_metadata_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".meta.json"))
}

fn strip_metadata_suffix(path: &Path) -> PathBuf {
    let without_json = path.with_extension("");
    without_json.with_extension("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Backend;

    type TestBackend = NdArray;

    fn policy_metadata(config: &PolicyConfig) -> CheckpointMetadata {
        CheckpointMetadata {
            model_type: CheckpointKind::Policy.as_str().to_string(),
            epoch: 3,
            loss: 0.125,
            timestamp: "2026-03-15T00:00:00Z".to_string(),
            config_json: serde_json::to_string(config).unwrap(),
        }
    }

    fn world_metadata(config: &WorldModelConfig) -> CheckpointMetadata {
        CheckpointMetadata {
            model_type: CheckpointKind::WorldModel.as_str().to_string(),
            epoch: 4,
            loss: 0.25,
            timestamp: "2026-03-15T00:00:00Z".to_string(),
            config_json: serde_json::to_string(config).unwrap(),
        }
    }

    #[test]
    fn test_metadata_path_helpers_accept_bin_and_mpk() {
        assert_eq!(
            metadata_path_for(Path::new("/tmp/model.bin")),
            PathBuf::from("/tmp/model.meta.json")
        );
        assert_eq!(
            metadata_path_for(Path::new("/tmp/model.mpk")),
            PathBuf::from("/tmp/model.meta.json")
        );
        assert_eq!(
            checkpoint_path(Path::new("/tmp/model.meta.json")),
            PathBuf::from("/tmp/model")
        );
    }

    #[test]
    fn test_metadata_serialization() {
        let meta = policy_metadata(&PolicyConfig::default());
        let json = serde_json::to_string(&meta).unwrap();
        let recovered: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.epoch, 3);
        assert_eq!(recovered.model_type, "policy");
    }

    #[test]
    fn test_save_and_load_checkpoint_bytes() {
        let dir = std::env::temp_dir().join(format!("gpc_test_checkpoint_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let meta = policy_metadata(&PolicyConfig::default());
        let weights = vec![1u8, 2, 3, 4, 5];
        let path = save_checkpoint(&weights, &meta, &dir, "test_model").unwrap();

        let loaded = load_checkpoint_bytes(&path).unwrap();
        assert_eq!(loaded, weights);

        let loaded_meta = load_metadata(&path).unwrap();
        assert_eq!(loaded_meta.epoch, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_policy_bin_mpk_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let base_dir = std::env::temp_dir().join(format!(
            "gpc_policy_roundtrip_{}_{}",
            std::process::id(),
            "bin"
        ));
        let _ = std::fs::remove_dir_all(&base_dir);
        std::fs::create_dir_all(&base_dir).unwrap();

        let config = gpc_policy::DiffusionPolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            hidden_dim: 16,
            time_embed_dim: 8,
            num_res_blocks: 1,
            diffusion_steps: 4,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        let model = config.init::<TestBackend>(&device);
        let metadata_config = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            action_horizon: 1,
            hidden_dim: 16,
            num_res_blocks: 1,
            noise_schedule: gpc_core::config::NoiseScheduleConfig {
                num_timesteps: 4,
                beta_start: 1e-4,
                beta_end: 0.02,
            },
        };

        let source = save_policy_checkpoint(
            model,
            &policy_metadata(&metadata_config),
            base_dir.join("policy_source"),
            CheckpointFormat::Bin,
        )
        .unwrap();
        let source_bytes = std::fs::read(&source.checkpoint_path).unwrap();

        let mpk = convert_policy_checkpoint::<TestBackend>(
            &source.checkpoint_path,
            base_dir.join("policy_converted"),
            CheckpointFormat::Mpk,
            &device,
        )
        .unwrap();
        assert!(mpk.checkpoint_path.ends_with("policy_converted.mpk"));
        assert!(mpk.metadata_path.ends_with("policy_converted.meta.json"));
        assert!(load_metadata(&mpk.checkpoint_path).is_ok());

        let roundtrip = convert_policy_checkpoint::<TestBackend>(
            &mpk.checkpoint_path,
            base_dir.join("policy_roundtrip"),
            CheckpointFormat::Bin,
            &device,
        )
        .unwrap();
        let roundtrip_bytes = std::fs::read(&roundtrip.checkpoint_path).unwrap();

        assert_eq!(source_bytes, roundtrip_bytes);
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn test_world_model_bin_mpk_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let base_dir = std::env::temp_dir().join(format!(
            "gpc_world_roundtrip_{}_{}",
            std::process::id(),
            "bin"
        ));
        let _ = std::fs::remove_dir_all(&base_dir);
        std::fs::create_dir_all(&base_dir).unwrap();

        let config = gpc_world::world_model::StateWorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
        };
        let model = config.init::<TestBackend>(&device);
        let metadata_config = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            dropout: 0.0,
        };

        let source = save_world_model_checkpoint(
            model,
            &world_metadata(&metadata_config),
            base_dir.join("world_source"),
            CheckpointFormat::Bin,
        )
        .unwrap();
        let source_bytes = std::fs::read(&source.checkpoint_path).unwrap();

        let mpk = convert_world_model_checkpoint::<TestBackend>(
            &source.checkpoint_path,
            base_dir.join("world_converted"),
            CheckpointFormat::Mpk,
            &device,
        )
        .unwrap();
        assert!(mpk.checkpoint_path.ends_with("world_converted.mpk"));
        assert!(load_metadata(&mpk.checkpoint_path).is_ok());

        let roundtrip = convert_world_model_checkpoint::<TestBackend>(
            &mpk.checkpoint_path,
            base_dir.join("world_roundtrip"),
            CheckpointFormat::Bin,
            &device,
        )
        .unwrap();
        let roundtrip_bytes = std::fs::read(&roundtrip.checkpoint_path).unwrap();

        assert_eq!(source_bytes, roundtrip_bytes);
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn test_checkpoint_kind_parsing() {
        assert_eq!(
            CheckpointKind::from_metadata(&policy_metadata(&PolicyConfig::default())).unwrap(),
            CheckpointKind::Policy
        );
        assert_eq!(
            CheckpointKind::from_metadata(&world_metadata(&WorldModelConfig::default())).unwrap(),
            CheckpointKind::WorldModel
        );
    }
}
