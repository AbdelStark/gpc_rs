//! Checkpoint save/load for Burn models.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Metadata stored alongside a model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model type identifier.
    pub model_type: String,
    /// Training epoch at which checkpoint was saved.
    pub epoch: usize,
    /// Training loss at checkpoint time.
    pub loss: f64,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Configuration used for training (JSON).
    pub config_json: String,
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

    let meta_path = dir.join(format!("{name}.meta.json"));
    let meta_json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(&meta_path, meta_json)?;

    tracing::info!("Checkpoint saved to {}", weights_path.display());
    Ok(weights_path)
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
    let meta_path = if path.extension().is_some_and(|e| e == "json") {
        path.to_path_buf()
    } else {
        path.with_extension("meta.json")
    };
    let data = std::fs::read_to_string(&meta_path)?;
    let metadata: CheckpointMetadata = serde_json::from_str(&data)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_serialization() {
        let meta = CheckpointMetadata {
            model_type: "world_model".to_string(),
            epoch: 100,
            loss: 0.001,
            timestamp: "2026-03-15T00:00:00Z".to_string(),
            config_json: "{}".to_string(),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let recovered: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.epoch, 100);
        assert_eq!(recovered.model_type, "world_model");
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let dir = std::env::temp_dir().join("gpc_test_checkpoint");
        let _ = std::fs::remove_dir_all(&dir);

        let meta = CheckpointMetadata {
            model_type: "test".to_string(),
            epoch: 1,
            loss: 0.5,
            timestamp: "2026-03-15T00:00:00Z".to_string(),
            config_json: "{}".to_string(),
        };

        let weights = vec![1u8, 2, 3, 4, 5];
        let path = save_checkpoint(&weights, &meta, &dir, "test_model").unwrap();

        let loaded = load_checkpoint_bytes(&path).unwrap();
        assert_eq!(loaded, weights);

        let loaded_meta = load_metadata(&path).unwrap();
        assert_eq!(loaded_meta.epoch, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
