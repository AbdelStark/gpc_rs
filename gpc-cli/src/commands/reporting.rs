//! Shared helpers for machine-readable command output.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub(super) fn default_train_report_path(output_dir: impl AsRef<Path>) -> PathBuf {
    output_dir.as_ref().join("train_report.json")
}

pub(super) fn write_json_report<T: Serialize>(path: impl AsRef<Path>, report: &T) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub(super) fn read_json_report<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}
