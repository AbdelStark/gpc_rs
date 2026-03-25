//! Shared helpers for machine-readable command output.

use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub(super) fn write_json_report<T: Serialize>(path: &str, report: &T) -> Result<()> {
    let path = PathBuf::from(path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub(super) fn read_json_report<T: DeserializeOwned>(path: &str) -> Result<T> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}
