//! Core abstractions for Generative Robot Policies.
//!
//! Provides foundational traits, tensor utilities, and shared types
//! used across all GPC crates. This includes:
//!
//! - [`error`]: Typed error handling via [`GpcError`] and [`Result`].
//! - [`types`]: Data structures for observations, actions, states, and trajectories.
//! - [`config`]: Hyperparameter configurations for all GPC components.
//! - [`traits`]: Core abstractions ([`Policy`], [`WorldModel`], [`Evaluator`]).
//! - [`noise`]: DDPM / EDM noise scheduling utilities.
//! - [`tensor_utils`]: Tensor helper functions for common operations.

pub mod config;
pub mod error;
pub mod noise;
pub mod tensor_utils;
pub mod traits;
pub mod types;

pub use error::{GpcError, Result};
