//! Predictive world model implementation.
//!
//! Learns environment dynamics to simulate future states from actions,
//! enabling trajectory evaluation for GPC-RANK and GPC-OPT.
//!
//! # Architecture
//!
//! The state-based world model uses an MLP dynamics predictor:
//! - **Encoder**: Projects `[state, action]` concatenation to hidden space
//! - **Dynamics core**: Residual MLP blocks predicting state deltas
//! - **Decoder**: Projects back to state space
//!
//! Training proceeds in two phases:
//! - **Phase 1**: Single-step prediction (warmup)
//! - **Phase 2**: Multi-step rollout with joint supervision

pub mod dynamics;
pub mod reward;
pub mod world_model;

pub use dynamics::{DynamicsNetwork, DynamicsNetworkConfig};
pub use reward::{L2RewardFunction, L2RewardFunctionConfig};
pub use world_model::StateWorldModel;
