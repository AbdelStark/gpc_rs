//! Diffusion-based action policy implementation.
//!
//! Implements the generative action policy that produces candidate
//! action sequences given observational input, using DDPM (Denoising
//! Diffusion Probabilistic Models).
//!
//! # Architecture
//!
//! The policy uses a conditional denoising network that takes:
//! - Noisy action sequence `[batch, pred_horizon * action_dim]`
//! - Observation conditioning `[batch, obs_horizon * obs_dim]`
//! - Diffusion timestep embedding
//!
//! And predicts the noise to remove, iteratively denoising pure
//! Gaussian noise into a valid action sequence.

pub mod network;
pub mod policy;

pub use network::{DenoisingNetwork, DenoisingNetworkConfig};
pub use policy::{DiffusionPolicy, DiffusionPolicyConfig};
