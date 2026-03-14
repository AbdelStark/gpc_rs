//! Training orchestration for GPC models.
//!
//! Handles the two-phase training pipeline:
//! - Phase 1: Single-step world model warmup
//! - Phase 2: Multi-step world model training
//! - Diffusion policy training (independent)
