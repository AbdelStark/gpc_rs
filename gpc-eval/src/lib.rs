//! Evaluation strategies for Generative Robot Policies.
//!
//! Implements GPC-RANK (trajectory ranking via world model scoring)
//! and GPC-OPT (gradient-based trajectory optimization).
//!
//! # GPC-RANK
//!
//! 1. Sample K candidate action sequences from the diffusion policy.
//! 2. Roll out each through the world model to get predicted states.
//! 3. Score each trajectory with the reward function.
//! 4. Select the trajectory with the highest reward.
//!
//! # GPC-OPT
//!
//! 1. Sample a single initial action sequence from the policy.
//! 2. Iteratively optimize via gradient ascent on the reward:
//!    `a ← a + η · ∇_a R(W(s, a))`
//! 3. Return the optimized action sequence.

pub mod gpc_opt;
pub mod gpc_rank;

pub use gpc_opt::{
    AutodiffGpcOpt, AutodiffGpcOptBuilder, GpcOpt, GpcOptBuilder, GpcOptStepTrace, GpcOptTrace,
};
pub use gpc_rank::{GpcRank, GpcRankBuilder, GpcRankTrace};
