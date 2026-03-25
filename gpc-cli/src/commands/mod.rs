//! CLI command implementations.

mod benchmark;
mod checkpoint;
mod demo;
mod demo_pipeline;
mod demo_tui;
mod eval;
mod init_config;
mod reporting;
mod train;

pub use benchmark::{BenchmarkArgs, run_benchmark};
pub use checkpoint::{CheckpointArgs, run_checkpoint};
pub use demo::{DemoArgs, run_demo};
pub use eval::{EvalArgs, run_eval};
pub use init_config::{InitConfigArgs, run_init_config};
pub use train::{TrainArgs, run_train};
