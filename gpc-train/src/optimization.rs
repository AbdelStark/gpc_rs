//! Shared optimizer utilities for training loops.

use burn::grad_clipping::GradientClipping;

use gpc_core::config::TrainingConfig;

/// Resolve an effective learning rate for a zero-based optimizer step.
pub(crate) fn learning_rate_for_step(
    base_learning_rate: f64,
    warmup_steps: usize,
    step_index: usize,
) -> f64 {
    if warmup_steps == 0 {
        return base_learning_rate;
    }

    let warmup_progress = (step_index + 1).min(warmup_steps) as f64 / warmup_steps as f64;
    base_learning_rate * warmup_progress
}

/// Resolve norm clipping configuration from the training config.
pub(crate) fn gradient_clipping(training_config: &TrainingConfig) -> Option<GradientClipping> {
    (training_config.grad_clip_norm > 0.0).then_some(GradientClipping::Norm(
        training_config.grad_clip_norm as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use gpc_core::config::TrainingConfig;

    #[test]
    fn learning_rate_without_warmup_is_constant() {
        let sequence: Vec<f64> = (0..4)
            .map(|step| learning_rate_for_step(1e-3, 0, step))
            .collect();
        assert_eq!(sequence, vec![1e-3, 1e-3, 1e-3, 1e-3]);
    }

    #[test]
    fn learning_rate_warmup_reaches_base_rate() {
        let sequence: Vec<f64> = (0..5)
            .map(|step| learning_rate_for_step(1e-3, 4, step))
            .collect();

        assert_eq!(sequence, vec![0.00025, 0.0005, 0.00075, 0.001, 0.001]);
    }

    #[test]
    fn gradient_clipping_is_disabled_for_non_positive_norms() {
        let config = TrainingConfig {
            grad_clip_norm: 0.0,
            ..Default::default()
        };
        assert!(gradient_clipping(&config).is_none());

        let config = TrainingConfig {
            grad_clip_norm: -1.0,
            ..Default::default()
        };
        assert!(gradient_clipping(&config).is_none());
    }

    #[test]
    fn gradient_clipping_is_enabled_for_positive_norms() {
        let config = TrainingConfig {
            grad_clip_norm: 0.75,
            ..Default::default()
        };

        match gradient_clipping(&config) {
            Some(GradientClipping::Norm(value)) => assert!((value - 0.75).abs() < f32::EPSILON),
            _ => panic!("expected norm clipping to be enabled"),
        }
    }
}
