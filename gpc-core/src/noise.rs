//! Noise scheduling for DDPM (Denoising Diffusion Probabilistic Models).
//!
//! Implements the linear beta schedule and precomputes alpha values
//! used in the forward (noising) and reverse (denoising) diffusion process.

use burn::tensor::{Tensor, backend::Backend};

use crate::config::NoiseScheduleConfig;

/// Precomputed DDPM noise schedule parameters.
///
/// Stores all alpha/beta values needed for efficient forward and reverse
/// diffusion steps without recomputation.
#[derive(Debug, Clone)]
pub struct DdpmSchedule {
    /// Beta values for each timestep.
    pub betas: Vec<f64>,
    /// Cumulative product of (1 - beta).
    pub alphas_cumprod: Vec<f64>,
    /// Square root of alphas_cumprod.
    pub sqrt_alphas_cumprod: Vec<f64>,
    /// Square root of (1 - alphas_cumprod).
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    /// 1 / sqrt(alpha_t).
    pub sqrt_recip_alphas: Vec<f64>,
    /// Posterior variance for reverse process.
    pub posterior_variance: Vec<f64>,
    /// Number of diffusion timesteps.
    pub num_timesteps: usize,
}

impl Default for DdpmSchedule {
    fn default() -> Self {
        Self::new(&NoiseScheduleConfig::default())
    }
}

impl DdpmSchedule {
    /// Create a new DDPM schedule from configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.num_timesteps` is 0.
    pub fn new(config: &NoiseScheduleConfig) -> Self {
        assert!(
            config.num_timesteps > 0,
            "DdpmSchedule requires at least 1 timestep"
        );
        let n = config.num_timesteps;
        let mut betas = Vec::with_capacity(n);
        for i in 0..n {
            let beta = config.beta_start
                + (config.beta_end - config.beta_start) * (i as f64) / (n as f64 - 1.0).max(1.0);
            betas.push(beta);
        }

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut prod = 1.0;
        for &a in &alphas {
            prod *= a;
            alphas_cumprod.push(prod);
        }

        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();

        let sqrt_one_minus_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();

        let sqrt_recip_alphas: Vec<f64> = alphas.iter().map(|a| (1.0 / a).sqrt()).collect();

        let mut posterior_variance = Vec::with_capacity(n);
        posterior_variance.push(betas[0]);
        for t in 1..n {
            let var = betas[t] * (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t]);
            posterior_variance.push(var);
        }

        Self {
            betas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
            num_timesteps: n,
        }
    }

    /// Forward diffusion: add noise to clean data at timestep t.
    ///
    /// q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    ///
    /// # Arguments
    /// * `x_0` - Clean data tensor
    /// * `noise` - Random Gaussian noise (same shape as x_0)
    /// * `t` - Timestep index
    ///
    /// # Returns
    /// Noised tensor x_t
    pub fn add_noise<B: Backend, const D: usize>(
        &self,
        x_0: &Tensor<B, D>,
        noise: &Tensor<B, D>,
        t: usize,
    ) -> Tensor<B, D> {
        let sqrt_alpha = self.sqrt_alphas_cumprod[t] as f32;
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t] as f32;

        x_0.clone() * sqrt_alpha + noise.clone() * sqrt_one_minus_alpha
    }

    /// Reverse diffusion step: denoise x_t to x_{t-1}.
    ///
    /// # Arguments
    /// * `x_t` - Noisy data at timestep t
    /// * `predicted_noise` - Model's noise prediction
    /// * `t` - Current timestep
    ///
    /// # Returns
    /// Partially denoised tensor x_{t-1}
    pub fn remove_noise<B: Backend, const D: usize>(
        &self,
        x_t: &Tensor<B, D>,
        predicted_noise: &Tensor<B, D>,
        t: usize,
    ) -> Tensor<B, D> {
        let beta_t = self.betas[t] as f32;
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t] as f32;
        let sqrt_recip_alpha = self.sqrt_recip_alphas[t] as f32;

        // mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
        let coeff = beta_t / sqrt_one_minus_alpha;
        (x_t.clone() - predicted_noise.clone() * coeff) * sqrt_recip_alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NoiseScheduleConfig;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_schedule_properties() {
        let config = NoiseScheduleConfig {
            num_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        let schedule = DdpmSchedule::new(&config);

        assert_eq!(schedule.betas.len(), 100);
        assert_eq!(schedule.alphas_cumprod.len(), 100);

        // Alpha cumprod should be monotonically decreasing
        for i in 1..100 {
            assert!(schedule.alphas_cumprod[i] < schedule.alphas_cumprod[i - 1]);
        }

        // At t=0, almost no noise
        assert_relative_eq!(schedule.alphas_cumprod[0], 1.0 - 1e-4, epsilon = 1e-6);

        // At t=T-1, significant noise (alpha_cumprod decreases)
        assert!(schedule.alphas_cumprod[99] < schedule.alphas_cumprod[0]);
    }

    #[test]
    fn test_add_noise_at_t0_preserves_signal() {
        let config = NoiseScheduleConfig::default();
        let schedule = DdpmSchedule::new(&config);
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let x_0 = Tensor::<TestBackend, 2>::ones([1, 4], &device);
        let noise = Tensor::<TestBackend, 2>::ones([1, 4], &device) * 0.5;

        let x_t = schedule.add_noise(&x_0, &noise, 0);
        // At t=0, sqrt_alpha ≈ 1.0, so x_t ≈ x_0
        let diff = (x_t - x_0).abs().max().into_scalar();
        assert!(
            diff < 0.01,
            "At t=0 noise should be minimal, got diff={diff}"
        );
    }

    #[test]
    fn test_single_timestep_schedule() {
        let config = NoiseScheduleConfig {
            num_timesteps: 1,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        let schedule = DdpmSchedule::new(&config);
        assert_eq!(schedule.betas.len(), 1);
        assert_eq!(schedule.alphas_cumprod.len(), 1);
    }

    #[test]
    fn test_add_noise_at_final_step_dominated_by_noise() {
        let config = NoiseScheduleConfig {
            num_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        let schedule = DdpmSchedule::new(&config);
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let x_0 = Tensor::<TestBackend, 2>::ones([1, 4], &device);
        let noise = Tensor::<TestBackend, 2>::ones([1, 4], &device) * 10.0;

        let x_t = schedule.add_noise(&x_0, &noise, 999);
        // At final step, noise dominates
        let max_val = x_t.max().into_scalar();
        assert!(max_val > 5.0, "At final step noise should dominate");
    }
}
