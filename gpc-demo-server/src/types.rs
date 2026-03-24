use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(self, factor: f32) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(self) -> Self {
        let length = self.length();
        if length <= f32::EPSILON {
            Self::default()
        } else {
            self.scale(1.0 / length)
        }
    }

    pub fn distance(self, other: Self) -> f32 {
        self.sub(other).length()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Obstacle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

impl Obstacle {
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmPose {
    pub base: Vec2,
    pub elbow: Vec2,
    pub effector: Vec2,
    pub theta1: f32,
    pub theta2: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissionSpec {
    pub id: String,
    pub title: String,
    pub eyebrow: String,
    pub summary: String,
    pub accent: String,
    pub difficulty: String,
    pub start_angles: [f32; 2],
    pub goal: Vec2,
    pub obstacles: Vec<Obstacle>,
    pub max_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateSummary {
    pub rank: usize,
    pub reward: f32,
    pub clearance: f32,
    pub terminal_distance: f32,
    pub effector_path: Vec<Vec2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanningFrame {
    pub step: usize,
    pub pose: ArmPose,
    pub executed_path: Vec<Vec2>,
    pub policy_path: Vec<Vec2>,
    pub ranked_path: Vec<Vec2>,
    pub optimized_path: Vec<Vec2>,
    pub candidates: Vec<CandidateSummary>,
    pub selected_action: [f32; 2],
    pub goal_distance: f32,
    pub min_clearance: f32,
    pub world_model_error: f32,
    pub reward_mean: f32,
    pub reward_best: f32,
    pub reward_spread: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissionSummary {
    pub success: bool,
    pub final_goal_distance: f32,
    pub min_clearance: f32,
    pub average_world_error: f32,
    pub executed_steps: usize,
    pub mode: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissionPlayback {
    pub mission: MissionSpec,
    pub frames: Vec<PlanningFrame>,
    pub summary: MissionSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeOverview {
    pub dataset_episodes: usize,
    pub dataset_transitions: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    pub obs_horizon: usize,
    pub pred_horizon: usize,
    pub bootstrap_ms: u128,
    pub world_loss_curve: Vec<f32>,
    pub policy_loss_curve: Vec<f32>,
    pub recommended_candidates: usize,
    pub recommended_opt_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub overview: RuntimeOverview,
    pub missions: Vec<MissionSpec>,
}
