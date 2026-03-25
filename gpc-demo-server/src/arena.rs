use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::dataset::{DatasetConfig, DemoDataset, Episode};
use crate::types::{ArmPose, MissionSpec, Obstacle, Vec2};

pub const STATE_DIM: usize = 12;
pub const ACTION_DIM: usize = 2;
pub const OBS_HORIZON: usize = 2;
pub const PRED_HORIZON: usize = 6;
pub const EXECUTION_STRIDE: usize = 1;

const LINK_1: f32 = 0.78;
const LINK_2: f32 = 0.58;
const MAX_DELTA: f32 = 0.16;
const JOINT_LIMIT: f32 = 2.7;
const BASE: Vec2 = Vec2::new(0.0, -0.02);

#[derive(Clone, Debug)]
pub struct EpisodeStats {
    pub min_clearance: f32,
    pub final_goal_distance: f32,
}

#[derive(Clone, Debug)]
pub struct TrainingDataset {
    pub dataset: DemoDataset,
    pub episodes: usize,
    pub transitions: usize,
}

#[derive(Clone, Debug)]
pub struct RobotState {
    pub theta1: f32,
    pub theta2: f32,
    pub goal: Vec2,
    pub obstacles: [Obstacle; 2],
}

pub fn preset_missions() -> Vec<MissionSpec> {
    vec![
        MissionSpec {
            id: "slipstream".to_string(),
            title: "Slipstream Dock".to_string(),
            eyebrow: "physical-ai showcase".to_string(),
            summary: "A precision reach that has to flare wide, slip past a blocker pair, and settle into the dock without clipping the hazard zones.".to_string(),
            accent: "#ef7d3b".to_string(),
            difficulty: "medium".to_string(),
            start_angles: [-1.45, 1.45],
            goal: Vec2::new(0.78, 0.72),
            obstacles: vec![
                Obstacle {
                    x: 0.34,
                    y: 0.34,
                    radius: 0.17,
                },
                Obstacle {
                    x: 0.66,
                    y: 0.42,
                    radius: 0.13,
                },
            ],
            max_steps: 24,
        },
        MissionSpec {
            id: "gate".to_string(),
            title: "Gate Thread".to_string(),
            eyebrow: "world-model control".to_string(),
            summary: "The arm has to thread a narrow aperture, which makes the candidate rollouts and reward ranking easy to read in motion.".to_string(),
            accent: "#4ea699".to_string(),
            difficulty: "hard".to_string(),
            start_angles: [-1.05, 1.7],
            goal: Vec2::new(0.2, 0.96),
            obstacles: vec![
                Obstacle {
                    x: 0.08,
                    y: 0.54,
                    radius: 0.16,
                },
                Obstacle {
                    x: 0.35,
                    y: 0.57,
                    radius: 0.15,
                },
            ],
            max_steps: 28,
        },
        MissionSpec {
            id: "late-pivot".to_string(),
            title: "Late Pivot".to_string(),
            eyebrow: "robotics planning".to_string(),
            summary: "A blocker sits near the touchdown point, forcing a late wrist pivot instead of a straight-line approach.".to_string(),
            accent: "#5972d9".to_string(),
            difficulty: "medium".to_string(),
            start_angles: [-1.8, 1.15],
            goal: Vec2::new(0.94, 0.38),
            obstacles: vec![
                Obstacle {
                    x: 0.72,
                    y: 0.24,
                    radius: 0.16,
                },
                Obstacle {
                    x: 0.56,
                    y: 0.62,
                    radius: 0.14,
                },
            ],
            max_steps: 24,
        },
    ]
}

pub fn mission_state(spec: &MissionSpec) -> RobotState {
    RobotState {
        theta1: spec.start_angles[0],
        theta2: spec.start_angles[1],
        goal: spec.goal,
        obstacles: [spec.obstacles[0].clone(), spec.obstacles[1].clone()],
    }
}

pub fn build_training_dataset(
    seed: u64,
    num_episodes: usize,
    episode_length: usize,
) -> TrainingDataset {
    let config = DatasetConfig {
        state_dim: STATE_DIM,
        action_dim: ACTION_DIM,
        obs_dim: STATE_DIM,
        obs_horizon: OBS_HORIZON,
        pred_horizon: PRED_HORIZON,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(num_episodes);

    while episodes.len() < num_episodes {
        if let Some(episode) = sample_episode(&mut rng, episode_length) {
            episodes.push(episode);
        }
    }

    let dataset = DemoDataset::new(episodes, config);
    let transitions = dataset.num_transitions();

    TrainingDataset {
        dataset,
        episodes: num_episodes,
        transitions,
    }
}

pub fn forward_kinematics(theta1: f32, theta2: f32) -> ArmPose {
    let elbow = Vec2::new(
        BASE.x + LINK_1 * theta1.cos(),
        BASE.y + LINK_1 * theta1.sin(),
    );
    let wrist_angle = theta1 + theta2;
    let effector = Vec2::new(
        elbow.x + LINK_2 * wrist_angle.cos(),
        elbow.y + LINK_2 * wrist_angle.sin(),
    );

    ArmPose {
        base: BASE,
        elbow,
        effector,
        theta1,
        theta2,
    }
}

pub fn state_to_vec(state: &RobotState) -> Vec<f32> {
    let pose = forward_kinematics(state.theta1, state.theta2);

    vec![
        state.theta1,
        state.theta2,
        pose.effector.x,
        pose.effector.y,
        state.goal.x,
        state.goal.y,
        state.obstacles[0].x,
        state.obstacles[0].y,
        state.obstacles[0].radius,
        state.obstacles[1].x,
        state.obstacles[1].y,
        state.obstacles[1].radius,
    ]
}

pub fn effector_from_slice(values: &[f32]) -> Vec2 {
    Vec2::new(values[2], values[3])
}

pub fn goal_distance_from_slice(values: &[f32]) -> f32 {
    effector_from_slice(values).distance(Vec2::new(values[4], values[5]))
}

pub fn min_clearance_from_slice(values: &[f32]) -> f32 {
    let effector = effector_from_slice(values);
    let obstacle_a = Obstacle {
        x: values[6],
        y: values[7],
        radius: values[8],
    };
    let obstacle_b = Obstacle {
        x: values[9],
        y: values[10],
        radius: values[11],
    };

    let clearance_a = effector.distance(obstacle_a.center()) - obstacle_a.radius;
    let clearance_b = effector.distance(obstacle_b.center()) - obstacle_b.radius;
    clearance_a.min(clearance_b)
}

pub fn apply_action(state: &RobotState, action: [f32; 2]) -> RobotState {
    RobotState {
        theta1: clamp(state.theta1 + action[0], -JOINT_LIMIT, JOINT_LIMIT),
        theta2: clamp(state.theta2 + action[1], -JOINT_LIMIT, JOINT_LIMIT),
        goal: state.goal,
        obstacles: state.obstacles.clone(),
    }
}

pub fn expert_action(state: &RobotState, rng: &mut StdRng) -> [f32; 2] {
    let pose = forward_kinematics(state.theta1, state.theta2);
    let effector = pose.effector;
    let to_goal = state.goal.sub(effector);

    let mut desired = to_goal
        .normalized()
        .scale((to_goal.length() * 0.22).clamp(0.02, 0.16));

    for obstacle in &state.obstacles {
        let away = effector.sub(obstacle.center());
        let clearance = away.length() - obstacle.radius;
        if clearance < 0.42 {
            let strength = (0.42 - clearance).powi(2) * 1.6;
            desired = desired.add(away.normalized().scale(strength));
        }
    }

    let wrist = state.theta1 + state.theta2;
    let j11 = -LINK_1 * state.theta1.sin() - LINK_2 * wrist.sin();
    let j12 = -LINK_2 * wrist.sin();
    let j21 = LINK_1 * state.theta1.cos() + LINK_2 * wrist.cos();
    let j22 = LINK_2 * wrist.cos();

    let bias_1 = -state.theta1 * 0.012;
    let bias_2 = -state.theta2 * 0.018;

    let mut delta_1 = j11 * desired.x + j21 * desired.y + bias_1;
    let mut delta_2 = j12 * desired.x + j22 * desired.y + bias_2;

    delta_1 += rng.gen_range(-0.012..0.012);
    delta_2 += rng.gen_range(-0.012..0.012);

    [
        clamp(delta_1 * 0.35, -MAX_DELTA, MAX_DELTA),
        clamp(delta_2 * 0.35, -MAX_DELTA, MAX_DELTA),
    ]
}

fn sample_episode(rng: &mut StdRng, episode_length: usize) -> Option<Episode> {
    let mission = sample_random_mission(rng);
    let mut state = mission_state(&mission);
    let mut states = Vec::with_capacity(episode_length);
    let mut actions = Vec::with_capacity(episode_length - 1);
    let mut observations = Vec::with_capacity(episode_length);

    let mut min_clearance = f32::INFINITY;

    for timestep in 0..episode_length {
        let snapshot = state_to_vec(&state);
        min_clearance = min_clearance.min(min_clearance_from_slice(&snapshot));
        observations.push(snapshot.clone());
        states.push(snapshot);

        if timestep == episode_length - 1 {
            break;
        }

        let action = expert_action(&state, rng);
        actions.push(action.to_vec());
        state = apply_action(&state, action);
    }

    let final_goal_distance = states
        .last()
        .map(|values| goal_distance_from_slice(values))
        .unwrap_or_default();

    let stats = EpisodeStats {
        min_clearance,
        final_goal_distance,
    };

    if is_episode_useful(&stats) {
        Some(Episode {
            states,
            actions,
            observations,
        })
    } else {
        None
    }
}

fn is_episode_useful(stats: &EpisodeStats) -> bool {
    stats.final_goal_distance < 0.16 && stats.min_clearance > 0.03
}

fn sample_random_mission(rng: &mut StdRng) -> MissionSpec {
    let start_angles = [rng.gen_range(-2.2..-0.6), rng.gen_range(0.7..2.1)];

    let goal_radius: f32 = rng.gen_range(0.7..1.18);
    let goal_angle: f32 = rng.gen_range(0.2..1.45);
    let goal = Vec2::new(
        goal_radius * goal_angle.cos(),
        goal_radius * goal_angle.sin(),
    );

    let obstacle_1 = sample_obstacle(rng, goal, 0.18..0.22, 0.2..0.85, 0.12..0.2);
    let obstacle_2 = sample_obstacle(rng, goal, 0.16..0.2, 0.3..0.95, 0.11..0.18);

    MissionSpec {
        id: "training".to_string(),
        title: "Training Mission".to_string(),
        eyebrow: "synthetic".to_string(),
        summary: String::new(),
        accent: "#000000".to_string(),
        difficulty: "train".to_string(),
        start_angles,
        goal,
        obstacles: vec![obstacle_1, obstacle_2],
        max_steps: 12,
    }
}

fn sample_obstacle(
    rng: &mut StdRng,
    goal: Vec2,
    x_range: core::ops::Range<f32>,
    y_range: core::ops::Range<f32>,
    radius_range: core::ops::Range<f32>,
) -> Obstacle {
    loop {
        let obstacle = Obstacle {
            x: rng.gen_range(x_range.clone()),
            y: rng.gen_range(y_range.clone()),
            radius: rng.gen_range(radius_range.clone()),
        };

        let clearance_from_goal = obstacle.center().distance(goal) - obstacle.radius;
        let clearance_from_base = obstacle.center().distance(BASE) - obstacle.radius;

        if clearance_from_goal > 0.22 && clearance_from_base > 0.25 {
            return obstacle;
        }
    }
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}
