//! Interactive TUI showcase for the demo pipeline.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::Backend as BurnBackend;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Gauge, List, ListItem, Paragraph, Row, Sparkline, Table, Wrap,
};

use gpc_core::config::TrainingConfig;
use gpc_core::noise::DdpmSchedule;
use gpc_core::tensor_utils;
use gpc_core::traits::{Policy, RewardFunction, WorldModel};
use gpc_policy::DiffusionPolicyConfig;
use gpc_train::data::{
    GpcDataset, GpcDatasetConfig, PolicyBatcher, PolicySampleData, WorldModelBatcher,
    WorldModelSequenceSample,
};
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

use super::demo::DemoArgs;

type TrainBackend = Autodiff<NdArray>;
type BackendDevice = <TrainBackend as BurnBackend>::Device;
type TerminalBackend = ratatui::backend::CrosstermBackend<Stdout>;
type ShowcaseTerminal = ratatui::Terminal<TerminalBackend>;

const UI_TICK: Duration = Duration::from_millis(90);
const STEP_DELAY: Duration = Duration::from_millis(16);
const AUTO_CLOSE_AFTER: Duration = Duration::from_secs(10);

const SURFACE: Color = Color::Rgb(15, 20, 28);
const PANEL: Color = Color::Rgb(24, 32, 44);
const INK: Color = Color::Rgb(229, 236, 244);
const MUTED: Color = Color::Rgb(122, 136, 153);
const ACCENT: Color = Color::Rgb(96, 202, 255);
const WORLD: Color = Color::Rgb(110, 220, 151);
const POLICY: Color = Color::Rgb(245, 185, 84);
const EVAL: Color = Color::Rgb(173, 139, 255);
const ERROR: Color = Color::Rgb(255, 107, 107);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineStage {
    Dataset,
    WorldModel,
    Policy,
    Evaluation,
}

impl PipelineStage {
    const ALL: [PipelineStage; 4] = [
        PipelineStage::Dataset,
        PipelineStage::WorldModel,
        PipelineStage::Policy,
        PipelineStage::Evaluation,
    ];

    fn label(self) -> &'static str {
        match self {
            PipelineStage::Dataset => "Synthetic dataset",
            PipelineStage::WorldModel => "World model warmup",
            PipelineStage::Policy => "Diffusion policy",
            PipelineStage::Evaluation => "GPC-RANK",
        }
    }

    fn color(self) -> Color {
        match self {
            PipelineStage::Dataset => ACCENT,
            PipelineStage::WorldModel => WORLD,
            PipelineStage::Policy => POLICY,
            PipelineStage::Evaluation => EVAL,
        }
    }

    fn index(self) -> usize {
        match self {
            PipelineStage::Dataset => 0,
            PipelineStage::WorldModel => 1,
            PipelineStage::Policy => 2,
            PipelineStage::Evaluation => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug)]
struct DatasetTelemetry {
    episodes: usize,
    transitions: usize,
    policy_windows: usize,
    rollout_windows: usize,
    avg_action_norm: f64,
    avg_delta_norm: f64,
}

#[derive(Clone, Debug)]
struct StageMetric {
    stage: PipelineStage,
    epoch: usize,
    total_epochs: usize,
    loss: f64,
    best_loss: f64,
    throughput: f64,
    secondary_label: &'static str,
    secondary_value: f64,
}

#[derive(Clone, Debug)]
struct CandidateTelemetry {
    rank: usize,
    candidate: usize,
    reward: f64,
    terminal_distance: f64,
    first_action: String,
}

#[derive(Clone, Debug)]
struct EvaluationTelemetry {
    goal_preview: String,
    best_reward: f64,
    mean_reward: f64,
    spread: f64,
    selected_action: String,
    rollout_distance: Vec<f64>,
    top_candidates: Vec<CandidateTelemetry>,
}

#[derive(Clone, Debug)]
struct FinalSummary {
    runtime: Duration,
    world_loss: Option<f64>,
    policy_loss: Option<f64>,
    best_reward: Option<f64>,
}

struct EvaluationInputs<'a> {
    num_candidates: usize,
    pred_horizon: usize,
    action_dim: usize,
    state_dim: usize,
    holdout: &'a WorldModelSequenceSample,
    eval_obs_tensor: &'a Tensor<TrainBackend, 3>,
    policy: &'a gpc_policy::DiffusionPolicy<TrainBackend>,
    world_model: &'a gpc_world::StateWorldModel<TrainBackend>,
}

#[derive(Clone, Debug)]
enum AppEvent {
    Log(String),
    DatasetReady(DatasetTelemetry),
    StageUpdate {
        stage: PipelineStage,
        status: StageStatus,
        progress: f64,
        detail: String,
    },
    Metric(StageMetric),
    EvaluationReady(EvaluationTelemetry),
    Completed(FinalSummary),
    Failed(String),
}

#[derive(Clone, Debug)]
struct StageCardState {
    stage: PipelineStage,
    status: StageStatus,
    progress: f64,
    detail: String,
}

#[derive(Debug)]
struct ShowcaseApp {
    args: DemoArgs,
    status: RunStatus,
    stages: Vec<StageCardState>,
    logs: VecDeque<String>,
    dataset: Option<DatasetTelemetry>,
    world_metrics: Vec<f64>,
    world_latest: Option<StageMetric>,
    policy_metrics: Vec<f64>,
    policy_latest: Option<StageMetric>,
    evaluation: Option<EvaluationTelemetry>,
    final_summary: Option<FinalSummary>,
    started_at: Instant,
    completed_at: Option<Instant>,
    spinner: usize,
    exit_requested: bool,
}

impl ShowcaseApp {
    fn new(args: DemoArgs) -> Self {
        let stages = PipelineStage::ALL
            .into_iter()
            .map(|stage| StageCardState {
                stage,
                status: StageStatus::Pending,
                progress: 0.0,
                detail: "waiting".to_string(),
            })
            .collect();

        Self {
            args,
            status: RunStatus::Running,
            stages,
            logs: VecDeque::with_capacity(12),
            dataset: None,
            world_metrics: Vec::new(),
            world_latest: None,
            policy_metrics: Vec::new(),
            policy_latest: None,
            evaluation: None,
            final_summary: None,
            started_at: Instant::now(),
            completed_at: None,
            spinner: 0,
            exit_requested: false,
        }
    }

    fn apply(&mut self, event: AppEvent) {
        match event {
            AppEvent::Log(message) => self.push_log(message),
            AppEvent::DatasetReady(dataset) => self.dataset = Some(dataset),
            AppEvent::StageUpdate {
                stage,
                status,
                progress,
                detail,
            } => {
                if let Some(card) = self.stages.get_mut(stage.index()) {
                    card.status = status;
                    card.progress = progress.clamp(0.0, 1.0);
                    card.detail = detail;
                }
            }
            AppEvent::Metric(metric) => match metric.stage {
                PipelineStage::WorldModel => {
                    self.world_metrics.push(metric.loss);
                    self.world_latest = Some(metric);
                }
                PipelineStage::Policy => {
                    self.policy_metrics.push(metric.loss);
                    self.policy_latest = Some(metric);
                }
                PipelineStage::Dataset | PipelineStage::Evaluation => {}
            },
            AppEvent::EvaluationReady(evaluation) => self.evaluation = Some(evaluation),
            AppEvent::Completed(summary) => {
                self.status = RunStatus::Completed;
                self.completed_at = Some(Instant::now());
                self.push_log(format!(
                    "summary: world {:.4} | policy {:.4} | reward {:.3}",
                    summary.world_loss.unwrap_or_default(),
                    summary.policy_loss.unwrap_or_default(),
                    summary.best_reward.unwrap_or_default()
                ));
                self.final_summary = Some(summary);
                self.push_log("pipeline complete".to_string());
            }
            AppEvent::Failed(error) => {
                self.status = RunStatus::Failed;
                self.completed_at = Some(Instant::now());
                if let Some(card) = self
                    .stages
                    .iter_mut()
                    .find(|card| card.status == StageStatus::Running)
                {
                    card.status = StageStatus::Failed;
                    card.detail = error.clone();
                }
                self.push_log(format!("error: {error}"));
            }
        }
    }

    fn on_tick(&mut self) {
        self.spinner = (self.spinner + 1) % 4;

        if let Some(completed_at) = self.completed_at {
            if completed_at.elapsed() >= AUTO_CLOSE_AFTER {
                self.exit_requested = true;
            }
        }
    }

    fn push_log(&mut self, message: String) {
        if self.logs.len() == 10 {
            self.logs.pop_front();
        }
        self.logs.push_back(message);
    }

    fn overall_progress(&self) -> f64 {
        let total: f64 = self.stages.iter().map(|stage| stage.progress).sum();
        total / self.stages.len() as f64
    }

    fn current_stage(&self) -> PipelineStage {
        self.stages
            .iter()
            .find(|stage| stage.status == StageStatus::Running)
            .map(|stage| stage.stage)
            .or_else(|| {
                self.stages
                    .iter()
                    .rev()
                    .find(|stage| stage.status == StageStatus::Completed)
                    .map(|stage| stage.stage)
            })
            .unwrap_or(PipelineStage::Dataset)
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    fn status_text(&self) -> String {
        match self.status {
            RunStatus::Running => {
                let spinner = ["|", "/", "-", "\\"][self.spinner];
                format!("{spinner} live on {}", self.current_stage().label())
            }
            RunStatus::Completed => "complete".to_string(),
            RunStatus::Failed => "failed".to_string(),
        }
    }

    fn countdown_text(&self) -> String {
        match self.completed_at {
            Some(instant) => {
                let remaining = AUTO_CLOSE_AFTER.saturating_sub(instant.elapsed());
                format!("q quit | auto close in {}s", remaining.as_secs())
            }
            None => "q quit".to_string(),
        }
    }
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

pub fn run_showcase(args: DemoArgs) -> Result<()> {
    let (mut terminal, _cleanup) = setup_terminal()?;
    let (tx, rx) = mpsc::channel();

    let worker_args = sanitize_args(args.clone());
    let worker_tx = tx.clone();
    thread::spawn(move || {
        if let Err(error) = run_pipeline(worker_args, &worker_tx) {
            let _ = worker_tx.send(AppEvent::Failed(error.to_string()));
        }
    });

    let mut app = ShowcaseApp::new(sanitize_args(args));

    loop {
        drain_events(&rx, &mut app);

        terminal.draw(|frame| render(frame, &app))?;

        if app.exit_requested {
            break;
        }

        if event::poll(UI_TICK)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            app.exit_requested = true;
                        }
                        KeyCode::Enter if app.status != RunStatus::Running => {
                            app.exit_requested = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        app.on_tick();
    }

    Ok(())
}

fn setup_terminal() -> Result<(ShowcaseTerminal, TerminalCleanup)> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let terminal = ratatui::Terminal::new(backend)?;
    Ok((terminal, TerminalCleanup))
}

fn drain_events(rx: &Receiver<AppEvent>, app: &mut ShowcaseApp) {
    while let Ok(event) = rx.try_recv() {
        app.apply(event);
    }
}

fn render(frame: &mut Frame<'_>, app: &ShowcaseApp) {
    frame.render_widget(
        Block::default().style(Style::default().bg(SURFACE)),
        frame.area(),
    );

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Min(20),
            Constraint::Length(8),
        ])
        .split(frame.area());

    render_header(frame, sections[0], app);
    render_metric_cards(frame, sections[1], app);
    render_body(frame, sections[2], app);
    render_logs(frame, sections[3], app);
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().bg(PANEL))
        .border_style(Style::default().fg(ACCENT))
        .title(" GPC SHOWCASE ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let header = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(38)])
        .split(header[0]);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "Generative Predictive Control",
            Style::default().fg(INK).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "  predict, rank, and act in one live pipeline",
            Style::default().fg(MUTED),
        ),
    ]));
    frame.render_widget(title, top[0]);

    let meta = Paragraph::new(format!(
        "runtime {}   seed {}   dims {}x{}   K {}",
        format_duration(app.elapsed()),
        app.args.seed,
        app.args.state_dim,
        app.args.action_dim,
        app.args.num_candidates
    ))
    .alignment(Alignment::Right)
    .style(Style::default().fg(MUTED));
    frame.render_widget(meta, top[1]);

    let label = format!(
        "{}  {:>3}%",
        app.status_text(),
        (app.overall_progress() * 100.0) as u16
    );
    let gauge = Gauge::default()
        .ratio(app.overall_progress())
        .label(label)
        .gauge_style(
            Style::default()
                .fg(app.current_stage().color())
                .bg(PANEL)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(gauge, header[1]);
}

fn render_metric_cards(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(28),
            Constraint::Percentage(24),
            Constraint::Percentage(24),
            Constraint::Percentage(24),
        ])
        .split(area);

    render_metric_card(
        frame,
        cards[0],
        "Current stage",
        app.current_stage().label(),
        &app.status_text(),
        app.current_stage().color(),
    );
    render_metric_card(
        frame,
        cards[1],
        "World loss",
        &format_optional(app.world_latest.as_ref().map(|metric| metric.loss), 4),
        &format_best(app.world_latest.as_ref().map(|metric| metric.best_loss)),
        WORLD,
    );
    render_metric_card(
        frame,
        cards[2],
        "Policy loss",
        &format_optional(app.policy_latest.as_ref().map(|metric| metric.loss), 4),
        &format_best(app.policy_latest.as_ref().map(|metric| metric.best_loss)),
        POLICY,
    );
    render_metric_card(
        frame,
        cards[3],
        "Best reward",
        &format_optional(app.evaluation.as_ref().map(|eval| eval.best_reward), 3),
        &app.final_summary
            .as_ref()
            .map(|summary| format!("runtime {}", format_duration(summary.runtime)))
            .unwrap_or_else(|| "waiting for ranking".to_string()),
        EVAL,
    );
}

fn render_metric_card(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    value: &str,
    subtitle: &str,
    color: Color,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .style(Style::default().bg(PANEL))
        .title(format!(" {title} "));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let content = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    frame.render_widget(
        Paragraph::new(value.to_string())
            .style(Style::default().fg(INK).add_modifier(Modifier::BOLD)),
        content[0],
    );
    frame.render_widget(
        Paragraph::new(subtitle.to_string()).style(Style::default().fg(MUTED)),
        content[1],
    );
}

fn render_body(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(area);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(9)])
        .split(columns[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(10)])
        .split(columns[1]);

    render_pipeline(frame, left[0], app);
    render_training(frame, left[1], app);
    render_dataset(frame, right[0], app);
    render_evaluation(frame, right[1], app);
}

fn render_pipeline(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(PANEL))
        .title(" Pipeline ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = app
        .stages
        .iter()
        .flat_map(|card| {
            let (badge, badge_color) = match card.status {
                StageStatus::Pending => ("WAIT", MUTED),
                StageStatus::Running => ("RUN ", card.stage.color()),
                StageStatus::Completed => ("DONE", WORLD),
                StageStatus::Failed => ("FAIL", ERROR),
            };
            let progress = ascii_bar(card.progress, 14);
            [
                Line::from(vec![
                    Span::styled(
                        format!("{badge} "),
                        Style::default()
                            .fg(badge_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("{:<18}", card.stage.label()),
                        Style::default().fg(INK),
                    ),
                    Span::styled(
                        format!(" {progress} {:>3}%", (card.progress * 100.0) as u16),
                        Style::default().fg(card.stage.color()),
                    ),
                ]),
                Line::from(Span::styled(
                    format!("    {}", card.detail),
                    Style::default().fg(MUTED),
                )),
            ]
        })
        .collect::<Vec<_>>();

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .style(Style::default().bg(PANEL)),
        inner,
    );
}

fn render_training(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(WORLD))
        .style(Style::default().bg(PANEL))
        .title(" Training telemetry ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);

    render_loss_panel(
        frame,
        panels[0],
        "WORLD MODEL",
        &app.world_metrics,
        app.world_latest.as_ref(),
        WORLD,
    );
    render_loss_panel(
        frame,
        panels[1],
        "DIFFUSION POLICY",
        &app.policy_metrics,
        app.policy_latest.as_ref(),
        POLICY,
    );
}

fn render_loss_panel(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    history: &[f64],
    latest: Option<&StageMetric>,
    color: Color,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .style(Style::default().bg(PANEL))
        .title(format!(" {title} "));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if history.is_empty() {
        frame.render_widget(
            Paragraph::new("collecting loss and throughput telemetry")
                .wrap(Wrap { trim: true })
                .style(Style::default().fg(MUTED)),
            inner,
        );
        return;
    }

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(inner);

    if let Some(metric) = latest {
        let info = vec![
            Line::from(vec![
                Span::styled(
                    format!("loss {:>8.4}", metric.loss),
                    Style::default().fg(INK).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("   best {:>8.4}", metric.best_loss),
                    Style::default().fg(color),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    format!(
                        "epoch {}/{}   {:.0} samp/s",
                        metric.epoch, metric.total_epochs, metric.throughput
                    ),
                    Style::default().fg(MUTED),
                ),
                Span::styled(
                    format!(
                        "   {} {:.4}",
                        metric.secondary_label, metric.secondary_value
                    ),
                    Style::default().fg(color),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(info), sections[0]);
    }

    let spark = sparkline_points(history, sections[1].width.saturating_sub(2) as usize);
    frame.render_widget(
        Sparkline::default()
            .style(Style::default().fg(color))
            .data(&spark),
        sections[1],
    );
}

fn render_dataset(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(PANEL))
        .title(" Dataset + config ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = match &app.dataset {
        Some(dataset) => vec![
            Line::from(vec![
                Span::styled("episodes ", Style::default().fg(MUTED)),
                Span::styled(dataset.episodes.to_string(), Style::default().fg(INK)),
                Span::styled("   transitions ", Style::default().fg(MUTED)),
                Span::styled(dataset.transitions.to_string(), Style::default().fg(INK)),
            ]),
            Line::from(vec![
                Span::styled("policy windows ", Style::default().fg(MUTED)),
                Span::styled(dataset.policy_windows.to_string(), Style::default().fg(INK)),
                Span::styled("   rollout windows ", Style::default().fg(MUTED)),
                Span::styled(
                    dataset.rollout_windows.to_string(),
                    Style::default().fg(INK),
                ),
            ]),
            Line::from(vec![
                Span::styled("dims ", Style::default().fg(MUTED)),
                Span::styled(
                    format!(
                        "state {}  action {}  horizon 4",
                        app.args.state_dim, app.args.action_dim
                    ),
                    Style::default().fg(INK),
                ),
            ]),
            Line::from(vec![
                Span::styled("episodes length ", Style::default().fg(MUTED)),
                Span::styled(
                    app.args.episode_length.to_string(),
                    Style::default().fg(INK),
                ),
                Span::styled("   epochs ", Style::default().fg(MUTED)),
                Span::styled(app.args.epochs.to_string(), Style::default().fg(INK)),
            ]),
            Line::from(vec![
                Span::styled("avg action norm ", Style::default().fg(MUTED)),
                Span::styled(
                    format!("{:.3}", dataset.avg_action_norm),
                    Style::default().fg(ACCENT),
                ),
            ]),
            Line::from(vec![
                Span::styled("avg delta norm ", Style::default().fg(MUTED)),
                Span::styled(
                    format!("{:.3}", dataset.avg_delta_norm),
                    Style::default().fg(WORLD),
                ),
            ]),
        ],
        None => vec![Line::from(Span::styled(
            "building the synthetic demonstration set",
            Style::default().fg(MUTED),
        ))],
    };

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .style(Style::default().bg(PANEL)),
        inner,
    );
}

fn render_evaluation(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(EVAL))
        .style(Style::default().bg(PANEL))
        .title(" Ranking + rollout ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(evaluation) = &app.evaluation else {
        frame.render_widget(
            Paragraph::new("waiting for policy sampling, world rollout, and reward scoring")
                .wrap(Wrap { trim: true })
                .style(Style::default().fg(MUTED)),
            inner,
        );
        return;
    };

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(6),
            Constraint::Length(3),
        ])
        .split(inner);

    let summary = vec![
        Line::from(vec![
            Span::styled("goal ", Style::default().fg(MUTED)),
            Span::styled(evaluation.goal_preview.clone(), Style::default().fg(INK)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("best {:>7.3}", evaluation.best_reward),
                Style::default().fg(EVAL).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "   mean {:>7.3}   spread {:>7.3}",
                    evaluation.mean_reward, evaluation.spread
                ),
                Style::default().fg(MUTED),
            ),
        ]),
        Line::from(vec![
            Span::styled("selected ", Style::default().fg(MUTED)),
            Span::styled(evaluation.selected_action.clone(), Style::default().fg(INK)),
        ]),
    ];
    frame.render_widget(Paragraph::new(summary), sections[0]);

    let rows = evaluation.top_candidates.iter().map(|candidate| {
        Row::new(vec![
            format!("#{}", candidate.rank),
            candidate.candidate.to_string(),
            format!("{:.3}", candidate.reward),
            format!("{:.3}", candidate.terminal_distance),
            candidate.first_action.clone(),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Min(12),
        ],
    )
    .header(
        Row::new(vec!["rank", "cand", "reward", "dist", "first action"])
            .style(Style::default().fg(MUTED).add_modifier(Modifier::BOLD)),
    )
    .column_spacing(1);
    frame.render_widget(table, sections[1]);

    let rollout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(sections[2]);
    frame.render_widget(
        Paragraph::new("best rollout distance to goal").style(Style::default().fg(MUTED)),
        rollout[0],
    );
    let spark = sparkline_points(
        &evaluation.rollout_distance,
        rollout[1].width.saturating_sub(2) as usize,
    );
    frame.render_widget(
        Sparkline::default()
            .style(Style::default().fg(EVAL))
            .data(&spark),
        rollout[1],
    );
}

fn render_logs(frame: &mut Frame<'_>, area: Rect, app: &ShowcaseApp) {
    let title = format!(" Logs | {} ", app.countdown_text());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(MUTED))
        .style(Style::default().bg(PANEL))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let items = if app.logs.is_empty() {
        vec![ListItem::new("booting showcase worker")]
    } else {
        app.logs
            .iter()
            .rev()
            .map(|line| ListItem::new(line.clone()))
            .collect()
    };
    frame.render_widget(List::new(items).style(Style::default().fg(INK)), inner);
}

fn run_pipeline(args: DemoArgs, tx: &Sender<AppEvent>) -> Result<()> {
    let started_at = Instant::now();
    let device = BackendDevice::default();

    let dataset_config = GpcDatasetConfig {
        data_dir: "demo".to_string(),
        state_dim: args.state_dim,
        action_dim: args.action_dim,
        obs_dim: args.state_dim,
        obs_horizon: 2,
        pred_horizon: 4,
    };
    let training_config = TrainingConfig {
        num_epochs: args.epochs,
        batch_size: 16,
        learning_rate: 1e-3,
        log_every: 1,
        ..Default::default()
    };

    send_event(tx, AppEvent::Log("booting synthetic pipeline".to_string()))?;
    send_stage(
        tx,
        PipelineStage::Dataset,
        StageStatus::Running,
        0.15,
        "synthesizing demonstrations".to_string(),
    )?;
    pace();

    let dataset = GpcDataset::generate_synthetic(
        dataset_config.clone(),
        args.episodes,
        args.episode_length,
        args.seed,
    );
    let world_samples = dataset.world_model_samples();
    let policy_samples = dataset.policy_samples();
    let rollout_sequences = dataset.world_model_sequences(dataset_config.pred_horizon);

    if world_samples.is_empty() || policy_samples.is_empty() || rollout_sequences.is_empty() {
        return Err(anyhow!(
            "demo configuration produced no training windows; increase episode length or episode count"
        ));
    }

    let dataset_metrics = DatasetTelemetry {
        episodes: dataset.num_episodes(),
        transitions: dataset.num_transitions(),
        policy_windows: policy_samples.len(),
        rollout_windows: rollout_sequences.len(),
        avg_action_norm: average_action_norm(&world_samples),
        avg_delta_norm: average_delta_norm(&world_samples),
    };
    send_event(tx, AppEvent::DatasetReady(dataset_metrics.clone()))?;
    send_stage(
        tx,
        PipelineStage::Dataset,
        StageStatus::Completed,
        1.0,
        format!(
            "{} episodes, {} transitions",
            dataset_metrics.episodes, dataset_metrics.transitions
        ),
    )?;
    send_event(
        tx,
        AppEvent::Log(format!(
            "[DATA] {} transitions | {} policy windows | {} rollout windows",
            dataset_metrics.transitions,
            dataset_metrics.policy_windows,
            dataset_metrics.rollout_windows
        )),
    )?;
    pace();

    let holdout = rollout_sequences
        .first()
        .cloned()
        .context("missing rollout sequence for preview")?;
    let eval_obs =
        build_observation_history(&holdout.0, dataset_config.obs_horizon, args.state_dim);
    let eval_obs_tensor = obs_history_tensor(&eval_obs, &device)?;

    let (world_model, world_best_loss) = train_world_model(
        tx,
        &device,
        &training_config,
        args.action_dim,
        args.state_dim,
        &world_samples,
        &holdout,
    )?;

    let (policy, policy_best_loss) = train_policy(
        tx,
        &device,
        &training_config,
        args.action_dim,
        args.state_dim,
        &policy_samples,
        &eval_obs_tensor,
    )?;

    let evaluation = evaluate_policy(
        tx,
        &device,
        EvaluationInputs {
            num_candidates: args.num_candidates,
            pred_horizon: dataset_config.pred_horizon,
            action_dim: args.action_dim,
            state_dim: args.state_dim,
            holdout: &holdout,
            eval_obs_tensor: &eval_obs_tensor,
            policy: &policy,
            world_model: &world_model,
        },
    )?;

    send_stage(
        tx,
        PipelineStage::Evaluation,
        StageStatus::Completed,
        1.0,
        "best rollout selected".to_string(),
    )?;

    let summary = FinalSummary {
        runtime: started_at.elapsed(),
        world_loss: Some(world_best_loss),
        policy_loss: Some(policy_best_loss),
        best_reward: Some(evaluation.best_reward),
    };
    send_event(tx, AppEvent::EvaluationReady(evaluation))?;
    send_event(tx, AppEvent::Completed(summary))?;
    Ok(())
}

fn train_world_model(
    tx: &Sender<AppEvent>,
    device: &BackendDevice,
    training_config: &TrainingConfig,
    action_dim: usize,
    state_dim: usize,
    world_samples: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    holdout: &WorldModelSequenceSample,
) -> Result<(gpc_world::StateWorldModel<TrainBackend>, f64)> {
    let config = StateWorldModelConfig {
        state_dim,
        action_dim,
        hidden_dim: 48,
        num_layers: 2,
    };
    let mut model = config.init::<TrainBackend>(device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(training_config.weight_decay as f32)
        .init();
    let batcher = WorldModelBatcher::<TrainBackend>::new(*device);
    let batches_per_epoch = world_samples.len().div_ceil(training_config.batch_size);
    let total_batches = training_config.num_epochs * batches_per_epoch;
    let mut global_batch = 0;
    let mut best_loss = f64::INFINITY;

    send_event(
        tx,
        AppEvent::Log("[WM ] calibrating one-step dynamics".to_string()),
    )?;
    send_stage(
        tx,
        PipelineStage::WorldModel,
        StageStatus::Running,
        0.0,
        "warming up residual dynamics".to_string(),
    )?;

    for epoch in 0..training_config.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0_f64;
        let mut epoch_items = 0usize;

        for (batch_idx, chunk) in world_samples.chunks(training_config.batch_size).enumerate() {
            let batch = batcher.batch(chunk.to_vec(), device);
            let predicted_delta = model.predict_delta(&batch.states, &batch.actions);
            let target_delta = batch.next_states.clone() - batch.states.clone();
            let loss = tensor_utils::mse_loss(&predicted_delta, &target_delta);
            let loss_value: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(training_config.learning_rate, model, grads);

            global_batch += 1;
            epoch_loss += loss_value as f64;
            epoch_items += chunk.len();

            send_stage(
                tx,
                PipelineStage::WorldModel,
                StageStatus::Running,
                global_batch as f64 / total_batches as f64,
                format!(
                    "epoch {}/{}   batch {}/{}",
                    epoch + 1,
                    training_config.num_epochs,
                    batch_idx + 1,
                    batches_per_epoch
                ),
            )?;
            pace();
        }

        let avg_loss = epoch_loss / batches_per_epoch as f64;
        best_loss = best_loss.min(avg_loss);
        let throughput = epoch_items as f64 / epoch_start.elapsed().as_secs_f64().max(1e-6);
        let preview_error = rollout_preview_error(&model, holdout, device)?;

        send_event(
            tx,
            AppEvent::Metric(StageMetric {
                stage: PipelineStage::WorldModel,
                epoch: epoch + 1,
                total_epochs: training_config.num_epochs,
                loss: avg_loss,
                best_loss,
                throughput,
                secondary_label: "rollout mse",
                secondary_value: preview_error,
            }),
        )?;
        send_event(
            tx,
            AppEvent::Log(format!(
                "[WM ] epoch {}/{} loss {:.4} rollout {:.4}",
                epoch + 1,
                training_config.num_epochs,
                avg_loss,
                preview_error
            )),
        )?;
    }

    send_stage(
        tx,
        PipelineStage::WorldModel,
        StageStatus::Completed,
        1.0,
        format!("best loss {:.4}", best_loss),
    )?;

    Ok((model, best_loss))
}

fn train_policy(
    tx: &Sender<AppEvent>,
    device: &BackendDevice,
    training_config: &TrainingConfig,
    action_dim: usize,
    state_dim: usize,
    policy_samples: &[PolicySampleData],
    eval_obs_tensor: &Tensor<TrainBackend, 3>,
) -> Result<(gpc_policy::DiffusionPolicy<TrainBackend>, f64)> {
    let config = DiffusionPolicyConfig {
        obs_dim: state_dim,
        action_dim,
        obs_horizon: 2,
        pred_horizon: 4,
        hidden_dim: 48,
        time_embed_dim: 32,
        num_res_blocks: 2,
    };
    let mut model = config.init::<TrainBackend>(device);
    let schedule = DdpmSchedule::new(&gpc_core::config::NoiseScheduleConfig::default());
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(training_config.weight_decay as f32)
        .init();
    let batcher = PolicyBatcher::<TrainBackend>::new(*device, state_dim, action_dim, 2, 4);
    let batches_per_epoch = policy_samples.len().div_ceil(training_config.batch_size);
    let total_batches = training_config.num_epochs * batches_per_epoch;
    let mut global_batch = 0;
    let mut best_loss = f64::INFINITY;

    send_event(
        tx,
        AppEvent::Log("[POL] denoising action sequences".to_string()),
    )?;
    send_stage(
        tx,
        PipelineStage::Policy,
        StageStatus::Running,
        0.0,
        "learning reverse diffusion".to_string(),
    )?;

    for epoch in 0..training_config.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0_f64;
        let mut epoch_items = 0usize;

        for (batch_idx, chunk) in policy_samples
            .chunks(training_config.batch_size)
            .enumerate()
        {
            let batch = batcher.batch(chunk.to_vec(), device);
            let [batch_size, pred_horizon, _] = batch.actions.dims();

            let actions_flat = tensor_utils::flatten_last_two(batch.actions.clone());
            let obs_flat = tensor_utils::flatten_last_two(batch.observations.clone());
            let timesteps_vec: Vec<f32> = (0..batch_size)
                .map(|_| {
                    (rand::random::<f32>() * schedule.num_timesteps as f32)
                        .floor()
                        .min((schedule.num_timesteps - 1) as f32)
                })
                .collect();
            let timesteps =
                Tensor::<TrainBackend, 1>::from_floats(timesteps_vec.as_slice(), device);
            let noise = Tensor::<TrainBackend, 2>::random(
                [batch_size, pred_horizon * action_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            );

            let t_idx = timesteps_vec[0] as usize;
            let noisy_actions = schedule.add_noise(&actions_flat, &noise, t_idx);
            let noise_pred = model.predict_noise(noisy_actions, obs_flat, timesteps);
            let loss = tensor_utils::mse_loss(&noise_pred, &noise);
            let loss_value: f32 = loss.clone().into_scalar().elem();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(training_config.learning_rate, model, grads);

            global_batch += 1;
            epoch_loss += loss_value as f64;
            epoch_items += chunk.len();

            send_stage(
                tx,
                PipelineStage::Policy,
                StageStatus::Running,
                global_batch as f64 / total_batches as f64,
                format!(
                    "epoch {}/{}   batch {}/{}",
                    epoch + 1,
                    training_config.num_epochs,
                    batch_idx + 1,
                    batches_per_epoch
                ),
            )?;
            pace();
        }

        let avg_loss = epoch_loss / batches_per_epoch as f64;
        best_loss = best_loss.min(avg_loss);
        let throughput = epoch_items as f64 / epoch_start.elapsed().as_secs_f64().max(1e-6);
        let sample_energy = policy_sample_energy(&model, eval_obs_tensor, device)?;

        send_event(
            tx,
            AppEvent::Metric(StageMetric {
                stage: PipelineStage::Policy,
                epoch: epoch + 1,
                total_epochs: training_config.num_epochs,
                loss: avg_loss,
                best_loss,
                throughput,
                secondary_label: "sample norm",
                secondary_value: sample_energy,
            }),
        )?;
        send_event(
            tx,
            AppEvent::Log(format!(
                "[POL] epoch {}/{} loss {:.4} sample {:.4}",
                epoch + 1,
                training_config.num_epochs,
                avg_loss,
                sample_energy
            )),
        )?;
    }

    send_stage(
        tx,
        PipelineStage::Policy,
        StageStatus::Completed,
        1.0,
        format!("best loss {:.4}", best_loss),
    )?;

    Ok((model, best_loss))
}

fn evaluate_policy(
    tx: &Sender<AppEvent>,
    device: &BackendDevice,
    inputs: EvaluationInputs<'_>,
) -> Result<EvaluationTelemetry> {
    let current_state = vector_tensor(&inputs.holdout.0, device).reshape([1, inputs.state_dim]);
    let goal = inputs
        .holdout
        .2
        .last()
        .cloned()
        .context("missing goal state from holdout sequence")?;
    let goal_tensor = vector_tensor(&goal, device);
    let reward_fn = L2RewardFunctionConfig {
        state_dim: inputs.state_dim,
    }
    .init::<TrainBackend>(device)
    .with_goal(goal_tensor.clone());

    send_event(
        tx,
        AppEvent::Log("[EVAL] sampling candidate trajectories".to_string()),
    )?;
    send_stage(
        tx,
        PipelineStage::Evaluation,
        StageStatus::Running,
        0.20,
        "sampling from diffusion policy".to_string(),
    )?;
    let candidates =
        inputs
            .policy
            .sample_k(inputs.eval_obs_tensor, inputs.num_candidates, device)?;
    pace();

    send_stage(
        tx,
        PipelineStage::Evaluation,
        StageStatus::Running,
        0.55,
        "rolling candidates through world model".to_string(),
    )?;
    let states_repeated =
        gpc_core::tensor_utils::repeat_batch_2d(&current_state, inputs.num_candidates);
    let predicted_states = inputs.world_model.rollout(&states_repeated, &candidates)?;
    pace();

    send_stage(
        tx,
        PipelineStage::Evaluation,
        StageStatus::Running,
        0.82,
        "scoring trajectories against the goal".to_string(),
    )?;
    let rewards = reward_fn.compute_reward(&predicted_states)?;
    let rewards_vec = tensor_to_vec(rewards, "reward")?;

    let candidate_values = tensor_to_vec(
        candidates
            .clone()
            .reshape([inputs.num_candidates * inputs.pred_horizon * inputs.action_dim]),
        "candidate actions",
    )?;
    let predicted_values = tensor_to_vec(
        predicted_states
            .clone()
            .reshape([inputs.num_candidates * inputs.pred_horizon * inputs.state_dim]),
        "predicted states",
    )?;

    let mut ordering: Vec<usize> = (0..rewards_vec.len()).collect();
    ordering.sort_by(|&left, &right| {
        rewards_vec[right]
            .partial_cmp(&rewards_vec[left])
            .unwrap_or(Ordering::Equal)
    });

    let best_idx = *ordering
        .first()
        .context("no candidate rewards available for evaluation")?;
    let mean_reward = rewards_vec.iter().sum::<f32>() as f64 / rewards_vec.len() as f64;
    let min_reward = rewards_vec.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let best_reward = rewards_vec[best_idx] as f64;

    let top_candidates = ordering
        .iter()
        .take(5)
        .enumerate()
        .map(|(rank, candidate)| {
            let action_offset = candidate * inputs.pred_horizon * inputs.action_dim;
            let first_action = &candidate_values[action_offset..action_offset + inputs.action_dim];
            CandidateTelemetry {
                rank: rank + 1,
                candidate: *candidate,
                reward: rewards_vec[*candidate] as f64,
                terminal_distance: -(rewards_vec[*candidate] as f64),
                first_action: preview_vector(first_action, 4),
            }
        })
        .collect::<Vec<_>>();

    let selected_action_offset = best_idx * inputs.pred_horizon * inputs.action_dim;
    let selected_action = preview_vector(
        &candidate_values[selected_action_offset..selected_action_offset + inputs.action_dim],
        4,
    );
    let rollout_distance = rollout_distance_series(
        &predicted_values,
        best_idx,
        inputs.pred_horizon,
        inputs.state_dim,
        &goal,
    );

    send_event(
        tx,
        AppEvent::Log(format!(
            "[EVAL] best reward {:.3} | goal {}",
            best_reward,
            preview_vector(&goal, 4)
        )),
    )?;

    Ok(EvaluationTelemetry {
        goal_preview: preview_vector(&goal, 4),
        best_reward,
        mean_reward,
        spread: best_reward - min_reward,
        selected_action,
        rollout_distance,
        top_candidates,
    })
}

fn rollout_preview_error(
    model: &gpc_world::StateWorldModel<TrainBackend>,
    holdout: &WorldModelSequenceSample,
    device: &BackendDevice,
) -> Result<f64> {
    let initial_state = vector_tensor(&holdout.0, device).reshape([1, holdout.0.len()]);
    let action_values = holdout
        .1
        .iter()
        .flat_map(|action| action.iter().copied())
        .collect::<Vec<_>>();
    let target_values = holdout
        .2
        .iter()
        .flat_map(|state| state.iter().copied())
        .collect::<Vec<_>>();
    let horizon = holdout.1.len();
    let action_dim = holdout.1[0].len();
    let state_dim = holdout.2[0].len();

    let actions = Tensor::<TrainBackend, 1>::from_floats(action_values.as_slice(), device)
        .reshape([1, horizon, action_dim]);
    let targets = Tensor::<TrainBackend, 1>::from_floats(target_values.as_slice(), device)
        .reshape([1, horizon, state_dim]);
    let predicted = model.rollout(&initial_state, &actions)?;
    let mse: f32 = tensor_utils::mse_loss(&predicted, &targets)
        .into_scalar()
        .elem();
    Ok(mse as f64)
}

fn policy_sample_energy(
    policy: &gpc_policy::DiffusionPolicy<TrainBackend>,
    eval_obs_tensor: &Tensor<TrainBackend, 3>,
    device: &BackendDevice,
) -> Result<f64> {
    let sample = policy.sample(eval_obs_tensor, device)?;
    let [_, pred_horizon, action_dim] = sample.dims();
    let values = tensor_to_vec(sample.reshape([pred_horizon * action_dim]), "policy sample")?;
    Ok(l2_norm(&values))
}

fn average_action_norm(samples: &[(Vec<f32>, Vec<f32>, Vec<f32>)]) -> f64 {
    let total = samples
        .iter()
        .map(|(_, action, _)| l2_norm(action))
        .sum::<f64>();
    total / samples.len() as f64
}

fn average_delta_norm(samples: &[(Vec<f32>, Vec<f32>, Vec<f32>)]) -> f64 {
    let total = samples
        .iter()
        .map(|(state, _, next_state)| {
            let delta = state
                .iter()
                .zip(next_state.iter())
                .map(|(left, right)| right - left)
                .collect::<Vec<_>>();
            l2_norm(&delta)
        })
        .sum::<f64>();
    total / samples.len() as f64
}

fn build_observation_history(
    initial_state: &[f32],
    obs_horizon: usize,
    obs_dim: usize,
) -> Vec<Vec<f32>> {
    let observation = initial_state[..obs_dim.min(initial_state.len())].to_vec();
    (0..obs_horizon).map(|_| observation.clone()).collect()
}

fn obs_history_tensor(
    history: &[Vec<f32>],
    device: &BackendDevice,
) -> Result<Tensor<TrainBackend, 3>> {
    let obs_horizon = history.len();
    let obs_dim = history[0].len();
    let flat = history
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Ok(
        Tensor::<TrainBackend, 1>::from_floats(flat.as_slice(), device).reshape([
            1,
            obs_horizon,
            obs_dim,
        ]),
    )
}

fn vector_tensor(values: &[f32], device: &BackendDevice) -> Tensor<TrainBackend, 1> {
    Tensor::<TrainBackend, 1>::from_floats(values, device)
}

fn rollout_distance_series(
    predicted_values: &[f32],
    candidate_idx: usize,
    pred_horizon: usize,
    state_dim: usize,
    goal: &[f32],
) -> Vec<f64> {
    let offset = candidate_idx * pred_horizon * state_dim;
    (0..pred_horizon)
        .map(|step| {
            let step_offset = offset + step * state_dim;
            let state = &predicted_values[step_offset..step_offset + state_dim];
            let squared = state
                .iter()
                .zip(goal.iter())
                .map(|(left, right)| {
                    let diff = left - right;
                    diff * diff
                })
                .sum::<f32>();
            squared.sqrt() as f64
        })
        .collect()
}

fn tensor_to_vec(tensor: Tensor<TrainBackend, 1>, label: &str) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .to_vec()
        .map_err(|error| anyhow!("failed to extract {label} tensor data: {error:?}"))
}

fn preview_vector(values: &[f32], limit: usize) -> String {
    let mut parts = values
        .iter()
        .take(limit)
        .map(|value| format!("{value:.2}"))
        .collect::<Vec<_>>();
    if values.len() > limit {
        parts.push("...".to_string());
    }
    format!("[{}]", parts.join(", "))
}

fn sparkline_points(values: &[f64], target_len: usize) -> Vec<u64> {
    if values.is_empty() {
        return vec![0];
    }

    let condensed = if target_len == 0 || values.len() <= target_len {
        values.to_vec()
    } else {
        let bucket = values.len().div_ceil(target_len);
        values
            .chunks(bucket)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect()
    };

    let min = condensed.iter().copied().fold(f64::INFINITY, f64::min);
    let max = condensed.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        return vec![50; condensed.len()];
    }

    condensed
        .into_iter()
        .map(|value| (((value - min) / (max - min)) * 100.0).round() as u64 + 1)
        .collect()
}

fn ascii_bar(progress: f64, width: usize) -> String {
    let filled = (progress.clamp(0.0, 1.0) * width as f64).round() as usize;
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    for idx in 0..width {
        bar.push(if idx < filled { '#' } else { '-' });
    }
    bar.push(']');
    bar
}

fn format_optional(value: Option<f64>, precision: usize) -> String {
    value
        .map(|value| format!("{value:.precision$}"))
        .unwrap_or_else(|| "--".to_string())
}

fn format_best(value: Option<f64>) -> String {
    value
        .map(|value| format!("best {:.4}", value))
        .unwrap_or_else(|| "collecting metrics".to_string())
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    format!(
        "{:02}:{:02}.{:01}",
        total_seconds / 60,
        total_seconds % 60,
        duration.subsec_millis() / 100
    )
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt()
}

fn send_event(tx: &Sender<AppEvent>, event: AppEvent) -> Result<()> {
    tx.send(event)
        .map_err(|_| anyhow!("showcase event receiver disconnected"))
}

fn send_stage(
    tx: &Sender<AppEvent>,
    stage: PipelineStage,
    status: StageStatus,
    progress: f64,
    detail: String,
) -> Result<()> {
    send_event(
        tx,
        AppEvent::StageUpdate {
            stage,
            status,
            progress,
            detail,
        },
    )
}

fn pace() {
    thread::sleep(STEP_DELAY);
}

fn sanitize_args(mut args: DemoArgs) -> DemoArgs {
    args.epochs = args.epochs.max(1);
    args.episodes = args.episodes.max(4);
    args.episode_length = args.episode_length.max(8);
    args.num_candidates = args.num_candidates.max(4);
    args.state_dim = args.state_dim.max(2);
    args.action_dim = args.action_dim.max(1);
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_args() -> DemoArgs {
        DemoArgs {
            epochs: 3,
            episodes: 8,
            episode_length: 20,
            num_candidates: 6,
            state_dim: 4,
            action_dim: 2,
            seed: 42,
            plain: true,
        }
    }

    #[test]
    fn sparkline_points_handles_constant_series() {
        let spark = sparkline_points(&[0.5, 0.5, 0.5], 6);
        assert_eq!(spark, vec![50, 50, 50]);
    }

    #[test]
    fn app_tracks_metric_updates() {
        let mut app = ShowcaseApp::new(demo_args());
        app.apply(AppEvent::Metric(StageMetric {
            stage: PipelineStage::WorldModel,
            epoch: 1,
            total_epochs: 3,
            loss: 0.25,
            best_loss: 0.25,
            throughput: 100.0,
            secondary_label: "rollout mse",
            secondary_value: 0.12,
        }));

        assert_eq!(app.world_metrics, vec![0.25]);
        assert_eq!(
            app.world_latest.as_ref().map(|metric| metric.epoch),
            Some(1)
        );
    }

    #[test]
    fn sanitize_args_guards_demo_inputs() {
        let args = DemoArgs {
            epochs: 0,
            episodes: 1,
            episode_length: 3,
            num_candidates: 0,
            state_dim: 1,
            action_dim: 0,
            seed: 7,
            plain: false,
        };
        let sanitized = sanitize_args(args);

        assert_eq!(sanitized.epochs, 1);
        assert_eq!(sanitized.episodes, 4);
        assert_eq!(sanitized.episode_length, 8);
        assert_eq!(sanitized.num_candidates, 4);
        assert_eq!(sanitized.state_dim, 2);
        assert_eq!(sanitized.action_dim, 1);
    }
}
