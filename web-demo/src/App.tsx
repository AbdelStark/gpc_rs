import {
  Fragment,
  startTransition,
  type CSSProperties,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
} from 'react'

import { RobotStage } from './components/RobotStage'
import { SignalChart } from './components/SignalChart'
import { createPlannerSession, formatPlannerSource, type PlannerClient } from './plannerClient'
import type {
  RuntimeBuildConfig,
  MissionPlayback,
  MissionSpec,
  PlannerMode,
  PlanningFrame,
  RuntimeSnapshot,
} from './types'
import { DEFAULT_RUNTIME_BUILD_CONFIG, PLANNER_MODES } from './types'

/* ─── strategy metadata ─────────────────────────────────────────── */

const STRATEGY_INFO: Record<
  PlannerMode,
  { label: string; tag: string; description: string; accent: string }
> = {
  policy: {
    label: 'GPC-POLICY',
    tag: 'Baseline',
    description: 'Raw diffusion sample executed without world-model evaluation.',
    accent: 'var(--orange)',
  },
  rank: {
    label: 'GPC-RANK',
    tag: 'Sample & Score',
    description: 'World model scores K candidates and selects the best trajectory.',
    accent: 'var(--lime)',
  },
  opt: {
    label: 'GPC-OPT',
    tag: 'Refine',
    description: 'Gradient refinement sharpens the winning trajectory after ranking.',
    accent: 'var(--blue)',
  },
}

/* ─── pipeline stages ───────────────────────────────────────────── */

const PIPELINE_STAGES = [
  { id: 'observe', label: 'Observe', detail: 'Current state', color: undefined },
  { id: 'propose', label: 'Propose', detail: 'Diffusion policy', color: '--orange' },
  { id: 'imagine', label: 'Imagine', detail: 'World model', color: '--lime' },
  { id: 'evaluate', label: 'Evaluate', detail: 'Score & select', color: '--blue' },
  { id: 'execute', label: 'Execute', detail: 'First action', color: undefined },
] as const

/* ─── settings field definitions ────────────────────────────────── */

const SETTINGS_FIELDS: {
  group: string
  fields: { key: keyof RuntimeBuildConfig; label: string; min: number }[]
}[] = [
  {
    group: 'Dataset',
    fields: [
      { key: 'dataset_seed', label: 'seed', min: 0 },
      { key: 'dataset_episodes', label: 'episodes', min: 1 },
      { key: 'episode_length', label: 'ep. length', min: 1 },
    ],
  },
  {
    group: 'World Model',
    fields: [
      { key: 'world_phase1_epochs', label: 'phase 1 epochs', min: 1 },
      { key: 'world_phase2_epochs', label: 'phase 2 epochs', min: 1 },
    ],
  },
  {
    group: 'Policy',
    fields: [
      { key: 'policy_epochs', label: 'epochs', min: 1 },
      { key: 'batch_size', label: 'batch size', min: 1 },
    ],
  },
  {
    group: 'Inference',
    fields: [
      { key: 'recommended_candidates', label: 'candidates', min: 1 },
      { key: 'recommended_opt_steps', label: 'opt steps', min: 1 },
    ],
  },
]

const NUMBER_INPUT_STYLE: CSSProperties = {
  border: '1px solid var(--line)',
  borderRadius: 'var(--radius-sm)',
  background: 'var(--s1)',
  color: 'var(--t1)',
  padding: '0.45rem 0.55rem',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: '0.84rem',
  fontVariantNumeric: 'tabular-nums',
  width: '100%',
  minWidth: 0,
}

/* ─── helpers ───────────────────────────────────────────────────── */

function normalizeRuntimeBuildConfig(config: RuntimeBuildConfig): RuntimeBuildConfig {
  return {
    dataset_seed: clampInteger(config.dataset_seed, 0),
    dataset_episodes: clampInteger(config.dataset_episodes, 1),
    episode_length: clampInteger(config.episode_length, 1),
    world_phase1_epochs: clampInteger(config.world_phase1_epochs, 1),
    world_phase2_epochs: clampInteger(config.world_phase2_epochs, 1),
    policy_epochs: clampInteger(config.policy_epochs, 1),
    batch_size: clampInteger(config.batch_size, 1),
    recommended_candidates: clampInteger(config.recommended_candidates, 1),
    recommended_opt_steps: clampInteger(config.recommended_opt_steps, 1),
  }
}

function clampInteger(value: number, minimum: number) {
  if (!Number.isFinite(value)) return minimum
  return Math.max(minimum, Math.trunc(value))
}

function hasRankingTelemetry(mode: PlannerMode) {
  return mode !== 'policy'
}

function getFrameNarration(frame: PlanningFrame, mode: PlannerMode): string {
  const step = frame.step + 1

  if (mode === 'policy') {
    if (frame.goal_distance < 0.08)
      return `Step ${step} — Approaching goal. Distance: ${frame.goal_distance.toFixed(3)}.`
    return `Step ${step} — Policy sampled a single trajectory. Executing first action toward goal.`
  }

  if (mode === 'rank') {
    return `Step ${step} — Sampled ${frame.candidates.length} candidates. World model scored each one. Best reward: ${frame.reward_best.toFixed(3)}.`
  }

  return `Step ${step} — Ranked ${frame.candidates.length} candidates, then refined the winner via gradients. Forecast error: ${frame.world_model_error.toFixed(3)}.`
}

/* ─── App ───────────────────────────────────────────────────────── */

function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [playback, setPlayback] = useState<MissionPlayback | null>(null)
  const [statusMessage, setStatusMessage] = useState('Connecting to planner\u2026')
  const [error, setError] = useState<string | null>(null)
  const [runtimeSource, setRuntimeSource] = useState<PlannerClient['source'] | null>(null)
  const [runtimeDraft, setRuntimeDraft] = useState<RuntimeBuildConfig>(DEFAULT_RUNTIME_BUILD_CONFIG)
  const [missionId, setMissionId] = useState('')
  const [mode, setMode] = useState<PlannerMode>('rank')
  const [candidateCount, setCandidateCount] = useState(18)
  const [frameIndex, setFrameIndex] = useState(0)
  const [autoplay, setAutoplay] = useState(true)
  const [simulating, setSimulating] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const plannerRef = useRef<PlannerClient | null>(null)
  const simulatingRef = useRef(simulating)
  const rebuildingRef = useRef(rebuilding)

  const deferredCandidates = useDeferredValue(candidateCount)

  /* keep refs in sync */
  useEffect(() => {
    simulatingRef.current = simulating
  }, [simulating])

  useEffect(() => {
    rebuildingRef.current = rebuilding
  }, [rebuilding])

  /* bootstrap planner */
  useEffect(() => {
    let cancelled = false
    let activeClient: PlannerClient | null = null

    async function boot() {
      try {
        setError(null)
        setStatusMessage('Connecting to planner\u2026')

        const session = await createPlannerSession((status) => {
          if (cancelled) return
          setRuntimeSource(status.source)
          setStatusMessage(status.message)
        })

        if (cancelled) {
          session.client.close()
          return
        }
        activeClient = session.client
        plannerRef.current = session.client
        setRuntimeSource(session.client.source)
        setSnapshot(session.snapshot)
        setCandidateCount(session.snapshot.overview.recommended_candidates)
        setRuntimeDraft(session.snapshot.overview.build_config)
        setMissionId((id) => id || session.snapshot.missions[0]?.id || '')
        setStatusMessage(`${formatPlannerSource(session.client.source)} ready.`)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to initialize planner')
        }
      }
    }

    boot()
    return () => {
      cancelled = true
      activeClient?.close()
      if (plannerRef.current === activeClient) {
        plannerRef.current = null
      }
    }
  }, [])

  /* sync draft with snapshot */
  useEffect(() => {
    if (!snapshot) return
    setRuntimeDraft(snapshot.overview.build_config)
  }, [snapshot])

  /* run simulation on param changes */
  useEffect(() => {
    if (!snapshot || !missionId || rebuildingRef.current || simulatingRef.current) return
    if (candidateCount !== deferredCandidates) return

    let cancelled = false
    const client = plannerRef.current
    if (!client) return

    async function runSimulation(currentClient: PlannerClient) {
      setSimulating(true)
      setError(null)
      setStatusMessage(
        `Planning ${STRATEGY_INFO[mode].label.toLowerCase()} with ${formatPlannerSource(currentClient.source)}\u2026`,
      )

      try {
        const result = await currentClient.simulate(missionId, mode, deferredCandidates)
        if (cancelled) return

        startTransition(() => {
          setPlayback(result)
          setFrameIndex(0)
          setStatusMessage(`${formatPlannerSource(currentClient.source)} planner ready.`)
        })
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Simulation failed')
          setStatusMessage(
            `Planning failed with ${formatPlannerSource(currentClient.source)}.`,
          )
        }
      } finally {
        if (!cancelled) setSimulating(false)
      }
    }

    void runSimulation(client)
    return () => {
      cancelled = true
    }
  }, [snapshot, missionId, mode, candidateCount, deferredCandidates])

  /* runtime rebuild */
  const applyRuntimeRebuild = async () => {
    const client = plannerRef.current
    if (!client || rebuilding || simulating) return

    const nextConfig = normalizeRuntimeBuildConfig(runtimeDraft)
    setRebuilding(true)
    setError(null)
    setStatusMessage(`Rebuilding with ${formatPlannerSource(client.source)}\u2026`)

    try {
      const nextSnapshot = await client.rebuild(nextConfig)
      setSnapshot(nextSnapshot)
      setPlayback(null)
      setFrameIndex(0)
      setCandidateCount(nextSnapshot.overview.recommended_candidates)
      setRuntimeDraft(nextSnapshot.overview.build_config)
      setMissionId((currentMissionId) =>
        nextSnapshot.missions.some((mission) => mission.id === currentMissionId)
          ? currentMissionId
          : nextSnapshot.missions[0]?.id || '',
      )
      setStatusMessage(`${formatPlannerSource(client.source)} runtime rebuilt.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rebuild runtime')
      setStatusMessage(`Rebuild failed with ${formatPlannerSource(client.source)}.`)
    } finally {
      setRebuilding(false)
    }
  }

  const updateRuntimeField = <K extends keyof RuntimeBuildConfig>(field: K, value: number) => {
    setRuntimeDraft((current) => ({ ...current, [field]: value }))
  }

  /* autoplay timer */
  const tickFrame = useEffectEvent(() => {
    setFrameIndex((current) => {
      const total = playback?.frames.length ?? 0
      return total === 0 ? current : (current + 1) % total
    })
  })

  useEffect(() => {
    if (!autoplay || !playback?.frames.length) return
    const timer = window.setInterval(() => tickFrame(), 1400)
    return () => window.clearInterval(timer)
  }, [autoplay, playback?.frames.length])

  /* keyboard shortcuts */
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.target !== document.body) return
      const total = playback?.frames.length ?? 0
      if (total === 0) return

      if (e.key === ' ') {
        e.preventDefault()
        setAutoplay((v) => !v)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        setAutoplay(false)
        setFrameIndex((i) => Math.min(i + 1, total - 1))
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setAutoplay(false)
        setFrameIndex((i) => Math.max(i - 1, 0))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [playback?.frames.length])

  /* derived state */
  const currentMission = useMemo<MissionSpec | null>(() => {
    if (!snapshot) return null
    return snapshot.missions.find((m) => m.id === missionId) ?? snapshot.missions[0]
  }, [missionId, snapshot])

  const frames = playback?.frames ?? []
  const currentFrame: PlanningFrame | null =
    frames[Math.min(frameIndex, Math.max(frames.length - 1, 0))] ?? null
  const displayedMode = playback ? playback.summary.mode : mode
  const rankingTelemetryVisible = hasRankingTelemetry(displayedMode)
  const stageReady = Boolean(currentMission && currentFrame && playback)

  /* ── render ───────────────────────────────────────────────────── */

  return (
    <main className="app-shell">
      {/* ── header ─────────────────────────────────── */}
      <header className="top-bar">
        <div className="top-bar__brand">
          <h1>GPC</h1>
          <span className="top-bar__tagline">Generative Policy Control</span>
          {runtimeSource && (
            <span className="top-bar__source mono">{formatPlannerSource(runtimeSource)}</span>
          )}
        </div>
        <nav className="top-bar__nav">
          <a
            className="top-bar__link"
            href="https://arxiv.org/pdf/2502.00622"
            rel="noopener noreferrer"
            target="_blank"
          >
            Paper
          </a>
          <a
            className="top-bar__link"
            href="https://github.com/AbdelStark/gpc_rs"
            rel="noopener noreferrer"
            target="_blank"
          >
            GitHub
          </a>
        </nav>
      </header>

      {/* ── hero ───────────────────────────────────── */}
      <section className="hero">
        <div className="hero__content">
          <h2 className="hero__headline">
            Plan far, execute short,
            <br />
            replan from reality.
          </h2>
          <p className="hero__description">
            Watch a robotic arm plan using diffusion-sampled trajectories, evaluate them with a
            learned world model, and execute only the winning action before replanning from actual
            observations. Switch strategies below to compare raw sampling, world-model ranking, and
            gradient refinement.
          </p>
        </div>

        {/* pipeline diagram */}
        <div className="pipeline">
          <div className="pipeline__flow">
            {PIPELINE_STAGES.map((stage, i) => (
              <Fragment key={stage.id}>
                {i > 0 && (
                  <div className="pipeline__arrow" aria-hidden="true">
                    <svg viewBox="0 0 16 10" className="pipeline__arrow-svg">
                      <path d="M 0 5 L 12 5 M 9 1.5 L 13 5 L 9 8.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                )}
                <div
                  className="pipeline__node"
                  style={
                    stage.color
                      ? ({ '--node-accent': `var(${stage.color})` } as CSSProperties)
                      : undefined
                  }
                >
                  <span className="pipeline__label">{stage.label}</span>
                  <span className="pipeline__detail">{stage.detail}</span>
                </div>
              </Fragment>
            ))}
          </div>
          <div className="pipeline__loop">
            <span className="pipeline__loop-line" aria-hidden="true" />
            <span className="pipeline__loop-label">closed-loop replanning</span>
            <span className="pipeline__loop-line" aria-hidden="true" />
          </div>
        </div>
      </section>

      {/* ── demo grid ──────────────────────────────── */}
      <div className="demo-layout">
        <div className="demo-layout__stage">
          {stageReady && currentMission && currentFrame ? (
            <>
              <RobotStage frame={currentFrame} mission={currentMission} mode={displayedMode} />
              <div className="narration">
                <p>{getFrameNarration(currentFrame, displayedMode)}</p>
                <span className="narration__hint mono">
                  space to play/pause &middot; arrow keys to step
                </span>
              </div>
            </>
          ) : (
            <section className="robot-stage--loading">
              <div className="robot-stage__loading">
                <span className="robot-stage__pulse" />
                <p>{statusMessage || 'Preparing simulation\u2026'}</p>
                <span className="robot-stage__loading-note">
                  The demo trains and runs entirely in your browser via WebAssembly.
                </span>
                {error && <p className="robot-stage__error">{error}</p>}
              </div>
            </section>
          )}
        </div>

        <aside className="control-rail">
          {/* ── 1. strategy selector ── */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Strategy</p>
              <strong>{STRATEGY_INFO[mode].label}</strong>
            </div>
            <div className="strategy-grid">
              {PLANNER_MODES.map((plannerMode) => {
                const info = STRATEGY_INFO[plannerMode]
                const isActive = mode === plannerMode
                return (
                  <button
                    key={plannerMode}
                    className={`strategy-card${isActive ? ' is-active' : ''}`}
                    disabled={simulating || rebuilding}
                    onClick={() => setMode(plannerMode)}
                    style={{ '--strategy-accent': info.accent } as CSSProperties}
                    type="button"
                  >
                    <div className="strategy-card__top">
                      <strong>{info.label}</strong>
                      <span className="strategy-card__tag">{info.tag}</span>
                    </div>
                    <p>{info.description}</p>
                  </button>
                )
              })}
            </div>
            {rankingTelemetryVisible ? (
              <label className="slider-field">
                <span>Candidates to sample</span>
                <strong className="mono">{deferredCandidates}</strong>
                <input
                  disabled={simulating || rebuilding}
                  max={28}
                  min={8}
                  onChange={(e) => setCandidateCount(Number(e.target.value))}
                  step={1}
                  type="range"
                  value={candidateCount}
                />
              </label>
            ) : null}
          </section>

          {/* ── 2. mission selector ── */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Mission</p>
              <strong>{currentMission?.title ?? '\u2026'}</strong>
            </div>
            <div className="mission-list">
              {snapshot?.missions.map((mission) => (
                <button
                  key={mission.id}
                  className={`mission-card${mission.id === missionId ? ' is-active' : ''}`}
                  disabled={simulating || rebuilding}
                  onClick={() => setMissionId(mission.id)}
                  style={{ '--mission-color': mission.accent } as CSSProperties}
                  type="button"
                >
                  <div className="mission-card__top">
                    <strong>{mission.title}</strong>
                    <span
                      className={`mission-card__difficulty mission-card__difficulty--${mission.difficulty}`}
                    >
                      {mission.difficulty}
                    </span>
                  </div>
                  <span>{mission.summary}</span>
                </button>
              ))}
            </div>
          </section>

          {/* ── 3. playback + telemetry ── */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Playback</p>
              <strong className="mono">
                {currentFrame ? frameIndex + 1 : 0} / {frames.length}
              </strong>
            </div>
            <div className="playback-controls">
              <button
                className="playback-btn"
                disabled={simulating || rebuilding || !frames.length}
                onClick={() => {
                  if (frameIndex >= frames.length - 1) {
                    setFrameIndex(0)
                    setAutoplay(true)
                  } else {
                    setAutoplay((v) => !v)
                  }
                }}
                type="button"
              >
                {!frames.length
                  ? 'No data'
                  : autoplay
                    ? 'Pause'
                    : frameIndex >= frames.length - 1
                      ? 'Replay'
                      : 'Play'}
              </button>
              <button
                className="playback-btn playback-btn--ghost"
                disabled={simulating || rebuilding || !frames.length}
                onClick={() => {
                  setAutoplay(false)
                  setFrameIndex((i) => Math.min(i + 1, frames.length - 1))
                }}
                type="button"
              >
                Step
              </button>
            </div>
            <input
              className="timeline-slider"
              disabled={simulating || rebuilding}
              max={Math.max(frames.length - 1, 0)}
              min={0}
              onChange={(e) => {
                setAutoplay(false)
                setFrameIndex(Number(e.target.value))
              }}
              type="range"
              value={Math.min(frameIndex, Math.max(frames.length - 1, 0))}
            />
            {currentFrame ? (
              <div className="telemetry-grid">
                <div className="telemetry-item">
                  <span>Goal dist.</span>
                  <strong className="mono">{currentFrame.goal_distance.toFixed(3)}</strong>
                </div>
                <div className="telemetry-item">
                  <span>Clearance</span>
                  <strong className="mono">{currentFrame.min_clearance.toFixed(3)}</strong>
                </div>
                {rankingTelemetryVisible ? (
                  <>
                    <div className="telemetry-item">
                      <span>Best reward</span>
                      <strong className="mono">{currentFrame.reward_best.toFixed(3)}</strong>
                    </div>
                    <div className="telemetry-item">
                      <span>Spread</span>
                      <strong className="mono">{currentFrame.reward_spread.toFixed(3)}</strong>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
          </section>

          {/* ── 4. top trajectories ── */}
          {rankingTelemetryVisible && currentFrame && currentFrame.candidates.length > 0 ? (
            <section className="control-card">
              <div className="control-card__header">
                <p>Top trajectories</p>
                <strong>{currentFrame.candidates.length}</strong>
              </div>
              <div className="ranking-list">
                {currentFrame.candidates.map((c) => (
                  <article className="ranking-row" key={c.rank}>
                    <div className="ranking-row__left">
                      <span className="ranking-row__rank">#{c.rank}</span>
                      <strong className="mono">{c.reward.toFixed(3)}</strong>
                    </div>
                    <div className="ranking-row__right">
                      <span>clearance {c.clearance.toFixed(2)}</span>
                      <span>dist {c.terminal_distance.toFixed(2)}</span>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          {/* ── 5. advanced settings (collapsible) ── */}
          <details className="control-card advanced-settings">
            <summary className="advanced-settings__toggle">
              <span className="advanced-settings__toggle-text">
                <p>Training Configuration</p>
                <span>Retrain both models with custom hyperparameters</span>
              </span>
              <span className="advanced-settings__chevron" aria-hidden="true" />
            </summary>
            <div className="advanced-settings__body">
              {SETTINGS_FIELDS.map((group) => (
                <div key={group.group} className="settings-group">
                  <span className="settings-group__label">{group.group}</span>
                  <div className="settings-group__fields">
                    {group.fields.map((field) => (
                      <label key={field.key} className="settings-field">
                        <span className="mono">{field.label}</span>
                        <input
                          disabled={simulating || rebuilding}
                          inputMode="numeric"
                          min={field.min}
                          onChange={(e) => updateRuntimeField(field.key, Number(e.target.value))}
                          style={NUMBER_INPUT_STYLE}
                          type="number"
                          value={runtimeDraft[field.key]}
                        />
                      </label>
                    ))}
                  </div>
                </div>
              ))}
              <div className="advanced-settings__actions">
                <button
                  disabled={!snapshot || simulating || rebuilding}
                  onClick={() => void applyRuntimeRebuild()}
                  type="button"
                >
                  {rebuilding ? 'Rebuilding\u2026' : 'Rebuild Runtime'}
                </button>
                <span className="advanced-settings__note">
                  Retrains world model and policy from scratch in-browser.
                </span>
              </div>
            </div>
          </details>
        </aside>
      </div>

      {/* ── evidence section ───────────────────────── */}
      {snapshot ? (
        <section className="evidence-section">
          <div className="evidence-signals">
            <SignalChart
              accent="world"
              points={snapshot.overview.world_loss_curve}
              title="World-model loss"
            />
            <SignalChart
              accent="policy"
              points={snapshot.overview.policy_loss_curve}
              title="Diffusion-policy loss"
            />
          </div>
          <div className="evidence-stats-grid">
            {[
              { label: 'State dims', value: snapshot.overview.state_dim },
              { label: 'Action dims', value: snapshot.overview.action_dim },
              { label: 'Obs horizon', value: snapshot.overview.obs_horizon },
              { label: 'Pred horizon', value: snapshot.overview.pred_horizon },
              { label: 'Episodes', value: snapshot.overview.dataset_episodes },
              { label: 'Transitions', value: snapshot.overview.dataset_transitions },
              { label: 'Bootstrap', value: `${snapshot.overview.bootstrap_ms}ms` },
            ].map((stat) => (
              <div key={stat.label} className="evidence-stat">
                <span>{stat.label}</span>
                <strong className="mono">{stat.value}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {/* ── summary strip ──────────────────────────── */}
      {playback ? (
        <section className="summary-strip">
          <article
            className={`summary-item${playback.summary.success ? ' summary-item--success' : ' summary-item--limit'}`}
          >
            <p>Outcome</p>
            <strong>{playback.summary.success ? 'Goal reached' : 'Replan limit'}</strong>
          </article>
          <article className="summary-item">
            <p>Strategy</p>
            <strong>{STRATEGY_INFO[displayedMode].label}</strong>
          </article>
          <article className="summary-item">
            <p>Final distance</p>
            <strong className="mono">{playback.summary.final_goal_distance.toFixed(3)}</strong>
          </article>
          <article className="summary-item">
            <p>Steps executed</p>
            <strong className="mono">{playback.summary.executed_steps}</strong>
          </article>
          {rankingTelemetryVisible ? (
            <>
              <article className="summary-item">
                <p>Min clearance</p>
                <strong className="mono">{playback.summary.min_clearance.toFixed(3)}</strong>
              </article>
              <article className="summary-item">
                <p>Forecast error</p>
                <strong className="mono">
                  {playback.summary.average_world_error.toFixed(3)}
                </strong>
              </article>
            </>
          ) : null}
        </section>
      ) : null}

      {/* ── status indicators ──────────────────────── */}
      {simulating ? <p className="simulating-hint">Simulating\u2026</p> : null}
      {error && stageReady ? <section className="error-banner">{error}</section> : null}
    </main>
  )
}

export default App
