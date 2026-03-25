import {
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
import { DEFAULT_RUNTIME_BUILD_CONFIG } from './types'

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

const SETTINGS_GRID_STYLE: CSSProperties = {
  display: 'grid',
  gap: '0.6rem',
  gridTemplateColumns: 'repeat(auto-fit, minmax(10rem, 1fr))',
}

const SETTINGS_FIELD_STYLE: CSSProperties = {
  display: 'grid',
  gap: '0.28rem',
}

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
  if (!Number.isFinite(value)) {
    return minimum
  }

  return Math.max(minimum, Math.trunc(value))
}

function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [playback, setPlayback] = useState<MissionPlayback | null>(null)
  const [statusMessage, setStatusMessage] = useState('Connecting to planner…')
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

  useEffect(() => {
    simulatingRef.current = simulating
  }, [simulating])

  useEffect(() => {
    rebuildingRef.current = rebuilding
  }, [rebuilding])

  useEffect(() => {
    let cancelled = false
    let activeClient: PlannerClient | null = null

    async function boot() {
      try {
        setError(null)
        setStatusMessage('Connecting to planner…')

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

  useEffect(() => {
    if (!snapshot) {
      return
    }

    setRuntimeDraft(snapshot.overview.build_config)
  }, [snapshot])

  useEffect(() => {
    if (!snapshot || !missionId || rebuildingRef.current || simulatingRef.current) return
    if (candidateCount !== deferredCandidates) return

    let cancelled = false
    const client = plannerRef.current
    if (!client) {
      return
    }

    async function runSimulation(currentClient: PlannerClient) {
      setSimulating(true)
      setError(null)
      setStatusMessage(`Planning with ${formatPlannerSource(currentClient.source)}…`)

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
          setStatusMessage(`Planning failed with ${formatPlannerSource(currentClient.source)}.`)
        }
      } finally {
        if (!cancelled) setSimulating(false)
      }
    }

    void runSimulation(client)

    return () => { cancelled = true }
  }, [snapshot, missionId, mode, candidateCount, deferredCandidates])

  const applyRuntimeRebuild = async () => {
    const client = plannerRef.current
    if (!client || rebuilding || simulating) {
      return
    }

    const nextConfig = normalizeRuntimeBuildConfig(runtimeDraft)

    setRebuilding(true)
    setError(null)
    setStatusMessage(`Rebuilding with ${formatPlannerSource(client.source)}…`)

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
    setRuntimeDraft((current) => ({
      ...current,
      [field]: value,
    }))
  }

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

  const currentMission = useMemo<MissionSpec | null>(() => {
    if (!snapshot) return null
    return snapshot.missions.find((m) => m.id === missionId) ?? snapshot.missions[0]
  }, [missionId, snapshot])

  const frames = playback?.frames ?? []
  const currentFrame: PlanningFrame | null =
    frames[Math.min(frameIndex, Math.max(frames.length - 1, 0))] ?? null
  const stageReady = Boolean(currentMission && currentFrame && playback)

  return (
    <main className="app-shell">
      {/* ── top bar ──────────────────────────────── */}
      <header className="top-bar">
        <div className="top-bar__brand">
          <h1>GPC</h1>
          <span>Generative Policy Control</span>
          <strong className="top-bar__source mono">{formatPlannerSource(runtimeSource)}</strong>
        </div>
        {snapshot ? (
          <div className="top-bar__stats">
            <div className="top-bar__stat">
              <span className="label">bootstrap</span>
              <span className="mono">{snapshot.overview.bootstrap_ms}ms</span>
            </div>
            <div className="top-bar__stat">
              <span className="label">transitions</span>
              <span className="mono">{snapshot.overview.dataset_transitions}</span>
            </div>
            <div className="top-bar__stat">
              <span className="label">horizon</span>
              <span className="mono">{snapshot.overview.pred_horizon}</span>
            </div>
          </div>
        ) : null}
      </header>

      {/* ── main demo grid ───────────────────────── */}
      <div className="demo-layout">
        <div className="demo-layout__stage">
          {stageReady && currentMission && currentFrame ? (
            <RobotStage frame={currentFrame} mission={currentMission} mode={mode} />
          ) : (
            <section className="robot-stage--loading">
              <div className="robot-stage__loading">
                <span className="robot-stage__pulse" />
                <p>{statusMessage || 'Preparing simulation…'}</p>
              </div>
            </section>
          )}
        </div>

        <aside className="control-rail">
          {/* runtime rebuild */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Runtime</p>
              <strong>{rebuilding ? 'Rebuilding…' : 'Training settings'}</strong>
            </div>
            <div style={SETTINGS_GRID_STYLE}>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  dataset seed
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('dataset_seed', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.dataset_seed}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  dataset episodes
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('dataset_episodes', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.dataset_episodes}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  episode length
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('episode_length', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.episode_length}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  phase 1 epochs
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('world_phase1_epochs', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.world_phase1_epochs}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  phase 2 epochs
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('world_phase2_epochs', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.world_phase2_epochs}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  policy epochs
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('policy_epochs', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.policy_epochs}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  batch size
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) => updateRuntimeField('batch_size', Number(event.target.value))}
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.batch_size}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  recommended candidates
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('recommended_candidates', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.recommended_candidates}
                />
              </label>
              <label style={SETTINGS_FIELD_STYLE}>
                <span className="mono" style={{ color: 'var(--t3)', fontSize: '0.7rem' }}>
                  opt steps
                </span>
                <input
                  disabled={simulating || rebuilding}
                  inputMode="numeric"
                  min={1}
                  onChange={(event) =>
                    updateRuntimeField('recommended_opt_steps', Number(event.target.value))
                  }
                  style={NUMBER_INPUT_STYLE}
                  type="number"
                  value={runtimeDraft.recommended_opt_steps}
                />
              </label>
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '0.75rem',
                marginTop: '0.8rem',
              }}
            >
              <span
                style={{
                  color: 'var(--t3)',
                  fontSize: '0.72rem',
                  lineHeight: 1.4,
                }}
              >
                Rebuilds both the browser/WASM runtime and the REST fallback with the same config.
              </span>
              <button
                disabled={!snapshot || simulating || rebuilding}
                onClick={() => {
                  void applyRuntimeRebuild()
                }}
                type="button"
              >
                {rebuilding ? 'Rebuilding…' : 'Apply rebuild'}
              </button>
            </div>
          </section>

          {/* missions */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Mission</p>
              <strong>{currentMission?.title ?? '…'}</strong>
            </div>
            <div className="mission-list">
              {snapshot?.missions.map((mission) => (
                <button
                  key={mission.id}
                  className={mission.id === missionId ? 'mission-card is-active' : 'mission-card'}
                  disabled={simulating || rebuilding}
                  onClick={() => setMissionId(mission.id)}
                  style={{ '--mission-color': mission.accent } as CSSProperties}
                  type="button"
                >
                  <p>{mission.eyebrow}</p>
                  <strong>{mission.title}</strong>
                  <span>{mission.summary}</span>
                </button>
              ))}
            </div>
          </section>

          {/* planner mode */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Planner</p>
              <strong>{mode === 'rank' ? 'Rank' : 'Optimize'}</strong>
            </div>
            <div className="mode-toggle" role="tablist" aria-label="Planner mode">
              <button
                className={mode === 'rank' ? 'mode-toggle__button is-active' : 'mode-toggle__button'}
                disabled={simulating || rebuilding}
                onClick={() => setMode('rank')}
                type="button"
              >
                GPC-RANK
              </button>
              <button
                className={mode === 'opt' ? 'mode-toggle__button is-active' : 'mode-toggle__button'}
                disabled={simulating || rebuilding}
                onClick={() => setMode('opt')}
                type="button"
              >
                GPC-OPT
              </button>
            </div>
            <label className="slider-field">
              <span>candidates</span>
              <strong>{deferredCandidates}</strong>
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
          </section>

          {/* playback */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Playback</p>
              <strong className="mono">
                {currentFrame ? frameIndex + 1 : 0}/{frames.length}
              </strong>
            </div>
            <div className="timeline-actions">
              <button disabled={simulating || rebuilding} onClick={() => setAutoplay((v) => !v)} type="button">
                {autoplay ? 'Pause' : 'Play'}
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
              <div className="stat-inline">
                <div>
                  <span>spread</span>
                  <strong className="mono">{currentFrame.reward_spread.toFixed(3)}</strong>
                </div>
                <div>
                  <span>best</span>
                  <strong className="mono">{currentFrame.reward_best.toFixed(3)}</strong>
                </div>
              </div>
            ) : null}
          </section>

          {/* top candidates */}
          <section className="control-card">
            <div className="control-card__header">
              <p>Top trajectories</p>
              <strong>{currentFrame?.candidates.length ?? 0}</strong>
            </div>
            <div className="ranking-list">
              {currentFrame?.candidates.map((c) => (
                <article className="ranking-row" key={c.rank}>
                  <div>
                    <p>#{c.rank}</p>
                    <strong className="mono">{c.reward.toFixed(3)}</strong>
                  </div>
                  <div>
                    <span>{c.clearance.toFixed(3)}</span>
                    <span>{c.terminal_distance.toFixed(3)}</span>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </aside>
      </div>

      {/* ── evidence section ─────────────────────── */}
      {snapshot ? (
        <section className="evidence-section">
          <div className="evidence-copy">
            <p className="eyebrow">closed-loop architecture</p>
            <h2>Plan far, execute short, replan from reality.</h2>
            <p>
              The diffusion policy samples diverse candidate trajectories. The world model
              imagines each one forward. The evaluator scores progress and safety. Only the
              first action of the winning trajectory executes — then the entire cycle repeats
              from the actual state, not the predicted one.
            </p>
            <dl className="evidence-stats">
              <div>
                <dt>state dims</dt>
                <dd>{snapshot.overview.state_dim}</dd>
              </div>
              <div>
                <dt>action dims</dt>
                <dd>{snapshot.overview.action_dim}</dd>
              </div>
              <div>
                <dt>obs horizon</dt>
                <dd>{snapshot.overview.obs_horizon}</dd>
              </div>
              <div>
                <dt>pred horizon</dt>
                <dd>{snapshot.overview.pred_horizon}</dd>
              </div>
              <div>
                <dt>episodes</dt>
                <dd>{snapshot.overview.dataset_episodes}</dd>
              </div>
              <div>
                <dt>transitions</dt>
                <dd>{snapshot.overview.dataset_transitions}</dd>
              </div>
            </dl>
          </div>

          <div className="evidence-signals">
            <SignalChart
              accent="world"
              points={snapshot.overview.world_loss_curve}
              title="world-model loss"
            />
            <SignalChart
              accent="policy"
              points={snapshot.overview.policy_loss_curve}
              title="diffusion-policy loss"
            />
          </div>
        </section>
      ) : null}

      {/* ── summary strip ────────────────────────── */}
      {playback ? (
        <section className="summary-strip">
          <article>
            <p>outcome</p>
            <strong>{playback.summary.success ? 'goal reached' : 'replan limit'}</strong>
          </article>
          <article>
            <p>final distance</p>
            <strong className="mono">{playback.summary.final_goal_distance.toFixed(3)}</strong>
          </article>
          <article>
            <p>min clearance</p>
            <strong className="mono">{playback.summary.min_clearance.toFixed(3)}</strong>
          </article>
          <article>
            <p>forecast error</p>
            <strong className="mono">{playback.summary.average_world_error.toFixed(3)}</strong>
          </article>
        </section>
      ) : null}

      {simulating ? <p className="simulating-hint">simulating…</p> : null}
      {error ? <section className="error-banner">{error}</section> : null}
    </main>
  )
}

export default App
