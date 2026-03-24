import {
  startTransition,
  type CSSProperties,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useMemo,
  useState,
} from 'react'

import { fetchSimulation, fetchSnapshot, healthCheck } from './api'
import { RobotStage } from './components/RobotStage'
import { SignalChart } from './components/SignalChart'
import type {
  MissionPlayback,
  MissionSpec,
  PlannerMode,
  PlanningFrame,
  RuntimeSnapshot,
} from './types'

function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [playback, setPlayback] = useState<MissionPlayback | null>(null)
  const [statusMessage, setStatusMessage] = useState('Connecting to planner…')
  const [error, setError] = useState<string | null>(null)
  const [missionId, setMissionId] = useState('')
  const [mode, setMode] = useState<PlannerMode>('rank')
  const [candidateCount, setCandidateCount] = useState(18)
  const [frameIndex, setFrameIndex] = useState(0)
  const [autoplay, setAutoplay] = useState(true)
  const [simulating, setSimulating] = useState(false)

  const deferredCandidates = useDeferredValue(candidateCount)

  useEffect(() => {
    let cancelled = false

    async function boot() {
      while (!cancelled) {
        if (await healthCheck()) break
        await new Promise((resolve) => setTimeout(resolve, 800))
      }
      if (cancelled) return

      setStatusMessage('Fetching engine snapshot…')

      try {
        const snap = await fetchSnapshot()
        if (cancelled) return
        setSnapshot(snap)
        setCandidateCount(snap.overview.recommended_candidates)
        setMissionId((id) => id || snap.missions[0]?.id || '')
        setStatusMessage('')
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch snapshot')
        }
      }
    }

    boot()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!snapshot || !missionId) return

    let cancelled = false
    setSimulating(true)

    fetchSimulation(missionId, mode, deferredCandidates)
      .then((result) => {
        if (cancelled) return
        startTransition(() => {
          setPlayback(result)
          setFrameIndex(0)
          setStatusMessage('')
        })
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Simulation failed')
        }
      })
      .finally(() => {
        if (!cancelled) setSimulating(false)
      })

    return () => { cancelled = true }
  }, [snapshot, missionId, mode, deferredCandidates])

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
                onClick={() => setMode('rank')}
                type="button"
              >
                GPC-RANK
              </button>
              <button
                className={mode === 'opt' ? 'mode-toggle__button is-active' : 'mode-toggle__button'}
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
              <button onClick={() => setAutoplay((v) => !v)} type="button">
                {autoplay ? 'Pause' : 'Play'}
              </button>
            </div>
            <input
              className="timeline-slider"
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
            <p className="eyebrow">architecture</p>
            <h2>The browser sees the same closed-loop the paper describes.</h2>
            <p>
              A diffusion policy proposes joint-space futures. A learned world model
              rolls each one forward. The evaluator scores safety and progress, then
              the best prefix executes before the whole cycle repeats.
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
                <dt>obs history</dt>
                <dd>{snapshot.overview.obs_horizon}</dd>
              </div>
              <div>
                <dt>episodes</dt>
                <dd>{snapshot.overview.dataset_episodes}</dd>
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
