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

function formatCompact(value: number) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value)
}

function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [playback, setPlayback] = useState<MissionPlayback | null>(null)
  const [statusMessage, setStatusMessage] = useState('Waiting for the Rust planner server…')
  const [error, setError] = useState<string | null>(null)
  const [missionId, setMissionId] = useState('')
  const [mode, setMode] = useState<PlannerMode>('rank')
  const [candidateCount, setCandidateCount] = useState(18)
  const [frameIndex, setFrameIndex] = useState(0)
  const [autoplay, setAutoplay] = useState(true)
  const [simulating, setSimulating] = useState(false)

  const deferredCandidates = useDeferredValue(candidateCount)

  // Poll for server readiness, then fetch the snapshot
  useEffect(() => {
    let cancelled = false

    async function boot() {
      setStatusMessage('Connecting to the Rust planner server…')

      // Poll until the server is up
      while (!cancelled) {
        if (await healthCheck()) break
        await new Promise((resolve) => setTimeout(resolve, 800))
      }
      if (cancelled) return

      setStatusMessage('Server online. Fetching engine snapshot…')

      try {
        const snap = await fetchSnapshot()
        if (cancelled) return
        setSnapshot(snap)
        setCandidateCount(snap.overview.recommended_candidates)
        setMissionId((currentId) => currentId || snap.missions[0]?.id || '')
        setStatusMessage('Planner warm. Rolling out a cinematic mission trace.')
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch snapshot')
          setStatusMessage('Could not connect to the planner server.')
        }
      }
    }

    boot()
    return () => {
      cancelled = true
    }
  }, [])

  // Simulate whenever mission/mode/candidates change
  useEffect(() => {
    if (!snapshot || !missionId) return

    let cancelled = false
    setSimulating(true)
    setStatusMessage('Sampling candidates and ranking them in the world model…')

    fetchSimulation(missionId, mode, deferredCandidates)
      .then((result) => {
        if (cancelled) return
        startTransition(() => {
          setPlayback(result)
          setFrameIndex(0)
          setStatusMessage('Mission trace ready. Scrub or autoplay the closed-loop plan.')
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

    return () => {
      cancelled = true
    }
  }, [snapshot, missionId, mode, deferredCandidates])

  const tickFrame = useEffectEvent(() => {
    setFrameIndex((current) => {
      const total = playback?.frames.length ?? 0
      if (total === 0) {
        return current
      }

      return (current + 1) % total
    })
  })

  useEffect(() => {
    if (!autoplay || !playback?.frames.length) {
      return
    }

    const timer = window.setInterval(() => tickFrame(), 1450)
    return () => window.clearInterval(timer)
  }, [autoplay, playback?.frames.length])

  const currentMission = useMemo<MissionSpec | null>(() => {
    if (!snapshot) {
      return null
    }

    return snapshot.missions.find((mission) => mission.id === missionId) ?? snapshot.missions[0]
  }, [missionId, snapshot])

  const frames = playback?.frames ?? []
  const currentFrame: PlanningFrame | null =
    frames[Math.min(frameIndex, Math.max(frames.length - 1, 0))] ?? null

  const stageReady = Boolean(currentMission && currentFrame && playback)

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-panel__copy">
          <p className="eyebrow">robotics demo</p>
          <h1>Rust world models steering a robot arm from a local planner server.</h1>
          <p className="lede">
            This demo runs the project&apos;s policy, world-model, training, and evaluation crates
            natively on your machine, then streams plans to the browser like a compact physical-AI
            control room.
          </p>
          <div className="hero-panel__tags">
            <span>gpc-policy</span>
            <span>gpc-world</span>
            <span>gpc-eval</span>
            <span>gpc-train</span>
            <span>native server</span>
          </div>
        </div>

        <div className="hero-panel__proof">
          <p>{statusMessage}</p>
          {simulating ? <p className="simulating-hint">simulating…</p> : null}
          {snapshot ? (
            <dl>
              <div>
                <dt>dataset windows</dt>
                <dd>{formatCompact(snapshot.overview.dataset_transitions)}</dd>
              </div>
              <div>
                <dt>bootstrap</dt>
                <dd>{snapshot.overview.bootstrap_ms} ms</dd>
              </div>
              <div>
                <dt>planner horizon</dt>
                <dd>{snapshot.overview.pred_horizon} steps</dd>
              </div>
            </dl>
          ) : null}
        </div>
      </section>

      <section className="demo-grid">
        {stageReady && currentMission && currentFrame ? (
          <RobotStage frame={currentFrame} mission={currentMission} mode={mode} />
        ) : (
          <section className="robot-stage robot-stage--loading">
            <div className="robot-stage__loading">
              <span className="robot-stage__pulse" />
              <p>{statusMessage}</p>
            </div>
          </section>
        )}

        <aside className="control-rail">
          <section className="control-card control-card--selector">
            <div className="control-card__header">
              <p>Mission presets</p>
              <strong>{currentMission?.title ?? 'Loading'}</strong>
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

          <section className="control-card">
            <div className="control-card__header">
              <p>Planner mode</p>
              <strong>{mode === 'rank' ? 'World-model rank' : 'GPC-OPT refine'}</strong>
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
              <span>candidate rollouts</span>
              <strong>{deferredCandidates}</strong>
              <input
                max={28}
                min={8}
                onChange={(event) => setCandidateCount(Number(event.target.value))}
                step={1}
                type="range"
                value={candidateCount}
              />
            </label>
          </section>

          <section className="control-card control-card--timeline">
            <div className="control-card__header">
              <p>Playback</p>
              <strong>
                frame {currentFrame ? frameIndex + 1 : 0}/{playback?.frames.length ?? 0}
              </strong>
            </div>
            <div className="timeline-actions">
              <button onClick={() => setAutoplay((value) => !value)} type="button">
                {autoplay ? 'Pause autoplay' : 'Resume autoplay'}
              </button>
            </div>
            <input
              className="timeline-slider"
              max={Math.max((playback?.frames.length ?? 1) - 1, 0)}
              min={0}
              onChange={(event) => {
                setAutoplay(false)
                setFrameIndex(Number(event.target.value))
              }}
              type="range"
              value={Math.min(frameIndex, Math.max((playback?.frames.length ?? 1) - 1, 0))}
            />
            {currentFrame ? (
              <div className="stat-inline">
                <div>
                  <span>reward spread</span>
                  <strong>{currentFrame.reward_spread.toFixed(3)}</strong>
                </div>
                <div>
                  <span>best reward</span>
                  <strong>{currentFrame.reward_best.toFixed(3)}</strong>
                </div>
              </div>
            ) : null}
          </section>

          <section className="control-card control-card--ranking">
            <div className="control-card__header">
              <p>Top trajectories</p>
              <strong>{currentFrame?.candidates.length ?? 0} visible</strong>
            </div>
            <div className="ranking-list">
              {currentFrame?.candidates.map((candidate) => (
                <article className="ranking-row" key={candidate.rank}>
                  <div>
                    <p>#{candidate.rank}</p>
                    <strong>{candidate.reward.toFixed(3)}</strong>
                  </div>
                  <div>
                    <span>clearance {candidate.clearance.toFixed(3)}</span>
                    <span>terminal {candidate.terminal_distance.toFixed(3)}</span>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </aside>
      </section>

      {snapshot ? (
        <section className="evidence-grid">
          <section className="evidence-copy">
            <p className="eyebrow">why this lands</p>
            <h2>The browser sees the same structure the paper argues for.</h2>
            <p>
              The diffusion policy proposes joint-space futures. The learned world model rolls them
              forward. The evaluator scores safety and progress. The chosen path only executes a
              short prefix before re-planning, so the demo behaves like a compact model-predictive
              robotics loop instead of a canned animation.
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
          </section>

          <section className="evidence-signals">
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
          </section>
        </section>
      ) : null}

      {playback ? (
        <section className="summary-strip">
          <article>
            <p>mission outcome</p>
            <strong>{playback.summary.success ? 'goal secured' : 'replan limit reached'}</strong>
          </article>
          <article>
            <p>final distance</p>
            <strong>{playback.summary.final_goal_distance.toFixed(3)}</strong>
          </article>
          <article>
            <p>minimum clearance</p>
            <strong>{playback.summary.min_clearance.toFixed(3)}</strong>
          </article>
          <article>
            <p>mean forecast error</p>
            <strong>{playback.summary.average_world_error.toFixed(3)}</strong>
          </article>
        </section>
      ) : null}

      {error ? <section className="error-banner">{error}</section> : null}
    </main>
  )
}

export default App
