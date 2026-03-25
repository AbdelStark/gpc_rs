/// <reference lib="webworker" />

import initWasm, { DemoRuntime } from '../wasm/pkg/gpc_wasm.js'
import type {
  MissionPlayback,
  PlannerMode,
  RuntimeBuildConfig,
  RuntimeSnapshot,
  WorkerRequest,
  WorkerResponse,
} from '../types'

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope

let runtimePromise: Promise<DemoRuntime> | null = null
type DemoRuntimeApi = DemoRuntime & {
  rebuild(config: RuntimeBuildConfig): RuntimeSnapshot | Promise<RuntimeSnapshot>
}

function describePlannerMode(mode: PlannerMode) {
  switch (mode) {
    case 'policy':
      return 'Generating the raw policy trajectory.'
    case 'rank':
      return 'Ranking candidate trajectories in the world model.'
    case 'opt':
      return 'Refining the selected candidate with gradient search.'
  }
}

function postMessageSafe(message: WorkerResponse) {
  workerScope.postMessage(message)
}

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      await initWasm(new URL('../wasm/pkg/gpc_wasm_bg.wasm', import.meta.url))
      return new DemoRuntime()
    })().catch((error) => {
      runtimePromise = null
      throw error
    })
  }

  return runtimePromise
}

async function bootstrap() {
  postMessageSafe({
    type: 'status',
    phase: 'initializing',
    message: 'Compiling the Rust planner and warming the browser-side models.',
  })
  const runtime = (await getRuntime()) as DemoRuntimeApi
  const snapshot = runtime.snapshot() as RuntimeSnapshot
  postMessageSafe({ type: 'snapshot', snapshot })
}

async function rebuild(request: Extract<WorkerRequest, { type: 'rebuild' }>) {
  const runtime = (await getRuntime()) as DemoRuntimeApi
  postMessageSafe({
    type: 'status',
    phase: 'rebuilding',
    message: 'Rebuilding the browser-side runtime with new training settings.',
  })
  const snapshot = await runtime.rebuild(request.config)
  postMessageSafe({ type: 'snapshot', snapshot })
}

async function simulate(request: Extract<WorkerRequest, { type: 'simulate' }>) {
  const runtime = (await getRuntime()) as DemoRuntimeApi
  postMessageSafe({
    type: 'status',
    phase: 'planning',
    message: describePlannerMode(request.mode),
  })
  const playback = runtime.simulate_mission(
    request.missionId,
    request.mode,
    request.numCandidates,
  ) as MissionPlayback
  postMessageSafe({ type: 'playback', playback })
}

workerScope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  void (async () => {
    try {
      if (event.data.type === 'bootstrap') {
        await bootstrap()
      } else if (event.data.type === 'rebuild') {
        await rebuild(event.data)
      } else {
        await simulate(event.data)
      }
    } catch (error) {
      postMessageSafe({
        type: 'error',
        message: error instanceof Error ? error.message : 'Worker failure',
      })
    }
  })()
}

export {}
