/// <reference lib="webworker" />

import initWasm, { DemoRuntime } from '../wasm/pkg/gpc_wasm.js'
import type { MissionPlayback, RuntimeSnapshot, WorkerRequest, WorkerResponse } from '../types'

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope

let runtimePromise: Promise<DemoRuntime> | null = null

function postMessageSafe(message: WorkerResponse) {
  workerScope.postMessage(message)
}

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      await initWasm(new URL('../wasm/pkg/gpc_wasm_bg.wasm', import.meta.url))
      return new DemoRuntime()
    })()
  }

  return runtimePromise
}

async function bootstrap() {
  postMessageSafe({
    type: 'status',
    phase: 'initializing',
    message: 'Compiling the Rust planner and warming the browser-side models.',
  })
  const runtime = await getRuntime()
  const snapshot = runtime.snapshot() as RuntimeSnapshot
  postMessageSafe({ type: 'snapshot', snapshot })
}

async function simulate(request: Extract<WorkerRequest, { type: 'simulate' }>) {
  const runtime = await getRuntime()
  postMessageSafe({
    type: 'status',
    phase: 'planning',
    message: 'Sampling candidates and ranking them in the world model.',
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
        return
      }

      await simulate(event.data)
    } catch (error) {
      postMessageSafe({
        type: 'error',
        message: error instanceof Error ? error.message : 'Worker failure',
      })
    }
  })()
}

export {}
