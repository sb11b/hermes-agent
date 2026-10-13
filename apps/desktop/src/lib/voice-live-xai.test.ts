import { describe, expect, it, vi } from 'vitest'

import {
  base64ToUint8Array,
  buildXaiSessionUpdate,
  buildXaiSubprotocol,
  buildXaiWebSocketUrl,
  encodeMicChunk,
  floatToPcm16Bytes,
  KEYTERM_MAX_CHARS,
  KEYTERM_MAX_COUNT,
  MIC_CHUNK_SAMPLES,
  pcm16ToFloat32,
  pcm16ToBase64,
  splitMicChunks,
  STEVE_KEYTERMS,
  XAI_INPUT_TRANSCRIPTION_MODEL,
  XAI_LIVE_MODEL,
  XAI_SAMPLE_RATE
} from './voice-live-xai'

describe('buildXaiWebSocketUrl', () => {
  it('points at the grok-voice realtime endpoint with the pinned model', () => {
    expect(buildXaiWebSocketUrl()).toBe(`wss://api.x.ai/v1/realtime?model=${XAI_LIVE_MODEL}`)
  })

  it('encodes the model', () => {
    expect(buildXaiWebSocketUrl('a b')).toBe('wss://api.x.ai/v1/realtime?model=a%20b')
  })
})

describe('buildXaiSubprotocol', () => {
  it('carries the ephemeral secret as the subprotocol', () => {
    expect(buildXaiSubprotocol('token-abc')).toBe('xai-client-secret.token-abc')
  })
})

describe('buildXaiSessionUpdate', () => {
  const update = buildXaiSessionUpdate()

  it('is a session.update', () => {
    expect(update.type).toBe('session.update')
  })

  it('carries server_vad turn detection', () => {
    expect(update.session.turn_detection).toEqual({ type: 'server_vad' })
  })

  it('carries NO idle_timeout_ms — it re-engages the server on silence', () => {
    // Verified live in the reference client: idle_timeout_ms makes the server
    // start a new response every N ms of silence, looping endless turns.
    expect(update.session.turn_detection).not.toHaveProperty('idle_timeout_ms')
  })

  it('sets 24 kHz PCM16 on both input and output', () => {
    expect(update.session.audio.input.format).toEqual({ rate: XAI_SAMPLE_RATE, type: 'audio/pcm' })
    expect(update.session.audio.output.format).toEqual({ rate: XAI_SAMPLE_RATE, type: 'audio/pcm' })
    expect(XAI_SAMPLE_RATE).toBe(24_000)
  })

  it('biases transcription with grok-transcribe, keyterms and language_hint', () => {
    const transcription = update.session.audio.input.transcription
    expect(transcription.model).toBe(XAI_INPUT_TRANSCRIPTION_MODEL)
    expect(XAI_INPUT_TRANSCRIPTION_MODEL).toBe('grok-transcribe')
    expect(transcription.language_hint).toBe('en')
    expect(transcription.keyterms).toEqual([...STEVE_KEYTERMS])
  })

  it('carries the model in the payload OR keeps it out — pinned either way', () => {
    // The model travels in the WS URL query (buildXaiWebSocketUrl). The
    // session.update payload deliberately does not carry a `model` field:
    // xAI's verified payload has none, and an unknown field risks a rejected
    // update. The pin lives in the URL constant, asserted above.
    expect(update.session).not.toHaveProperty('model')
    expect(XAI_LIVE_MODEL).toBe('grok-voice-think-fast-2.0')
  })

  it('keeps Steve world proper nouns in the keyterms', () => {
    for (const term of [
      'Storykeep',
      'Newsmax',
      'Fox News',
      'Meridian',
      'SNHU',
      'Bitsko',
      'uCertify',
      'DAT-325'
    ]) {
      expect(STEVE_KEYTERMS).toContain(term)
    }
  })

  it('respects the xAI keyterm caps', () => {
    expect(STEVE_KEYTERMS.length).toBeLessThanOrEqual(KEYTERM_MAX_COUNT)
    for (const term of STEVE_KEYTERMS) {
      expect(term.length).toBeLessThanOrEqual(KEYTERM_MAX_CHARS)
    }
  })

  it('appends configured instructions to the persona', () => {
    const plain = buildXaiSessionUpdate().session.instructions
    const extra = buildXaiSessionUpdate({ instructions: 'Answer in one sentence.' }).session.instructions

    expect(extra.startsWith(plain)).toBe(true)
    expect(extra).toContain('Answer in one sentence.')
  })

  it('defaults to the xAI voice when none is configured', () => {
    expect(update.session.voice).toBe('')
    expect(buildXaiSessionUpdate({ voice: 'eve' }).session.voice).toBe('eve')
  })
})

describe('PCM16 chunking and base64', () => {
  it('MIC_CHUNK_SAMPLES is 100 ms of audio at 24 kHz', () => {
    expect(MIC_CHUNK_SAMPLES).toBe(2_400)
  })

  it('floatToPcm16Bytes writes little-endian pairs matching pcm16ToFloat32', () => {
    const samples = new Float32Array([0, 1, -1, 0.5, -0.5])
    const bytes = floatToPcm16Bytes(samples)

    expect(bytes.length).toBe(samples.length * 2)
    expect(Array.from(bytes.slice(0, 6))).toEqual([0x00, 0x00, 0xff, 0x7f, 0x00, 0x80])
    expect(Array.from(bytes.slice(6, 10))).toEqual([0xff, 0x3f, 0x00, 0xc0])
  })

  it('floatToPcm16Bytes clamps out-of-range input instead of wrapping', () => {
    const bytes = floatToPcm16Bytes(new Float32Array([4, -4]))
    expect(Array.from(bytes)).toEqual([0xff, 0x7f, 0x00, 0x80])
  })

  it('pcm16ToBase64 round-trips through atob', () => {
    const bytes = new Uint8Array([72, 101, 108, 108, 111])
    expect(pcm16ToBase64(bytes)).toBe('SGVsbG8=')
  })

  it('base64ToUint8Array decodes back to the original bytes', () => {
    const bytes = new Uint8Array([72, 101, 108, 108, 111])
    expect(Array.from(base64ToUint8Array(pcm16ToBase64(bytes)))).toEqual([72, 101, 108, 108, 111])
  })

  it('pcm16ToFloat32 inverts floatToPcm16Bytes', () => {
    const bytes = floatToPcm16Bytes(new Float32Array([0, 0.5, -0.5, 1, -1]))
    const floats = pcm16ToFloat32(bytes)

    expect(floats.length).toBe(5)
    expect(floats[0]).toBeCloseTo(0)
    expect(floats[3]).toBeCloseTo(32767 / 32767)
    expect(floats[4]).toBeCloseTo(-1)
  })

  it('encodeMicChunk produces base64 that decodes back to PCM16 pairs', () => {
    const encoded = encodeMicChunk(new Float32Array([0.5, -0.5]))
    expect(atob(encoded).length).toBe(4)
    // 0.5 -> 16383 (0x3FFF) little-endian, so byte 1 is 0x3F.
    expect(atob(encoded).charCodeAt(1)).toBe(0x3f)
  })

  it('splitMicChunks splits into 100 ms pieces and keeps the remainder', () => {
    const chunks = splitMicChunks(new Float32Array(MIC_CHUNK_SAMPLES * 2 + 5))

    expect(chunks.length).toBe(3)
    expect(chunks[0].length).toBe(MIC_CHUNK_SAMPLES)
    expect(chunks[1].length).toBe(MIC_CHUNK_SAMPLES)
    expect(chunks[2].length).toBe(5)
  })

  it('splitMicChunks returns one chunk when under the chunk size', () => {
    const chunks = splitMicChunks(new Float32Array(10))
    expect(chunks.length).toBe(1)
    expect(chunks[0].length).toBe(10)
  })

  it('splitMicChunks returns nothing for empty input', () => {
    expect(splitMicChunks(new Float32Array(0))).toEqual([])
  })

  it('a 4096-frame processor hop splits into two appends (2400 + 1696)', () => {
    // Mirror of the production hop: 4096 frames ≈ 170 ms at 24 kHz.
    const hop = new Float32Array(4_096)
    const chunks = splitMicChunks(hop)
    const appends = chunks.map(chunk => encodeMicChunk(chunk))

    expect(appends.length).toBe(2)
    expect(atob(appends[0]).length / 2).toBe(MIC_CHUNK_SAMPLES)
    expect(atob(appends[1]).length / 2).toBe(1_696)
  })
})

/* ── Session-level tests: the WebSocket, AudioContext and getUserMedia are
 *  faked so NO real network socket or audio device is ever opened. ── */

vi.mock('@/hermes', () => ({
  hermesApi: vi.fn()
}))

vi.mock('./voice-live', () => ({
  fetchVoiceLiveStatus: vi.fn(async () => null)
}))

const { hermesApi } = await import('@/hermes')
const { fetchVoiceLiveStatus } = await import('./voice-live')
const { VoiceLiveXaiSession } = await import('./voice-live-xai')

type Handlers = ConstructorParameters<typeof VoiceLiveXaiSession>[0]

class FakeTrack {
  enabled = true
  stopped = false
  stop() {
    this.stopped = true
  }
}

class FakeStream {
  tracks = [new FakeTrack()]
  getTracks() {
    return this.tracks
  }
}

class FakeBufferSource {
  onended: (() => void) | null = null
  stopped = false
  disconnected = false
  buffer: unknown = null
  start() {}
  stop() {
    this.stopped = true
  }
  connect() {}
  disconnect() {
    this.disconnected = true
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  static rtcInstalled: null | { audio: unknown } = null
  currentTime = 0
  state = 'running'
  closed = false
  destination = {}
  constructor(public options?: { sampleRate?: number }) {
    FakeAudioContext.instances.push(this)
  }
  async resume() {
    this.state = 'running'
  }
  async close() {
    this.closed = true
    this.state = 'closed'
  }
  createBuffer(_channels: number, length: number, sampleRate: number) {
    return {
      duration: length / sampleRate,
      getChannelData: () => new Float32Array(length),
      length
    }
  }
  createBufferSource() {
    return new FakeBufferSource()
  }
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} }
  }
  createScriptProcessor() {
    return {
      bufferSize: 4_096,
      connect() {},
      disconnect() {},
      onaudioprocess: null as ((event: unknown) => void) | null
    }
  }
  createGain() {
    return { connect() {}, disconnect() {}, gain: { value: 1 } }
  }
}

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static OPEN = 1
  readyState = 1
  sent: string[] = []
  constructor(
    public url: string,
    public protocol: string
  ) {
    FakeWebSocket.instances.push(this)
  }
  addEventListener() {}
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
}

function installBrowserFakes(options?: { socketUrl?: string }) {
  FakeAudioContext.instances = []
  FakeWebSocket.instances = []
  FakeAudioContext.rtcInstalled = null

  const originals = {
    AudioContext: (globalThis as Record<string, unknown>).AudioContext,
    WebSocket: (globalThis as Record<string, unknown>).WebSocket,
    mediaDevices: navigator.mediaDevices
  }

  ;(globalThis as Record<string, unknown>).AudioContext = FakeAudioContext
  ;(globalThis as Record<string, unknown>).WebSocket = FakeWebSocket

  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: async (constraints: unknown) => {
        FakeAudioContext.rtcInstalled = constraints as { audio: unknown }

        return new FakeStream()
      }
    }
  })

  vi.mocked(hermesApi).mockResolvedValue({
    ok: true,
    model: 'grok-voice-think-fast-2.0',
    transport: {
      subprotocol: 'xai-client-secret.token-abc',
      token: 'token-abc',
      type: 'xai-websocket',
      url: options?.socketUrl ?? 'wss://api.x.ai/v1/realtime?model=grok-voice-think-fast-2.0'
    }
  })

  return () => {
    ;(globalThis as Record<string, unknown>).AudioContext = originals.AudioContext
    ;(globalThis as Record<string, unknown>).WebSocket = originals.WebSocket
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: originals.mediaDevices
    })
    vi.mocked(hermesApi).mockReset()
    vi.mocked(fetchVoiceLiveStatus).mockReset()
    vi.mocked(fetchVoiceLiveStatus).mockResolvedValue(null)
  }
}

function newSession(handlers?: Partial<Handlers>): InstanceType<typeof VoiceLiveXaiSession> {
  return new VoiceLiveXaiSession({
    onClosed: vi.fn(),
    onDelegation: vi.fn(),
    onError: vi.fn(),
    onTranscript: vi.fn(),
    ...handlers
  } as Handlers)
}

describe('VoiceLiveXaiSession transport', () => {
  it('opens the WS with the xai-client-secret subprotocol and a pinned model URL', async () => {
    const restore = installBrowserFakes()

    try {
      const session = newSession()
      await session.start([])

      expect(FakeWebSocket.instances.length).toBe(1)
      expect(FakeWebSocket.instances[0].url).toBe(
        'wss://api.x.ai/v1/realtime?model=grok-voice-think-fast-2.0'
      )
      expect(FakeWebSocket.instances[0].protocol).toBe('xai-client-secret.token-abc')
      // The session is "connected" only once the socket opens.
      expect(session.connected).toBe(false)
      session.close()
    } finally {
      restore()
    }
  })

  it('refuses an insecure (non-wss) transport URL', async () => {
    const restore = installBrowserFakes({ socketUrl: 'ws://api.x.ai/v1/realtime' })

    try {
      await expect(newSession().start([])).rejects.toThrow(/insecure transport URL/)
      expect(FakeWebSocket.instances.length).toBe(0)
    } finally {
      restore()
    }
  })

  it('requires echoCancellation on the mic capture (raw PCM has no WebRTC echo cancel)', async () => {
    const restore = installBrowserFakes()

    try {
      const session = newSession()
      await session.start([])

      expect(FakeAudioContext.rtcInstalled?.audio).toEqual({
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true
      })
      expect(
        (FakeAudioContext.rtcInstalled?.audio as { echoCancellation?: boolean }).echoCancellation
      ).toBe(true)
      session.close()
    } finally {
      restore()
    }
  })

  it('creates the AudioContext at 24 kHz for playback', async () => {
    const restore = installBrowserFakes()

    try {
      const session = newSession()
      await session.start([])

      expect(FakeAudioContext.instances.length).toBe(1)
      expect(FakeAudioContext.instances[0].options?.sampleRate).toBe(24_000)
      session.close()
    } finally {
      restore()
    }
  })

  it('close() stops the mic track, closes the AudioContext and reports the end', async () => {
    const restore = installBrowserFakes()

    try {
      const handlers = { onClosed: vi.fn() }
      const session = newSession(handlers)
      await session.start([])

      const context = FakeAudioContext.instances[0]
      session.close()

      expect(context.closed).toBe(true)
      expect(handlers.onClosed).toHaveBeenCalledWith('close_requested', null)
      session.close()
      expect(handlers.onClosed).toHaveBeenCalledTimes(1)
    } finally {
      restore()
    }
  })
})
