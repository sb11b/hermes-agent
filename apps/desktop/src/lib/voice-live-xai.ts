import { type OwnerScope, ownerScoped } from '@/api/client'
import { hermesApi } from '@/hermes'

import {
  fetchVoiceLiveStatus,
  type LiveHistoryMessage,
  type LiveTranscriptFragment,
  type VoiceLiveHandlers
} from './voice-live'

/**
 * xAI-Live voice chat: the full-duplex voice frontend that DELEGATES to Hermes.
 *
 * `voice.voice_chat_mode: xai-live` swaps the chained mic → STT → turn → TTS
 * loop for ONE xAI voice model (`grok-voice-think-fast-2.0`) that listens and
 * speaks at the same time over a single WebSocket and has no tools of its own.
 * Whenever the user finishes an utterance the session emits `onDelegation`;
 * the desktop turns that into an ordinary Hermes turn on the open session and
 * streams the reply back through `speak()`, which the voice paraphrases aloud.
 * Hermes keeps every capability — model choice, tools, memory, approvals.
 *
 * This module owns the transport only: the ephemeral client secret comes from
 * the gateway (POST /api/audio/voice-live/session in xai mode — the xAI key
 * never reaches the renderer), the WebSocket carries PCM16 24 kHz in and out,
 * and the handler contract is byte-for-byte the gpt-live one so
 * `use-voice-live-conversation` drives both engines through the same code.
 *
 * Protocol mirrors the production xAI realtime client (Junior's Talk mode):
 * subprotocol `xai-client-secret.<token>`, `session.update` with server VAD,
 * PCM16 24 kHz in + out, `grok-transcribe` with keyterms, ~100 ms
 * `input_audio_buffer.append` chunks, and PCM16 deltas queued through an
 * AudioContext by timestamp.
 *
 * Vendor contract: https://docs.x.ai/docs/guides/realtime
 */

/** Pinned to the versioned flagship; an alias move on the vendor side must not
 *  shift our voice stack. The model travels in the WebSocket URL query (see
 *  `buildXaiWebSocketUrl`) — xAI's verified `session.update` payload carries no
 *  `model` field, and sending one the server does not know risks a rejected
 *  update, so it is deliberately NOT in the payload. */
export const XAI_LIVE_MODEL = 'grok-voice-think-fast-2.0'

/** The transport type the gateway returns for this mode. */
export const XAI_TRANSPORT_TYPE = 'xai-websocket'

/** WebSocket subprotocol prefix carrying the ephemeral client secret. */
export const XAI_CLIENT_SECRET_SUBPROTOCOL_PREFIX = 'xai-client-secret.'

/** Wire rate for both directions. xAI expects 24 kHz PCM16. */
export const XAI_SAMPLE_RATE = 24_000

/** ~100 ms of mic audio per input_audio_buffer.append (the chunk size server
 *  VAD expects). */
export const MIC_CHUNK_MS = 100

/** Samples per mic chunk (100 ms at 24 kHz). */
export const MIC_CHUNK_SAMPLES = (XAI_SAMPLE_RATE * MIC_CHUNK_MS) / 1000

/** ScriptProcessor buffer; 4096 frames ≈ 170 ms at 24 kHz, split into
 *  MIC_CHUNK_SAMPLES appends per hop. */
export const MIC_PROCESSOR_BUFFER = 4_096

/** Realtime transcriptions use grok-transcribe on the same socket. */
export const XAI_INPUT_TRANSCRIPTION_MODEL = 'grok-transcribe'

/** xAI caps: at most 100 keyterms, each at most 50 characters. */
export const KEYTERM_MAX_COUNT = 100
export const KEYTERM_MAX_CHARS = 50

/**
 * ASR bias terms for Steve's world — the proper nouns transcription guesses at
 * (SNHU → "S N H U"-ish garbage, Bitsko → "bit sco"). Mirrors Junior's
 * JUNIOR_KEYTERMS with Steve's own business/school names added.
 */
export const STEVE_KEYTERMS: readonly string[] = [
  'Storykeep',
  'Newsmax',
  'Fox News',
  'Meridian',
  'Bitsko',
  'Bitsko Group',
  'SNHU',
  'uCertify',
  'DAT-325',
  'Hermes',
  'Grok',
  'Fastmail',
  'Dayton'
]

/** How much the assistant reply is coalesced into ONE spoken response: the
 *  conversation hook streams `speak()` sentence by sentence, and injecting an
 *  item + response.create per sentence would queue a separate spoken turn each
 *  time and let the model chit-chat between them. */
const SPEAK_COALESCE_MS = 400

/** Mirrors `lib/voice-live.ts`: how much conversation rides a delegation. */
const CONTEXT_WINDOW_MS = 5 * 60_000
const CONTEXT_MAX_FRAGMENTS = 80
/** Cap the accumulated transcript so a long session cannot grow unbounded. */
const TRANSCRIPT_MAX_FRAGMENTS = 2_000

/** The persona: the voice has no tools, so it converses and waits for Hermes. */
const XAI_LIVE_PERSONA =
  'You are the voice of Hermes, a personal AI agent, speaking with Steve. ' +
  'You are the ears and mouth only: you have NO tools, NO web access and NO ' +
  'memory of your own, so never claim to have done something and never ' +
  'invent facts. When Steve asks for real work — anything about his files, ' +
  'email, calendar, crons, code, news, or anything you are not certain of — ' +
  'stay quiet and wait: Hermes answers those turns and speaks through you. ' +
  'Answer only simple conversational remarks yourself, in one or two short ' +
  'sentences, with no markdown, no lists and no code. When you are handed ' +
  "Hermes' answer as an assistant message, read it aloud exactly as written: " +
  'do not paraphrase, do not add words, do not stop early. If Steve goes ' +
  'quiet, stay quiet — do not fill the silence.'

export interface XaiSessionUpdateOptions {
  /** Extra persona sentences from `voice.xai_live.instructions`. */
  instructions?: null | string
  /** xAI voice id; empty = the xAI default voice. */
  voice?: string
}

/** The realtime WebSocket URL: the model rides the query string. */
export function buildXaiWebSocketUrl(model = XAI_LIVE_MODEL): string {
  return `wss://api.x.ai/v1/realtime?model=${encodeURIComponent(model)}`
}

/** The WebSocket subprotocol carrying the ephemeral client secret. */
export function buildXaiSubprotocol(token: string): string {
  return `${XAI_CLIENT_SECRET_SUBPROTOCOL_PREFIX}${token}`
}

/**
 * The `session.update` payload, mirroring the verified production client:
 * server VAD (no idle_timeout_ms — it re-engages the server every N ms of
 * silence and a listening Steve reads as silence, looping endless responses),
 * PCM16 24 kHz in and out, `grok-transcribe` with Steve's keyterms and
 * language_hint `en`.
 */
export function buildXaiSessionUpdate({
  instructions,
  voice = ''
}: XaiSessionUpdateOptions = {}) {
  const extra = instructions?.trim()
  const persona = extra ? `${XAI_LIVE_PERSONA}\n\n${extra}` : XAI_LIVE_PERSONA

  return {
    type: 'session.update',
    session: {
      audio: {
        input: {
          format: { rate: XAI_SAMPLE_RATE, type: 'audio/pcm' },
          transcription: {
            keyterms: [...STEVE_KEYTERMS],
            language_hint: 'en',
            model: XAI_INPUT_TRANSCRIPTION_MODEL
          }
        },
        output: { format: { rate: XAI_SAMPLE_RATE, type: 'audio/pcm' } }
      },
      instructions: persona,
      turn_detection: { type: 'server_vad' },
      voice
    }
  }
}

/** Clamp to [-1, 1] so overflow cannot wrap the int16. */
export function clampSample(value: number): number {
  if (value > 1) {
    return 1
  }

  if (value < -1) {
    return -1
  }

  return value
}

/** Convert a Float32 mic frame to little-endian PCM16 bytes. */
export function floatToPcm16Bytes(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2)
  const view = new DataView(out.buffer)

  for (let index = 0; index < samples.length; index += 1) {
    const value = clampSample(samples[index]!)
    view.setInt16(index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true)
  }

  return out
}

/** Base64-encode PCM16 bytes (browser + Node 18+ globals). */
export function pcm16ToBase64(bytes: Uint8Array): string {
  let binary = ''

  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]!)
  }

  return btoa(binary)
}

/** Decode a base64 string into a Uint8Array. */
export function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return bytes
}

/** Convert interleaved PCM16 bytes to a Float32Array for the Web Audio API. */
export function pcm16ToFloat32(pcm16: Uint8Array): Float32Array {
  const view = new DataView(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength)
  const float32 = new Float32Array(view.byteLength / 2)

  for (let index = 0; index < float32.length; index += 1) {
    const int16 = view.getInt16(index * 2, true)
    float32[index] = int16 < 0 ? int16 / 0x8000 : int16 / 0x7fff
  }

  return float32
}

/** One input_audio_buffer.append payload for a Float32 mic frame. */
export function encodeMicChunk(samples: Float32Array): string {
  return pcm16ToBase64(floatToPcm16Bytes(samples))
}

/** Split a long Float32 capture into fixed-size append chunks. */
export function splitMicChunks(samples: Float32Array, chunkSamples = MIC_CHUNK_SAMPLES): Float32Array[] {
  if (chunkSamples <= 0) {
    return samples.length ? [samples.slice()] : []
  }

  const chunks: Float32Array[] = []

  for (let offset = 0; offset < samples.length; offset += chunkSamples) {
    chunks.push(samples.slice(offset, offset + chunkSamples))
  }

  return chunks
}

interface XaiServerEvent {
  type?: string
  item_id?: string
  delta?: string
  transcript?: string
  item?: { content?: Array<{ transcript?: string }> }
  output_audio?: { delta?: string }
  error?: { code?: null | string; message?: string }
  reason?: string
  usage?: { seconds?: number }
}

/** The assistant PCM16 delta from either wire shape, or null. */
function extractAudioDelta(event: XaiServerEvent): null | string {
  const delta = typeof event.delta === 'string' ? event.delta : event.output_audio?.delta

  return delta && delta.length > 0 ? delta : null
}

/** The user transcript from a committed utterance, trimmed; null when blank. */
function extractUserTranscript(event: XaiServerEvent): null | string {
  if (typeof event.transcript === 'string') {
    return event.transcript.trim() || null
  }

  for (const part of event.item?.content ?? []) {
    if (typeof part?.transcript === 'string') {
      return part.transcript.trim() || null
    }
  }

  return null
}

export class VoiceLiveXaiSession {
  private readonly handlers: VoiceLiveHandlers
  private readonly owner: null | OwnerScope
  private socket: null | WebSocket = null
  private audioContext: null | AudioContext = null
  private microphone: null | MediaStream = null
  private micSource: null | AudioNode = null
  private micProcessor: null | ScriptProcessorNode = null
  private micSink: null | GainNode = null
  private playingSources: Set<AudioBufferSourceNode> = new Set()
  private nextStartTime = 0
  private speakingProbe: null | number = null
  private speakTimer: null | number = null
  private pendingSpeak = ''
  private finalized = false
  private started = false
  private delegationCounter = 0
  /** Set while a delegation is awaiting its Hermes turn, so one utterance
   *  cannot spawn two turns. */
  private delegationInFlight = false
  /** xAI repeats transcription.completed three times per utterance (same
   *  item_id, distinct event_ids) — dedupe so one spoken turn yields exactly
   *  one transcript record and one delegation. */
  private handledTranscriptItems: Set<string> = new Set()
  private transcript: LiveTranscriptFragment[] = []
  private lastSpeaking = false
  private status: null | { instructions?: string; model: string; voice: string } = null

  constructor(handlers: VoiceLiveHandlers, owner: null | OwnerScope = null) {
    this.handlers = handlers
    this.owner = owner
  }

  get connected(): boolean {
    return this.started && this.socket?.readyState === WebSocket.OPEN
  }

  /** Recent conversation, oldest first, bounded by time and count. */
  contextWindow(): LiveTranscriptFragment[] {
    const last = this.transcript.at(-1)

    if (!last) {
      return []
    }

    const floor = last.endMs - CONTEXT_WINDOW_MS

    return this.transcript.filter(fragment => fragment.endMs >= floor).slice(-CONTEXT_MAX_FRAGMENTS)
  }

  async start(history: LiveHistoryMessage[]): Promise<void> {
    if (this.socket) {
      throw new Error('xAI-Live session already started')
    }

    // Best-effort: the configured voice and any extra persona sentences. The
    // handshake below is the authoritative gate, so a status failure (older
    // backend) degrades to defaults instead of blocking the conversation.
    const status = await fetchVoiceLiveStatus()
    this.status = status ? { instructions: status.instructions, model: status.model, voice: status.voice } : null

    const handshake = await hermesApi<{
      ok: boolean
      model?: string
      transport?: { subprotocol?: string; token?: string; type?: string; url?: string }
    }>({
      ...ownerScoped(this.owner ?? undefined),
      // The xAI transport has no SDP offer; the field is required by the
      // request model and ignored in xai mode.
      body: { history, sdp: '' },
      method: 'POST',
      path: '/api/audio/voice-live/session',
      timeoutMs: 45_000
    })

    const transport = handshake?.ok ? handshake.transport : undefined

    if (transport?.type !== XAI_TRANSPORT_TYPE || !transport.url || !transport.token) {
      throw new Error('xAI-Live session creation failed')
    }

    // Only an encrypted vendor socket; a downgraded URL is a refusal, not a
    // transport to try.
    if (!transport.url.startsWith('wss://')) {
      throw new Error('xAI-Live session creation failed (insecure transport URL)')
    }

    const model = handshake.model || this.status?.model || XAI_LIVE_MODEL
    const socket = new WebSocket(transport.url, buildXaiSubprotocol(transport.token))
    this.socket = socket
    socket.addEventListener('open', () => {
      this.started = true
      this.send(
        buildXaiSessionUpdate({
          instructions: this.status?.instructions,
          voice: this.status?.voice ?? ''
        })
      )
    })
    socket.addEventListener('message', event => this.handleEvent(String(event.data)))
    socket.addEventListener('close', () => {
      if (!this.finalized) {
        this.finish('connection_lost', null)
      }
    })
    socket.addEventListener('error', () => {
      if (!this.finalized) {
        this.handlers.onError('xAI-Live connection error', false)
      }
    })

    try {
      await this.openAudio()
    } catch (error) {
      // A start that dies after the socket opened must tear the socket down,
      // or the mic-less session keeps billing on the vendor side.
      this.teardown()

      throw error
    }

    // The model rides the WS URL query, which the gateway pinned from the same
    // config; a missing model on either side is a refusal, not a default.
    if (!model) {
      this.teardown()

      throw new Error('xAI-Live session creation failed (no model)')
    }
  }

  /** Mic capture + playback context. ECHO CANCELLATION IS REQUIRED: raw PCM
   *  has no WebRTC echo cancel, so without it the speaker bleeds into the
   *  microphone, the model transcribes its own voice and answers itself. */
  private async openAudio(): Promise<void> {
    const context = new AudioContext({ sampleRate: XAI_SAMPLE_RATE })

    try {
      await context.resume()
    } catch {
      void context.close().catch(() => undefined)
      throw new Error('xAI-Live audio could not start')
    }

    if (context.state !== 'running') {
      void context.close().catch(() => undefined)
      throw new Error('xAI-Live audio could not start')
    }

    this.audioContext = context
    this.nextStartTime = context.currentTime + 0.05

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: true,
        // REQUIRED — see the note above.
        echoCancellation: true,
        noiseSuppression: true
      }
    })
    this.microphone = stream

    const source = context.createMediaStreamSource(stream)
    const processor = context.createScriptProcessor(MIC_PROCESSOR_BUFFER, 1, 1)
    // A zero-gain sink keeps the ScriptProcessor running without echoing the
    // mic back through the speakers.
    const sink = context.createGain()
    sink.gain.value = 0
    source.connect(processor)
    processor.connect(sink)
    sink.connect(context.destination)

    this.micSource = source
    this.micProcessor = processor
    this.micSink = sink

    processor.onaudioprocess = event => {
      const frames = event.inputBuffer.getChannelData(0)

      for (const chunk of splitMicChunks(frames)) {
        this.send({ audio: encodeMicChunk(chunk), type: 'input_audio_buffer.append' })
      }
    }

    this.speakingProbe = window.setInterval(() => {
      // Audio is still queued ahead of the playhead → the voice is speaking.
      const speaking = this.audioContext !== null && this.audioContext.currentTime < this.nextStartTime - 0.02

      if (speaking !== this.lastSpeaking) {
        this.lastSpeaking = speaking
        this.handlers.onSpeakingChange?.(speaking)
      }
    }, 100)
  }

  private send(event: Record<string, unknown>): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false
    }

    this.socket.send(JSON.stringify(event))

    return true
  }

  private handleEvent(raw: string): void {
    let event: XaiServerEvent

    try {
      event = JSON.parse(raw) as XaiServerEvent
    } catch {
      return
    }

    switch (event.type) {
      case 'session.created':
      case 'session.updated':
        this.started = true

        return

      // Barge-in: the user started talking, so the assistant goes silent now.
      case 'input_audio_buffer.speech_started':
        this.stopPlayback()

        return

      case 'conversation.item.input_audio_transcription.completed': {
        const text = extractUserTranscript(event)

        if (!text) {
          return
        }

        const itemId = event.item_id

        if (itemId) {
          if (this.handledTranscriptItems.has(itemId)) {
            return
          }

          this.handledTranscriptItems.add(itemId)
        }

        const now = this.now()
        const fragment: LiveTranscriptFragment = { endMs: now, speaker: 'user', startMs: now, text }
        this.pushTranscript(fragment)

        // xAI has no native delegation event: a committed user utterance is
        // the trigger. One delegation in flight at a time — the reply arrives
        // through speak(), which must not re-enter this path.
        if (!this.delegationInFlight) {
          this.delegationInFlight = true
          this.delegationCounter += 1
          this.handlers.onDelegation(`xai_live_delegation_${this.delegationCounter}`, this.contextWindow())
        }

        return
      }

      case 'response.output_audio_transcript.delta':
      case 'response.audio_transcript.delta': {
        const now = this.now()
        const fragment: LiveTranscriptFragment = {
          endMs: now,
          speaker: 'assistant',
          startMs: now,
          text: event.delta ?? ''
        }
        this.pushTranscript(fragment)

        return
      }

      case 'response.output_audio.delta':
      case 'response.audio.delta': {
        const delta = extractAudioDelta(event)

        if (delta) {
          this.scheduleAssistantAudio(delta)
        }

        return
      }

      case 'error': {
        const code = event.error?.code ?? ''
        // Late appends after our own close are expected noise.
        if (code === 'context_injection_incomplete') {
          return
        }

        this.handlers.onError(event.error?.message ?? 'xAI-Live error', false)

        return
      }

      default:
        return
    }
  }

  private pushTranscript(fragment: LiveTranscriptFragment): void {
    this.transcript.push(fragment)

    if (this.transcript.length > TRANSCRIPT_MAX_FRAGMENTS) {
      this.transcript.splice(0, this.transcript.length - 1_500)
    }

    this.handlers.onTranscript?.(fragment)
  }

  private now(): number {
    return Math.round(performance.now())
  }

  /** Queue one PCM16 delta through the AudioContext, back-to-back. */
  private scheduleAssistantAudio(delta: string): void {
    const context = this.audioContext

    if (!context || context.state === 'closed') {
      return
    }

    const pcm = base64ToUint8Array(delta)

    if (!pcm.length) {
      return
    }

    const floats = pcm16ToFloat32(pcm)

    if (!floats.length) {
      return
    }

    const buffer = context.createBuffer(1, floats.length, XAI_SAMPLE_RATE)
    buffer.getChannelData(0).set(floats)

    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)

    const startTime = Math.max(context.currentTime + 0.05, this.nextStartTime)

    try {
      source.start(startTime)
    } catch {
      return
    }

    this.nextStartTime = startTime + buffer.duration
    this.playingSources.add(source)
    source.onended = () => {
      this.playingSources.delete(source)

      try {
        source.disconnect()
      } catch {
        // Already gone.
      }
    }
  }

  /** Barge-in: kill every scheduled assistant buffer so the mic is heard. */
  private stopPlayback(): void {
    for (const source of this.playingSources) {
      try {
        source.onended = null
        source.stop()
        source.disconnect()
      } catch {
        // Already stopped.
      }
    }

    this.playingSources.clear()
    this.nextStartTime = this.audioContext ? this.audioContext.currentTime + 0.05 : 0

    if (this.lastSpeaking) {
      this.lastSpeaking = false
      this.handlers.onSpeakingChange?.(false)
    }
  }

  /** Quiet progress for the live model. xAI has no thinking channel; the
   *  voice stays silent while Hermes works, which is the same user-visible
   *  behaviour. */
  think(): void {
    // No-op by design.
  }

  /** A result the voice should say aloud: coalesced into one assistant item
   *  plus one spoken response per Hermes reply. */
  speak(_delegationId: null | string, content: string): void {
    const text = content.replace(/\s+/g, ' ').trim()

    if (!text) {
      return
    }

    this.pendingSpeak = this.pendingSpeak ? `${this.pendingSpeak} ${text}` : text
    this.delegationInFlight = false

    if (this.speakTimer !== null) {
      return
    }

    this.speakTimer = window.setTimeout(() => {
      this.speakTimer = null
      this.flushSpeak()
    }, SPEAK_COALESCE_MS)
  }

  private flushSpeak(): void {
    const text = this.pendingSpeak
    this.pendingSpeak = ''

    if (!text || !this.connected) {
      return
    }

    this.send({
      item: { content: [{ text, type: 'text' }], role: 'assistant', type: 'message' },
      type: 'conversation.item.create'
    })
    this.send({ type: 'response.create' })
  }

  /** Steer the live persona mid-conversation: ask it to answer what was said.
   *  The xAI socket has no instructions channel, so a `response.create` makes
   *  the model answer the committed utterance now — the hook's intent. The
   *  text is dropped: server_vad already committed the audio it describes. */
  instruct(_content: string): void {
    this.send({ type: 'response.create' })
  }

  setMuted(muted: boolean): void {
    for (const track of this.microphone?.getAudioTracks() ?? []) {
      track.enabled = !muted
    }
  }

  /** Graceful close: tear the transport down and report the end. */
  close(): void {
    if (this.finalized) {
      return
    }

    this.finalized = true
    this.teardown()
    this.handlers.onClosed('close_requested', null)
  }

  private teardown(): void {
    if (this.speakTimer) {
      window.clearTimeout(this.speakTimer)
      this.speakTimer = null
    }

    if (this.speakingProbe) {
      window.clearInterval(this.speakingProbe)
      this.speakingProbe = null
    }

    if (this.lastSpeaking) {
      this.lastSpeaking = false
      this.handlers.onSpeakingChange?.(false)
    }

    for (const source of this.playingSources) {
      try {
        source.onended = null
        source.stop()
        source.disconnect()
      } catch {
        // Already stopped.
      }
    }

    this.playingSources.clear()
    this.pendingSpeak = ''

    if (this.micProcessor) {
      this.micProcessor.onaudioprocess = null

      try {
        this.micProcessor.disconnect()
      } catch {
        // Already gone.
      }

      this.micProcessor = null
    }

    for (const node of [this.micSource, this.micSink]) {
      if (!node) {
        continue
      }

      try {
        node.disconnect()
      } catch {
        // Already gone.
      }
    }

    this.micSource = null
    this.micSink = null

    for (const track of this.microphone?.getTracks() ?? []) {
      track.stop()
    }

    this.microphone = null

    void this.audioContext?.close().catch(() => undefined)
    this.audioContext = null

    const socket = this.socket
    this.socket = null

    if (socket) {
      socket.onopen = null
      socket.onmessage = null
      socket.onclose = null
      socket.onerror = null

      try {
        socket.close()
      } catch {
        // Already closed.
      }
    }

    this.handledTranscriptItems.clear()
    this.delegationInFlight = false
  }

  private finish(reason: string, usageSeconds: null | number): void {
    if (this.finalized) {
      return
    }

    this.finalized = true
    this.teardown()
    this.handlers.onClosed(reason, usageSeconds)
  }
}
