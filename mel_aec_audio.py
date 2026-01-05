"""
Shared audio helpers that bridge Melodeus to the mel-aec engine.

We keep a singleton AudioStream so microphone capture and playback
share the same duplex stream (required for echo cancellation).
Utility functions handle float/int16 conversion and sample-rate
resampling for the STT (Deepgram) and TTS (ElevenLabs) pipelines.
"""

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional
import resampy

import numpy as np
from scipy import signal

from audio_aec import AudioStream


@dataclass(frozen=True)
class StreamSettings:
    """Configuration applied to the shared mel-aec duplex stream."""

    sample_rate: int = 48000  # Hz
    channels: int = 1
    buffer_size: int = 480  # Samples per audio chunk
    enable_aec: bool = True
    aec_filter_length: int = 2048
    input_device: Optional[str] = None
    output_device: Optional[str] = None

_DEFAULT_STREAM_SETTINGS = StreamSettings()
_stream_lock = threading.Lock()
_stream_settings: StreamSettings = _DEFAULT_STREAM_SETTINGS
_shared_stream: Optional[AudioStream] = None
_stream_started = False
_MISSING = object()


@dataclass(frozen=True)
class AutoGainSettings:
    target_rms: float
    smoothing: float
    ratio_max: float
    ratio_min: float
    min_gain: float
    max_gain: float


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (TypeError, ValueError, AttributeError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _load_auto_gain_settings() -> AutoGainSettings:
    target = _env_float("AEC_TARGET_RMS", 0.2)
    smoothing = float(np.clip(_env_float("AEC_GAIN_SMOOTHING", 0.1), 0.0, 1.0))
    ratio_max = max(1.0, _env_float("AEC_GAIN_RATIO_MAX", 3.0))
    ratio_min_raw = _env_float("AEC_GAIN_RATIO_MIN", 0.3333333333)
    ratio_min = max(0.1, min(ratio_min_raw, 1.0)) if math.isfinite(ratio_min_raw) else 0.3333333333
    min_gain = max(1e-6, _env_float("AEC_MIN_GAIN", 0.1))
    max_gain_candidate = _env_float("AEC_MAX_GAIN", 0.8)
    max_gain = max(min_gain, max_gain_candidate if math.isfinite(max_gain_candidate) else 20.0)
    target = target if target > 0.0 and math.isfinite(target) else 0.2
    return AutoGainSettings(
        target_rms=target,
        smoothing=smoothing,
        ratio_max=ratio_max,
        ratio_min=ratio_min,
        min_gain=min_gain,
        max_gain=max_gain,
    )


_AUTO_GAIN_SETTINGS = _load_auto_gain_settings()
_auto_gain_lock = threading.Lock()
global _auto_gain_value
_auto_gain_value = float(np.clip(1.0, _AUTO_GAIN_SETTINGS.min_gain, _AUTO_GAIN_SETTINGS.max_gain))
_AUTO_GAIN_ENABLED = True


def _normalize_device_name(name: Optional[Any]) -> Optional[str]:
    if name is None:
        return None
    value = str(name).strip()
    return value or None


def _coerce_optional_bool(value: Optional[Any]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


_auto_gain_enabled_env = _coerce_optional_bool(os.getenv("AEC_ENABLE_AUTO_GAIN"))
if _auto_gain_enabled_env is not None:
    _AUTO_GAIN_ENABLED = _auto_gain_enabled_env and False


def _reset_auto_gain() -> None:
    global _auto_gain_value
    with _auto_gain_lock:
        _auto_gain_value = float(
            np.clip(1.0, _AUTO_GAIN_SETTINGS.min_gain, _AUTO_GAIN_SETTINGS.max_gain)
        )


def current_auto_gain() -> float:
    global _auto_gain_value
    with _auto_gain_lock:
        return _auto_gain_value


def _update_auto_gain(rms: float) -> float:
    global _auto_gain_value
    settings = _AUTO_GAIN_SETTINGS
    if not _AUTO_GAIN_ENABLED or settings.target_rms <= 0.0:
        with _auto_gain_lock:
            clamped = float(np.clip(_auto_gain_value, settings.min_gain, settings.max_gain))
            _auto_gain_value = clamped
            return clamped
    if not math.isfinite(rms) or rms < 0.0:
        rms = 0.0
    with _auto_gain_lock:
        auto_gain = _auto_gain_value
        if rms > 1e-6:
            ratio = settings.target_rms / max(rms, 1e-6)
            ratio = float(np.clip(ratio, settings.ratio_min, settings.ratio_max))
            desired = float(np.clip(auto_gain * ratio, settings.min_gain, settings.max_gain))
        else:
            desired = min(settings.max_gain, auto_gain * settings.ratio_max)
        smoothing = float(np.clip(settings.smoothing, 0.0, 1.0))
        updated = auto_gain + (desired - auto_gain) * smoothing
        if not math.isfinite(updated):
            updated = settings.min_gain
        updated = float(np.clip(updated, settings.min_gain, settings.max_gain))
        _auto_gain_value = updated
        return updated


def configure_audio_stream(
    *,
    sample_rate: Any = _MISSING,
    channels: Any = _MISSING,
    buffer_size: Any = _MISSING,
    enable_aec: Any = _MISSING,
    aec_filter_length: Any = _MISSING,
    input_device: Any = _MISSING,
    output_device: Any = _MISSING,
) -> None:
    """
    Update global stream settings. Stops and releases the shared stream if parameters change.
    """

    def _coerce_int(value: Optional[Any], minimum: int) -> Optional[int]:
        if value is None:
            return None
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate >= minimum else None

    global _stream_settings, _shared_stream, _stream_started

    with _stream_lock:
        current = _stream_settings

        if sample_rate in (_MISSING, None):
            new_sample_rate = current.sample_rate
        else:
            new_sample_rate = _coerce_int(sample_rate, 4000) or current.sample_rate

        if channels in (_MISSING, None):
            new_channels = current.channels
        else:
            new_channels = _coerce_int(channels, 1) or current.channels

        if buffer_size is _MISSING:
            coalesced_buffer = current.buffer_size
        else:
            coalesced_buffer = _coerce_int(buffer_size, 1)
            if coalesced_buffer is None:
                coalesced_buffer = current.buffer_size

        if buffer_size is _MISSING and sample_rate not in (_MISSING, None) and new_sample_rate != current.sample_rate:
                coalesced_buffer = max(1, new_sample_rate // 100)  # ~10 ms buffers
        new_aec_filter = (
            _coerce_int(aec_filter_length, 32) if aec_filter_length not in (_MISSING, None) else None
        )
        if new_aec_filter is None:
            new_aec_filter = current.aec_filter_length

        if enable_aec is _MISSING:
            new_enable_aec = current.enable_aec
        else:
            coerced_enable = _coerce_optional_bool(enable_aec)
            new_enable_aec = current.enable_aec if coerced_enable is None else coerced_enable

        if input_device is _MISSING:
            new_input = current.input_device
        else:
            new_input = _normalize_device_name(input_device)

        if output_device is _MISSING:
            new_output = current.output_device
        else:
            new_output = _normalize_device_name(output_device)

        new_settings = StreamSettings(
            sample_rate=new_sample_rate,
            channels=new_channels,
            buffer_size=coalesced_buffer,
            enable_aec=new_enable_aec,
            aec_filter_length=new_aec_filter,
            input_device=new_input,
            output_device=new_output,
        )

        if new_settings == current:
            return

        # Tear down any existing stream so the next access picks up new settings
        if _shared_stream is not None:
            try:
                if _stream_started:
                    _shared_stream.stop()
            except Exception:
                pass
            finally:
                _shared_stream = None
                _stream_started = False

        _stream_settings = new_settings
        _reset_auto_gain()


def configure_audio_stream_from_config(config: Any) -> None:
    """
    Apply configuration values from a VoiceAIConfig to the shared mel-aec stream.
    The function gracefully ignores missing attributes to remain backward compatible.
    """
    audio_cfg = getattr(config, "audio", None)
    stt_cfg = getattr(config, "stt", None)
    tts_cfg = getattr(config, "tts", None)
    conversation_cfg = getattr(config, "conversation", None)

    sample_rate = getattr(audio_cfg, "stream_sample_rate", _MISSING) if audio_cfg else _MISSING
    channels = getattr(audio_cfg, "stream_channels", _MISSING) if audio_cfg else _MISSING
    buffer_size = getattr(audio_cfg, "stream_buffer_size", _MISSING) if audio_cfg else _MISSING

    enable_aec = getattr(audio_cfg, "stream_enable_aec", _MISSING) if audio_cfg else _MISSING
    if enable_aec is _MISSING and conversation_cfg is not None:
        enable_aec = getattr(conversation_cfg, "enable_echo_cancellation", _MISSING)

    aec_filter_length = getattr(audio_cfg, "aec_filter_length", _MISSING) if audio_cfg else _MISSING
    if aec_filter_length is _MISSING and conversation_cfg is not None:
        aec_filter_length = getattr(conversation_cfg, "aec_filter_length", _MISSING)

    input_device = getattr(audio_cfg, "input_device_name", _MISSING) if audio_cfg else _MISSING
    if (input_device is _MISSING or not input_device) and stt_cfg is not None:
        fallback_input = getattr(stt_cfg, "input_device_name", None)
        if fallback_input:
            input_device = fallback_input
        elif input_device is _MISSING:
            input_device = _MISSING

    output_device = getattr(audio_cfg, "output_device_name", _MISSING) if audio_cfg else _MISSING
    if (output_device is _MISSING or not output_device) and tts_cfg is not None:
        fallback_output = getattr(tts_cfg, "output_device_name", None)
        if fallback_output:
            output_device = fallback_output
        elif output_device is _MISSING:
            output_device = _MISSING

    configure_audio_stream(
        sample_rate=sample_rate,
        channels=channels,
        buffer_size=buffer_size,
        enable_aec=enable_aec,
        aec_filter_length=aec_filter_length,
        input_device=input_device,
        output_device=output_device,
    )


def _current_settings() -> StreamSettings:
    with _stream_lock:
        return _stream_settings


def _get_shared_stream() -> AudioStream:
    """Return the singleton AudioStream instance, creating it on demand."""
    global _shared_stream
    with _stream_lock:
        if _shared_stream is None:
            settings = _stream_settings
            _shared_stream = AudioStream(
                sample_rate=settings.sample_rate,
                channels=settings.channels,
                buffer_size=settings.buffer_size,
                enable_aec=settings.enable_aec,
                aec_filter_length=settings.aec_filter_length,
                input_device=settings.input_device,
                output_device=settings.output_device,
            )
            print(
                "🎚️ mel-aec stream initialized "
                f"@ {settings.sample_rate}Hz, buffer={settings.buffer_size} samples, "
                f"channels={settings.channels}, AEC={'on' if settings.enable_aec else 'off'}, "
                f"input={settings.input_device or 'default'}, "
                f"output={settings.output_device or 'default'}"
            )
        return _shared_stream


def ensure_stream_started() -> AudioStream:
    """Start the duplex stream exactly once and return it."""
    global _stream_started
    stream = _get_shared_stream()
    with _stream_lock:
        if not _stream_started:
            stream.start()
            _stream_started = True
    return stream


def stop_stream():
    """Stop the shared stream (used during shutdown)."""
    global _stream_started
    with _stream_lock:
        if _shared_stream and _stream_started:
            _shared_stream.stop()
            _stream_started = False
            _reset_auto_gain()


def shared_sample_rate() -> int:
    """Expose the sample rate used by the shared stream."""
    with _stream_lock:
        return _stream_settings.sample_rate


def int16_bytes_to_float(audio_bytes: bytes) -> np.ndarray:
    """Convert signed 16-bit PCM bytes to float32 in [-1, 1]."""
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def prepare_capture_chunk(float_audio: np.ndarray, target_rate: int) -> bytes:
    """
    Convert float32 capture audio into PCM16 bytes at the desired sample rate and length.
    """
    if float_audio is None:
        return b""

    audio = np.asarray(float_audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""

    settings = _current_settings()
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    active_gain = _update_auto_gain(rms)

    resampled = _resample(audio, settings.sample_rate, target_rate)

    if resampled.size:
        resampled = resampled * active_gain

    return float_to_int16_bytes(resampled)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio with sane defaults."""
    if src_rate == dst_rate:
        return audio
    return resampy.resample(audio, src_rate, dst_rate)


def write_playback_pcm(pcm_bytes: bytes, source_rate: int) -> int:
    """
    Write PCM16 playback data (e.g. ElevenLabs chunks) into the shared stream.

    Returns the number of samples written to the mel-aec stream.
    """
    stream = ensure_stream_started()
    target_rate = shared_sample_rate()
    float_audio = int16_bytes_to_float(pcm_bytes)
    resampled = _resample(float_audio, source_rate, target_rate)
    return stream.write(resampled)


def write_playback_float(audio: np.ndarray, source_rate: int) -> int:
    """
    Write float32 audio into the shared stream, resampling if needed.
    """
    stream = ensure_stream_started()
    target_rate = shared_sample_rate()
    resampled = _resample(audio, source_rate, target_rate)
    return stream.write(resampled)


def interrupt_playback() -> None:
    """Interrupt the shared output stream immediately if it is running."""
    with _stream_lock:
        stream = _shared_stream
        started = _stream_started
    if stream and started:
        try:
            stream.interrupt()
        except Exception as exc:
            print(f"⚠️ Unable to interrupt mel-aec playback: {exc}")


def read_capture_chunk(target_samples: int, target_rate: int) -> bytes:
    """
    Read microphone audio from the shared stream and return PCM16 bytes
    at the desired sample rate (Deepgram expects 16 kHz).
    """
    settings = _current_settings()
    stream = ensure_stream_started()
    # Request enough samples from the shared stream to satisfy the target chunk.
    samples_needed = max(
        int(math.ceil(target_samples * settings.sample_rate / target_rate)), settings.buffer_size
    )
    float_audio = stream.read(samples_needed)
    return prepare_capture_chunk(float_audio, target_samples, target_rate)
