"""Thinking sound player for voice conversation system."""
import asyncio
import threading
import time
from typing import Callable, List, Optional

import numpy as np

try:
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
    )
except ImportError:  # pragma: no cover - fallback when running from repo root
    import os
    import sys

    CURRENT_DIR = os.path.dirname(__file__)
    if CURRENT_DIR and CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
    )


class ThinkingSoundPlayer:
    """Plays a looping thinking sound until stopped using the shared mel-aec stream."""

    def __init__(
        self,
        chunk_size: int = 256,
    ):
        self.sample_rate = sample_rate or shared_sample_rate()
        self.chunk_size = max(1, int(chunk_size))
        self._chunk_bytes = self.chunk_size * 2
        self._chunk_duration = (
            self.chunk_size / float(self.sample_rate) if self.sample_rate else 0.0
        )

        # Attributes maintained for backwards compatibility with previous PyAudio usage
        self.stream = None

        self.is_playing = False
        self.play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_generation = 0
        self._lock = threading.Lock()

        # Echo cancellation callback
        self.echo_cancellation_callback: Optional[Callable[[bytes], None]] = None

        # Pre-compute thinking sound chunks
        self.thinking_sound = self._generate_thinking_sound()
        self._thinking_sound_bytes = self.thinking_sound.tobytes()
        self._thinking_chunks = self._prepare_playback_chunks(self._thinking_sound_bytes)

    def set_echo_cancellation_callback(self, callback: Callable[[bytes], None]):
        """Set callback for echo cancellation."""
        self.echo_cancellation_callback = callback

    def _prepare_playback_chunks(self, audio_bytes: bytes) -> List[bytes]:
        """Split thinking sound into fixed-size PCM chunks for playback."""
        if self._chunk_bytes <= 0:
            return [audio_bytes] if audio_bytes else [b""]

        if not audio_bytes:
            return [b"\x00" * self._chunk_bytes]

        chunks: List[bytes] = []
        for idx in range(0, len(audio_bytes), self._chunk_bytes):
            chunk = audio_bytes[idx : idx + self._chunk_bytes]
            if len(chunk) < self._chunk_bytes:
                chunk = chunk + b"\x00" * (self._chunk_bytes - len(chunk))
            chunks.append(chunk)
        return chunks

    def _generate_thinking_sound(self) -> np.ndarray:
        """Generate a soft, rhythmic thinking sound."""
        duration = 0.15  # 150ms per pulse
        silence_duration = 0.35  # 350ms silence between pulses

        # Create time arrays
        t_sound = np.linspace(
            0.0, duration, int(self.sample_rate * duration), endpoint=False
        )
        t_silence = np.zeros(int(self.sample_rate * silence_duration), dtype=np.float32)

        # Generate a soft sine wave with envelope
        frequency = 440  # A4 note
        envelope = np.sin(np.pi * t_sound / duration) ** 2  # Smooth fade in/out
        sound = envelope * np.sin(2 * np.pi * frequency * t_sound) * 0.1  # Low volume

        # Add a subtle lower harmonic for warmth
        sound += envelope * np.sin(2 * np.pi * frequency / 2 * t_sound) * 0.05

        # Combine sound and silence for one complete pulse
        pulse = np.concatenate([sound, t_silence])

        # Convert to int16 PCM
        return (pulse * 32767).astype(np.int16)

    def _play_loop(self) -> None:
        """Worker thread that streams the thinking sound through mel-aec."""
        try:
            ensure_stream_started()
            chunk_duration = self._chunk_duration

            while not self._stop_event.is_set():
                next_frame_time = time.perf_counter()

                for chunk in self._thinking_chunks:
                    if self._stop_event.is_set():
                        break

                    try:
                        write_playback_pcm(chunk, self.sample_rate)
                    except Exception as playback_error:
                        print(f"Error playing thinking sound: {playback_error}")
                        return
                    
                    if chunk_duration > 0.0:
                        next_frame_time += chunk_duration
                        sleep_for = next_frame_time - time.perf_counter()
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        else:
                            next_frame_time = time.perf_counter()
        except Exception as e:
            print(f"Error in thinking sound setup: {e}")
        finally:
            with self._lock:
                self.is_playing = False

    async def start(self, generation: Optional[int] = None):
        """Start playing the thinking sound for a specific generation."""
        with self._lock:
            if generation is not None:
                self._current_generation = generation

            # Clean up any completed thread
            if self.play_thread and not self.play_thread.is_alive():
                self.play_thread = None
                self.is_playing = False

            if self.play_thread and self.play_thread.is_alive():
                print(
                    f"🎵 Thinking sound already playing, updated to generation "
                    f"{self._current_generation}"
                )
                return

            self.is_playing = True
            self._stop_event.clear()

            self.play_thread = threading.Thread(
                target=self._play_loop,
                daemon=True,
                name="ThinkingSound",
            )
            self.play_thread.start()
            print(f"🎵 Started thinking sound for generation {self._current_generation}")

    async def stop(self, generation: Optional[int] = None):
        """Stop playing the thinking sound only if it matches the generation."""
        with self._lock:
            if generation is not None and generation != self._current_generation:
                print(
                    f"🎵 Ignoring stop for old generation {generation}, "
                    f"current is {self._current_generation}"
                )
                return

            if not self.is_playing:
                return

            self.is_playing = False
            self._stop_event.set()
            play_thread = self.play_thread

        if play_thread and play_thread.is_alive():
            await asyncio.get_event_loop().run_in_executor(
                None, play_thread.join
            )

        with self._lock:
            if self.play_thread is play_thread:
                self.play_thread = None

        print(f"🔇 Stopped thinking sound for generation {generation or self._current_generation}")

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.is_playing:
                self.is_playing = False
                self._stop_event.set()

            play_thread = self.play_thread
            if play_thread and play_thread.is_alive():
                play_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Error during thinking sound cleanup: {e}")
        finally:
            self.play_thread = None
            self.stream = None


async def test_thinking_sound():
    """Test the thinking sound player."""
    player = ThinkingSoundPlayer()

    try:
        print("Starting thinking sound...")
        await player.start()

        # Play for 3 seconds
        await asyncio.sleep(3)

        print("Stopping thinking sound...")
        await player.stop()

    finally:
        player.cleanup()


if __name__ == "__main__":
    asyncio.run(test_thinking_sound())
