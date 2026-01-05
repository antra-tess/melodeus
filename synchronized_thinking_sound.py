#!/usr/bin/env python3
"""
Synchronized thinking sound player that streams audio in real-time.
"""
import asyncio
import threading
import time
from typing import Optional, Callable
import traceback
import numpy as np

from mel_aec_audio import ensure_stream_started, write_playback_pcm, shared_sample_rate, interrupt_playback

class SynchronizedThinkingSoundPlayer:
    """Plays thinking sound with proper timing for echo cancellation."""
    
    def __init__(self):
        self.play_task = None
        self.sample_rate = shared_sample_rate()        
        # Generate thinking sound frames
        self._generate_thinking_frames()
        self.frame_size = 256  # samples
        self.frame_duration = self.frame_size / self.sample_rate  # seconds
        
    def _generate_thinking_frames(self):
        """Generate a single pulse followed by a single silence segment."""
        pulse_duration = 0.15  # seconds
        silence_duration = 0.35  # seconds
        frequency = 440  # Hz

        pulse_samples = int(self.sample_rate * pulse_duration)
        t = np.linspace(0, pulse_duration, pulse_samples, endpoint=False)
        envelope = np.sin(np.pi * t / pulse_duration) ** 2
        carrier = (
            np.sin(2 * np.pi * frequency * t) * 0.1
            + np.sin(2 * np.pi * (frequency / 2) * t) * 0.05
        )
        pulse = (carrier * envelope).astype(np.float32)

        silence = np.zeros(int(self.sample_rate * silence_duration), dtype=np.float32)

        #self.audio_frames = np.concatenate([silence, silence, silence, silence] + [silence, silence]*100)
        self.audio_frames = np.concatenate([silence, pulse]*100)

    async def _play_helper(self):
        try:
            audio_stream = ensure_stream_started()
            while True:
                audio_stream.write(self.audio_frames)
                # wait for buffered audio to drain (polling)
                # leave 0.5 second so we have time to populate it with new audio
                while audio_stream.get_buffered_duration() > 0.5:
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            interrupt_playback()
            raise
        except Exception as e:
            print(f"thinking sound error")
            print(traceback.print_exc())
       
    async def play(self):
        # interrupt (if already running)
        await self.interrupt()

        # start the task (this way it's cancellable and we don't need to spam checks)
        self.play_task = asyncio.create_task(self._play_helper())
    
    async def stop(self):
        await self.interrupt()
    
    async def interrupt(self):
        if self.play_task is not None: # if existing one, stop it
            try:
                self.play_task.cancel()
                await self.play_task # wait for it to cancel
            except asyncio.CancelledError:
                pass # intentional
            except Exception as e:
                print(traceback.print_exc()) 
            finally:
                self.play_task = None
        
    async def cleanup(self):
        await self.interrupt()