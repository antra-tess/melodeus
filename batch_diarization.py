#!/usr/bin/env python3
"""
Batch Diarization Module using ElevenLabs Scribe v2 (non-realtime)

This module handles:
1. Audio buffering with frame timestamps
2. Batch transcription with diarization via ElevenLabs
3. Transcript matching between realtime and batch results
4. Audio rechunking based on proper turn boundaries
5. TitaNet recalculation on clean speaker segments
"""

import asyncio
import aiohttp
import base64
import io
import json
import time
import wave
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import difflib


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    audio: np.ndarray  # Float32 audio samples
    frame_start: int   # Starting frame number
    frame_end: int     # Ending frame number
    timestamp: float   # Wall clock time when captured


@dataclass
class RealtimeTranscript:
    """A transcript from the realtime STT."""
    text: str
    timestamp: float
    frame_start: int
    frame_end: int
    speaker_guess: Optional[str] = None  # TitaNet's initial guess
    message_id: Optional[str] = None


@dataclass 
class BatchTurn:
    """A speaker turn from batch diarization."""
    speaker_id: str       # e.g., "speaker_0", "speaker_1"
    text: str
    start_time: float     # Seconds from start of audio
    end_time: float       # Seconds from start of audio
    words: List[Dict] = field(default_factory=list)  # Word-level timestamps if available


@dataclass
class DiarizedSegment:
    """Final diarized segment with matched audio."""
    speaker_id: str
    text: str
    audio: np.ndarray
    frame_start: int
    frame_end: int
    titanet_embedding: Optional[np.ndarray] = None
    matched_realtime_ids: List[str] = field(default_factory=list)


class BatchDiarizer:
    """
    Handles batch diarization and transcript matching.
    
    Flow:
    1. Collect audio chunks and realtime transcripts
    2. Periodically send audio to batch API for diarization
    3. Match batch results to realtime transcripts
    4. Rechunk audio based on proper turn boundaries
    5. Recalculate TitaNet on clean segments
    """
    
    # ElevenLabs batch STT endpoint
    BATCH_API_URL = "https://api.elevenlabs.io/v1/speech-to-text/convert"
    
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        buffer_duration_sec: float = 15.0,  # How much audio to buffer before batch
        min_buffer_sec: float = 5.0,  # Minimum audio before triggering batch
        voice_fingerprinter=None,  # Optional TitaNet instance
        on_diarization_complete: Optional[Callable] = None,  # Callback when batch completes
    ):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.buffer_duration_sec = buffer_duration_sec
        self.min_buffer_sec = min_buffer_sec
        self.voice_fingerprinter = voice_fingerprinter
        self.on_diarization_complete = on_diarization_complete
        
        # Audio buffer (rolling window)
        self.audio_buffer: deque[AudioChunk] = deque()
        self.total_frames_buffered = 0
        
        # Realtime transcripts waiting for diarization
        self.pending_transcripts: List[RealtimeTranscript] = []
        
        # Results
        self.diarized_segments: List[DiarizedSegment] = []
        
        # State
        self.is_processing = False
        self.last_batch_time = 0
        self._batch_task: Optional[asyncio.Task] = None
    
    def add_audio_chunk(self, audio: np.ndarray, frame_start: int, frame_end: int):
        """Add an audio chunk to the buffer."""
        chunk = AudioChunk(
            audio=audio.copy(),
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp=time.time()
        )
        self.audio_buffer.append(chunk)
        self.total_frames_buffered += len(audio)
        
        # Trim old audio if buffer is too long
        max_frames = int(self.buffer_duration_sec * self.sample_rate * 2)  # 2x buffer for overlap
        while self.total_frames_buffered > max_frames and len(self.audio_buffer) > 1:
            old_chunk = self.audio_buffer.popleft()
            self.total_frames_buffered -= len(old_chunk.audio)
    
    def add_realtime_transcript(self, transcript: RealtimeTranscript):
        """Add a realtime transcript to be matched with batch results."""
        self.pending_transcripts.append(transcript)
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self.total_frames_buffered / self.sample_rate
    
    def should_process_batch(self) -> bool:
        """Check if we should trigger batch processing."""
        if self.is_processing:
            return False
        
        buffer_sec = self.get_buffer_duration()
        
        # Process if buffer is full enough
        if buffer_sec >= self.buffer_duration_sec:
            return True
        
        # Or if there's been silence for a while and we have minimum audio
        if buffer_sec >= self.min_buffer_sec:
            time_since_last = time.time() - self.last_batch_time
            if time_since_last > 3.0:  # 3 seconds of potential silence
                return True
        
        return False
    
    async def trigger_batch_processing(self):
        """Trigger async batch processing."""
        if self.is_processing or not self.audio_buffer:
            return
        
        self.is_processing = True
        self._batch_task = asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Process the current audio buffer through batch API."""
        try:
            # Concatenate audio chunks
            audio_chunks = list(self.audio_buffer)
            if not audio_chunks:
                return
            
            full_audio = np.concatenate([c.audio for c in audio_chunks])
            frame_start = audio_chunks[0].frame_start
            frame_end = audio_chunks[-1].frame_end
            
            print(f"🔄 Batch diarization: {len(full_audio)/self.sample_rate:.1f}s audio, "
                  f"frames {frame_start}-{frame_end}")
            
            # Convert to WAV bytes
            wav_bytes = self._audio_to_wav(full_audio)
            
            # Call batch API
            batch_turns = await self._call_batch_api(wav_bytes)
            
            if not batch_turns:
                print("⚠️  Batch diarization returned no results")
                return
            
            print(f"📝 Batch returned {len(batch_turns)} turns")
            for turn in batch_turns:
                print(f"   {turn.speaker_id}: \"{turn.text[:50]}...\" "
                      f"({turn.start_time:.2f}s - {turn.end_time:.2f}s)")
            
            # Match transcripts and rechunk audio
            segments = self._match_and_rechunk(
                batch_turns=batch_turns,
                audio=full_audio,
                frame_offset=frame_start,
                pending_transcripts=self.pending_transcripts.copy()
            )
            
            # Recalculate TitaNet embeddings on clean segments
            if self.voice_fingerprinter:
                for segment in segments:
                    try:
                        embedding = self.voice_fingerprinter.get_embedding(
                            segment.audio, 
                            self.sample_rate
                        )
                        segment.titanet_embedding = embedding
                    except Exception as e:
                        print(f"⚠️  TitaNet embedding failed: {e}")
            
            # Store results
            self.diarized_segments.extend(segments)
            
            # Clear processed transcripts
            matched_ids = set()
            for seg in segments:
                matched_ids.update(seg.matched_realtime_ids)
            self.pending_transcripts = [
                t for t in self.pending_transcripts 
                if t.message_id not in matched_ids
            ]
            
            # Callback
            if self.on_diarization_complete:
                try:
                    await self.on_diarization_complete(segments)
                except Exception as e:
                    print(f"⚠️  Diarization callback error: {e}")
            
            self.last_batch_time = time.time()
            
        except Exception as e:
            print(f"❌ Batch diarization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
    
    def _audio_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    async def _call_batch_api(self, wav_bytes: bytes) -> List[BatchTurn]:
        """Call ElevenLabs batch STT API with diarization."""
        headers = {
            "xi-api-key": self.api_key,
        }
        
        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field(
            'file',
            wav_bytes,
            filename='audio.wav',
            content_type='audio/wav'
        )
        data.add_field('model_id', 'scribe_v2')
        data.add_field('diarize', 'true')
        data.add_field('timestamps_granularity', 'word')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.BATCH_API_URL,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Batch API error {response.status}: {error_text}")
                    return []
                
                result = await response.json()
        
        # Parse response into BatchTurn objects
        turns = []
        
        # ElevenLabs returns diarized segments
        if 'utterances' in result:
            for utt in result['utterances']:
                turn = BatchTurn(
                    speaker_id=utt.get('speaker', 'unknown'),
                    text=utt.get('text', ''),
                    start_time=utt.get('start', 0),
                    end_time=utt.get('end', 0),
                    words=utt.get('words', [])
                )
                turns.append(turn)
        elif 'text' in result:
            # Fallback if no diarization
            turn = BatchTurn(
                speaker_id='speaker_0',
                text=result['text'],
                start_time=0,
                end_time=len(wav_bytes) / (self.sample_rate * 2),  # Approximate
                words=result.get('words', [])
            )
            turns.append(turn)
        
        return turns
    
    def _match_and_rechunk(
        self,
        batch_turns: List[BatchTurn],
        audio: np.ndarray,
        frame_offset: int,
        pending_transcripts: List[RealtimeTranscript]
    ) -> List[DiarizedSegment]:
        """
        Match realtime transcripts to batch turns and rechunk audio.
        
        Uses fuzzy text matching to align transcripts, then extracts
        the correct audio segments based on batch timing.
        """
        segments = []
        
        for turn in batch_turns:
            # Calculate frame range for this turn
            start_frame = int(turn.start_time * self.sample_rate)
            end_frame = int(turn.end_time * self.sample_rate)
            
            # Clamp to audio bounds
            start_frame = max(0, min(start_frame, len(audio)))
            end_frame = max(start_frame, min(end_frame, len(audio)))
            
            # Extract audio segment
            turn_audio = audio[start_frame:end_frame]
            
            if len(turn_audio) < self.sample_rate * 0.1:  # Skip very short segments
                continue
            
            # Find matching realtime transcripts
            matched_ids = []
            turn_text_lower = turn.text.lower()
            
            for rt in pending_transcripts:
                # Fuzzy match - check if realtime text is contained in or similar to batch text
                rt_text_lower = rt.text.lower()
                
                # Check for substring match
                if rt_text_lower in turn_text_lower or turn_text_lower in rt_text_lower:
                    if rt.message_id:
                        matched_ids.append(rt.message_id)
                    continue
                
                # Check similarity ratio
                similarity = difflib.SequenceMatcher(None, rt_text_lower, turn_text_lower).ratio()
                if similarity > 0.6:  # 60% similar
                    if rt.message_id:
                        matched_ids.append(rt.message_id)
            
            segment = DiarizedSegment(
                speaker_id=turn.speaker_id,
                text=turn.text,
                audio=turn_audio,
                frame_start=frame_offset + start_frame,
                frame_end=frame_offset + end_frame,
                matched_realtime_ids=matched_ids
            )
            segments.append(segment)
        
        return segments
    
    def get_speaker_audio_segments(self, speaker_id: str) -> List[np.ndarray]:
        """Get all audio segments for a specific speaker."""
        return [
            seg.audio for seg in self.diarized_segments 
            if seg.speaker_id == speaker_id
        ]
    
    def get_speaker_embeddings(self, speaker_id: str) -> List[np.ndarray]:
        """Get all TitaNet embeddings for a specific speaker."""
        return [
            seg.titanet_embedding for seg in self.diarized_segments 
            if seg.speaker_id == speaker_id and seg.titanet_embedding is not None
        ]


class DiarizationManager:
    """
    High-level manager that integrates batch diarization with the STT pipeline.
    
    Handles:
    - Automatic batch triggering
    - Speaker profile updates
    - Transcript corrections
    """
    
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        voice_fingerprinter=None,
        speakers_config=None,
    ):
        self.batch_diarizer = BatchDiarizer(
            api_key=api_key,
            sample_rate=sample_rate,
            voice_fingerprinter=voice_fingerprinter,
            on_diarization_complete=self._on_diarization_complete
        )
        self.speakers_config = speakers_config
        self.voice_fingerprinter = voice_fingerprinter
        
        # Map batch speaker IDs to known speaker names
        self.speaker_mapping: Dict[str, str] = {}
        
        # Callbacks for transcript updates
        self.on_transcript_update: Optional[Callable] = None
    
    def feed_audio(self, audio: np.ndarray, frame_start: int, frame_end: int):
        """Feed audio to the batch diarizer."""
        self.batch_diarizer.add_audio_chunk(audio, frame_start, frame_end)
    
    def feed_transcript(
        self, 
        text: str, 
        frame_start: int, 
        frame_end: int,
        speaker_guess: Optional[str] = None,
        message_id: Optional[str] = None
    ):
        """Feed a realtime transcript for later matching."""
        transcript = RealtimeTranscript(
            text=text,
            timestamp=time.time(),
            frame_start=frame_start,
            frame_end=frame_end,
            speaker_guess=speaker_guess,
            message_id=message_id
        )
        self.batch_diarizer.add_realtime_transcript(transcript)
    
    async def check_and_process(self):
        """Check if we should process batch and trigger if needed."""
        if self.batch_diarizer.should_process_batch():
            await self.batch_diarizer.trigger_batch_processing()
    
    async def _on_diarization_complete(self, segments: List[DiarizedSegment]):
        """Called when batch diarization completes."""
        print(f"✅ Diarization complete: {len(segments)} segments")
        
        # Update speaker mapping using TitaNet
        for segment in segments:
            if segment.titanet_embedding is not None and self.voice_fingerprinter:
                # Try to match to known speaker
                known_speaker = self.voice_fingerprinter.identify_speaker(
                    segment.titanet_embedding
                )
                if known_speaker:
                    self.speaker_mapping[segment.speaker_id] = known_speaker
                    print(f"   {segment.speaker_id} → {known_speaker}")
        
        # Update speaker profiles with clean audio
        if self.voice_fingerprinter:
            for segment in segments:
                speaker_name = self.speaker_mapping.get(segment.speaker_id)
                if speaker_name and segment.titanet_embedding is not None:
                    # Add embedding to speaker profile
                    self.voice_fingerprinter.add_speaker_embedding(
                        speaker_name,
                        segment.titanet_embedding
                    )
        
        # Notify about transcript updates
        if self.on_transcript_update:
            for segment in segments:
                speaker_name = self.speaker_mapping.get(
                    segment.speaker_id, 
                    segment.speaker_id
                )
                await self.on_transcript_update(
                    message_ids=segment.matched_realtime_ids,
                    corrected_speaker=speaker_name,
                    text=segment.text
                )


if __name__ == "__main__":
    # Test the module
    import asyncio
    
    async def test():
        print("Testing BatchDiarizer...")
        
        # Create a test instance (won't actually call API without valid key)
        diarizer = BatchDiarizer(
            api_key="test_key",
            sample_rate=16000,
            buffer_duration_sec=5.0
        )
        
        # Add some fake audio
        for i in range(10):
            audio = np.random.randn(1600).astype(np.float32) * 0.1  # 0.1s chunks
            diarizer.add_audio_chunk(
                audio=audio,
                frame_start=i * 1600,
                frame_end=(i + 1) * 1600
            )
        
        print(f"Buffer duration: {diarizer.get_buffer_duration():.2f}s")
        print(f"Should process: {diarizer.should_process_batch()}")
        
        # Add a transcript
        diarizer.add_realtime_transcript(RealtimeTranscript(
            text="Hello, this is a test",
            timestamp=time.time(),
            frame_start=0,
            frame_end=8000,
            speaker_guess="User",
            message_id="test-123"
        ))
        
        print(f"Pending transcripts: {len(diarizer.pending_transcripts)}")
        print("✅ Test complete")
    
    asyncio.run(test())
