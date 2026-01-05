#!/usr/bin/env python3
"""
TitaNet-based voice fingerprinting system for robust speaker recognition.
Uses NVIDIA NeMo's state-of-the-art TitaNet model for speaker embeddings.
"""

import numpy as np
import torch
import librosa
import pickle
import time
import os
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
from mel_aec_audio import shared_sample_rate, _resample
import wave
#import wespeaker


# Try to import NeMo
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
    print("✅ NVIDIA NeMo toolkit available - TitaNet support enabled")
except ImportError as e:
    NEMO_AVAILABLE = False
    print(f"⚠️  NVIDIA NeMo toolkit not available: {e}")
    print("📝 Install with: pip install 'nemo_toolkit[asr]'")

@dataclass
class WordTiming:
    """Word timing information from speech recognition."""
    word: str
    speaker_id: Optional[int]
    start_time: float
    end_time: float
    confidence: float

@dataclass
class VoiceSegment:
    """Extracted audio segment for a specific speaker."""
    audio_data: np.ndarray
    speaker_id: int
    start_time: float
    end_time: float
    words: List[str]
    sample_rate: int = 16000

@dataclass  
class TitaNetFingerprint:
    """TitaNet-based voice fingerprint."""
    speaker_profile_id: str
    embedding: np.ndarray  # 192-dimensional TitaNet embedding
    confidence: float
    duration: float  # Duration of audio used to create this fingerprint
    timestamp: float  # When this fingerprint was created

class TitaNetVoiceFingerprinter:
    """
    Voice fingerprinting system using NVIDIA TitaNet embeddings.
    
    TitaNet produces 192-dimensional speaker embeddings that are much more
    discriminative than traditional MFCC features.
    """
    
    def __init__(self, speakers_config, fingerprints_path: str = "titanet_fingerprints.pkl", debug_save_audio=False, verbose=False):
        """Initialize TitaNet voice fingerprinting system."""
        self.verbose = verbose
        if self.verbose:
            print("🔊 [TITANET] Initializing TitaNet voice fingerprinting system")
        
        if not NEMO_AVAILABLE:
            raise ImportError("NVIDIA NeMo toolkit is required for TitaNet fingerprinting")
        
        self.speakers_config = speakers_config
        self.fingerprints_path = fingerprints_path
        self.sample_rate = 16000
        self.debug_save_audio = debug_save_audio
        self.debug_counter = 0
        self.debug_dir = Path("testouput")
        
        # Configuration
        self.confidence_threshold = speakers_config.recognition.confidence_threshold
        self.learning_mode = speakers_config.recognition.learning_mode
        self.max_speakers = speakers_config.recognition.max_speakers
        self.max_fingerprints_per_speaker = 20  # Limit to prevent memory issues
        
        # Audio buffering for real-time processing
        self.audio_buffer = deque()
        
        # Speaker fingerprints: {speaker_profile_id: [TitaNetFingerprint, ...]}
        self.reference_fingerprints: Dict[str, List[TitaNetFingerprint]] = {}
        
        # Session speaker mapping (Deepgram speaker ID -> profile name)
        self.session_speakers: Dict[int, str] = {}
        self.unknown_speaker_count = 0
        
        # Setup debug directory if needed
        if debug_save_audio:
            print(f"🐛 [TITANET] Debug mode: Will save extracted audio segments")
            self.debug_dir = Path("debug_audio_segments")
            self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize TitaNet model
        self._load_titanet_model()
        #self._load_wespeaker_model()

        # Load existing fingerprints
        self._load_reference_fingerprints()
        
        # Load reference audio files and create initial fingerprints
        self._load_reference_audio_files()
        
        print(f"🔊 [TITANET] System ready with {len(self.reference_fingerprints)} speaker profiles")
    
    def _load_wespeaker_model(self):
        self.wespeaker_model = wespeaker.load_model("english")
        self.device = "cpu"

    def _load_titanet_model(self):
        """Load pre-trained TitaNet model."""
        print("🔊 [TITANET] Loading pre-trained TitaNet model...")
        
        try:
            # Load TitaNet-Large model (best performance)
            model_name = "nvidia/speakerverification_en_titanet_large"
            self.titanet_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_name
            )
            self.titanet_model.eval()
            
            # Move to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.titanet_model = self.titanet_model.to(device)
            self.device = device
            
            print(f"✅ [TITANET] Loaded TitaNet-Large model on {device}")
            
            # Get embedding dimension by testing with dummy input
            try:
                dummy_audio = torch.randn(1, 16000).to(device)  # 1 second of audio
                dummy_length = torch.tensor([16000]).to(device)
                
                with torch.no_grad():
                    _, embeddings = self.titanet_model.get_normalized_embedding(
                        input_signal=dummy_audio,
                        input_signal_length=dummy_length
                    )
                    embedding_dim = embeddings.shape[-1]
                    print(f"🔢 [TITANET] Embedding dimension: {embedding_dim}")
            except Exception as dim_e:
                print(f"⚠️ [TITANET] Could not determine embedding dimension: {dim_e}")
                print("🔢 [TITANET] Assuming standard 192-dimensional embeddings")
            
        except Exception as e:
            print(f"❌ [TITANET] Failed to load TitaNet model: {e}")
            raise RuntimeError(f"Failed to load TitaNet model: {e}")
    
    def clear_audio_buffer(self):
        """Clear the audio buffer. Call this when STT connection resets."""
        self.audio_buffer.clear()
        print("🔄 [TITANET] Audio buffer cleared for new session")

    def add_audio_chunk(self, audio_data: np.ndarray, num_frames_before_this: int, sample_rate: int):
        """Add raw audio data to the circular buffer for later processing."""
        # Debug: Check chunk timing
        # if len(self.audio_buffer) % (self.sample_rate * 10) == 0:  # Every 10 seconds of audio
        #     print(f"🔧 [BUFFER DEBUG] Chunk: {len(audio_data)} samples, {chunk_duration:.3f}s duration, timestamp: {timestamp:.3f}s")
        # can't resample here or it adds a few too many (or too few) and messes up timestamps
        # so we resample later
        # Add samples to circular buffer with correct timestamps
        self.audio_buffer.append((num_frames_before_this, audio_data))
        chunk_len = len(audio_data)
        # 3 min capacity
        while len(audio_data)*len(self.audio_buffer)/float(sample_rate) > 360.0:
            self.audio_buffer.popleft()
    

    def _slice_audio(self, start_frames: int, end_frames: int, sample_rate: int) -> np.ndarray:
        """Return samples between start_time and end_time."""
        pieces = []
        
        for seg_start, samples in self.audio_buffer:
            seg_end = seg_start + len(samples)
            if seg_end <= start_frames:
                continue
            if seg_start >= end_frames:
                break  # later segments start after the window
            #print(f"Found segment {start_frames} {end_frames} {seg_start} {seg_end}")
            clip_start = max(start_frames, seg_start)
            clip_end = min(end_frames, seg_end)
            if clip_start >= clip_end:
                continue

            offset_start = clip_start - seg_start
            offset_end = clip_end - seg_start
            #print(f"Offset start {offset_start} offset end {offset_end}")
            pieces.append(samples[offset_start:offset_end])

        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def process_transcript_words(self, word_timings: List[WordTiming], sample_rate: int):
        """
        Process word timings to extract speaker segments and update fingerprints.
        
        Args:
            word_timings: List of words witch timing and speaker info
        """
        if not word_timings:
            return
        
        # print(f"🔊 [TITANET] Processing {len(word_timings)} words for fingerprinting")
        # Group words by speaker
        
        speaker_audios = defaultdict(list)
        start_times = {}
        end_times = {}
        duration = 0.0
        words = defaultdict(list)
        for word_timing in word_timings:
            audio_pieces = []
            word_audio = self._slice_audio(
                    round(word_timing.start_time*sample_rate),
                    round(word_timing.end_time*sample_rate),
                    sample_rate)
            print(f"Got word audio of len {len(word_audio)}")
            if len(word_audio) > 0:
                speaker_audios[word_timing.speaker_id].append(word_audio)
                start_times[word_timing.speaker_id] = min(start_times.get(word_timing.speaker_id, word_timing.start_time), word_timing.start_time)
                end_times[word_timing.speaker_id] = max(end_times.get(word_timing.speaker_id, word_timing.end_time), word_timing.end_time)
                words[word_timing.speaker_id].append(word_timing.word)
        # Process each speaker segment
        for speaker_id, speaker_audio in speaker_audios.items():
            # Check if segment has sufficient duration (at least 0.5s)
            segment_duration = len(speaker_audio)/float(sample_rate)
            # don't do more than 10 seconds
            speaker_audio_trimmed = np.array(np.concatenate(speaker_audio)[:sample_rate*10])
            # resample now to the sample rate that titanet wants
            resampled_speaker_audio = _resample(speaker_audio_trimmed, sample_rate, self.sample_rate)
            # Create voice segment
            voice_segment = VoiceSegment(
                audio_data=resampled_speaker_audio,
                speaker_id=0,  # Dummy ID
                start_time=start_times[speaker_id],
                end_time=end_times[speaker_id],
                words=words[speaker_id],
                sample_rate=self.sample_rate
            )
            # Create TitaNet embedding
            fingerprint = self._create_titanet_fingerprint(voice_segment)

            if fingerprint is None:
                continue
            
            if self.debug_save_audio or True:
                debug_filename = f"segment_{self.debug_counter:03d}_speaker_{voice_segment.speaker_id}_{len(resampled_speaker_audio)}samples.wav"
                debug_path = self.debug_dir / debug_filename
                self.debug_counter += 1
                
                # Save audio with proper format
                import soundfile as sf
                audio = np.clip(np.array(resampled_speaker_audio), -1.0, 1.0)
                pcm16 = (audio * 32767).astype(np.int16)

                with wave.open(str(debug_path), "wb") as wf:
                    wf.setnchannels(1)                    # audio is mono
                    wf.setsampwidth(2)                    # int16 -> 2 bytes
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm16.tobytes())
                print(f"writing debug audio to {debug_filename}")
            
            # Try to match against known speakers
            matched_id = self._match_speaker(fingerprint)
            if matched_id:
                print(f"✅ [TITANET] Matched speaker {speaker_id} to {matched_id}")
                self.session_speakers[speaker_id] = matched_id
                
                # Add to reference fingerprints for continued learning
                if self.learning_mode:
                    self._add_reference_fingerprint(matched_id, fingerprint)
            else:
                # Unknown speaker
                if self.learning_mode:
                    matched_id = self._handle_unknown_speaker(speaker_id, fingerprint)
                    print(f"👤 [TITANET] Learning new speaker: {matched_id}")
                else:
                    print(f"❓ [TITANET] Unknown speaker {speaker_id} (learning disabled)")
            
            # Unknown speaker - return friendly name
            if matched_id.startswith("unknown_speaker_"):
                count = matched_id.replace("unknown_speaker_", "")
                return f"User {count}"
            else:
                return matched_id
        return "User"
    def _extract_voice_segment(self, words: List[WordTiming], utterance_start: float) -> Optional[VoiceSegment]:
        """Extract audio segment from buffer based on word timings."""
        if not words:
            return None
        
        # Use Deepgram word timings directly (they're already stream-relative)
        first_word = words[0]
        last_word = words[-1]
        
        # Add generous padding to avoid cutting off speech
        padding = 1.0  # 1000ms (1 second) padding
        segment_start = first_word.start_time - padding
        segment_end = last_word.end_time + padding
        segment_duration = segment_end - segment_start
        
        # print(f"🔍 [TITANET] Debug extraction:")
        # print(f"   Stream-relative word timings:")
        # print(f"   First word: '{first_word.word}' at {first_word.start_time:.3f}-{first_word.end_time:.3f}s")
        # print(f"   Last word: '{last_word.word}' at {last_word.start_time:.3f}-{last_word.end_time:.3f}s")
        # print(f"   Target segment: {segment_start:.3f}-{segment_end:.3f}s (duration: {segment_duration:.1f}s)")
        
        # Debug: Check if timing seems reasonable for speech
        words_text = [w.word for w in words]
        estimated_syllables = sum(len(word) // 2 + 1 for word in words_text)  # Rough syllable estimate
        expected_duration = estimated_syllables / 3.5  # ~3.5 syllables/second normal speech
        if segment_duration < expected_duration * 0.5 and self.verbose:
            print(f"⚠️ [TITANET] Suspiciously fast speech: {segment_duration:.1f}s for {estimated_syllables} syllables (expected ~{expected_duration:.1f}s)")
            print(f"⚠️ [TITANET] This suggests a sample rate mismatch - Deepgram may think audio is faster than reality")
        
        if segment_duration < 0.5:  # Too short
            if self.verbose:
                print(f"❌ [TITANET] Segment too short: {segment_duration:.1f}s < 0.5s")
            return None
        if segment_duration > 10.0:  # Too long  
            if self.verbose:
                print(f"❌ [TITANET] Segment too long: {segment_duration:.1f}s > 10.0s")
            return None
        
        # Find buffer indices
        if not self.buffer_timestamps:
            if self.verbose:
                print(f"❌ [TITANET] No buffer timestamps available")
            return None
        
        buffer_times = np.array(list(self.buffer_timestamps))
        if self.verbose:
            print(f"   Buffer time range: {buffer_times[0]:.3f}-{buffer_times[-1]:.3f}s ({len(buffer_times)} samples)")
        
        start_idx = np.searchsorted(buffer_times, segment_start, side='left')
        end_idx = np.searchsorted(buffer_times, segment_end, side='right')
        
        # Debug the timestamp calculations
        if self.verbose:
            print(f"   Segment target: {segment_start:.3f}-{segment_end:.3f}s")
        if start_idx < len(buffer_times):
            if self.verbose:
                print(f"   Buffer start time: {buffer_times[start_idx]:.3f}s (index {start_idx})")
        if end_idx > 0 and end_idx-1 < len(buffer_times):
            if self.verbose:
                print(f"   Buffer end time: {buffer_times[end_idx-1]:.3f}s (index {end_idx-1})")
        expected_samples = int((segment_end - segment_start) * self.sample_rate)
        if self.verbose:
            print(f"   Expected samples: {expected_samples}, Will extract: {end_idx - start_idx}")
        
        if self.verbose:
            print(f"   Buffer indices: {start_idx}-{end_idx} (of {len(self.audio_buffer)} total)")
        
        if start_idx >= len(self.audio_buffer):
            print(f"❌ [TITANET] Start index beyond buffer: {start_idx} >= {len(self.audio_buffer)}")
            return None
        if end_idx <= start_idx:
            print(f"❌ [TITANET] Invalid index range: {end_idx} <= {start_idx}")
            return None
        
        # Extract audio samples
        audio_samples = []
        for i in range(start_idx, min(end_idx, len(self.audio_buffer))):
            audio_samples.append(self.audio_buffer[i])
        
        if len(audio_samples) < self.sample_rate * 0.5:  # Less than 0.5 seconds
            if self.verbose:
                print(f"❌ [TITANET] Too few samples extracted: {len(audio_samples)} < {self.sample_rate * 0.5}")
            return None
        
        audio_data = np.array(audio_samples, dtype=np.float32)
        
        # Check if audio actually contains speech energy
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_max = np.max(np.abs(audio_data))
        
        if self.verbose:
            print(f"   Audio energy: RMS={audio_rms:.6f}, Max={audio_max:.6f}")
        
        if audio_rms < 0.001:  # Very quiet audio, likely silence
            if self.verbose:
                print(f"⚠️ [TITANET] Audio segment appears to be silent (RMS={audio_rms:.6f})")
            # Still proceed, but warn about it
        
        words_text = [w.word for w in words]
        
        if self.verbose:
            print(f"✅ [TITANET] Extracted {len(audio_samples)} samples ({segment_duration:.1f}s) from speaker {words[0].speaker_id}: '{' '.join(words_text)}'")
        
        return VoiceSegment(
            audio_data=audio_data,
            speaker_id=words[0].speaker_id,
            start_time=segment_start,
            end_time=segment_end,
            words=words,  # Pass full WordTiming objects, not just strings
            sample_rate=self.sample_rate
        )
    
    def _create_titanet_fingerprint(self, voice_segment: VoiceSegment) -> Optional[TitaNetFingerprint]:
        """Create TitaNet embedding from audio segment."""
        try:
            audio_data = voice_segment.audio_data
            
            # Ensure minimum length
            #min_samples = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
            #if len(audio_data) < min_samples:
            #    return None
            '''
            # Debug: Save original audio if enabled
            if self.debug_save_audio or True:
                self.debug_counter += 1
                debug_filename = f"segment_{self.debug_counter:03d}_speaker_{voice_segment.speaker_id}_{len(audio_data)}samples.wav"
                debug_path = self.debug_dir / debug_filename
                
                # Save audio with proper format
                import soundfile as sf
                try:
                    sf.write(debug_path, audio_data, self.sample_rate)
                    if self.verbose:
                        print(f"🐛 [TITANET] Saved debug audio: {debug_filename}")
                    
                    # Also save audio characteristics
                    audio_rms = np.sqrt(np.mean(audio_data**2))
                    audio_range = [np.min(audio_data), np.max(audio_data)]
                    if self.verbose:
                        print(f"🐛 [TITANET]   RMS: {audio_rms:.6f}, Range: [{audio_range[0]:.3f}, {audio_range[1]:.3f}]")
                except ImportError:
                    # Fallback to simple numpy save if soundfile not available
                    np.save(debug_path.with_suffix('.npy'), audio_data)
                    print(f"🐛 [TITANET] Saved debug audio (npy): {debug_filename}")
                except Exception as e:
                    print(f"🐛 [TITANET] Failed to save debug audio: {e}")
            '''
            # Normalize audio
            #audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(self.device)
            
            # Get TitaNet embedding
            with torch.no_grad():
                # Get length tensor
                audio_length = torch.tensor([len(audio_data)]).to(self.device)
                
                # Get embedding
                _, embeddings = self.titanet_model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_length
                )
                embedding = embeddings[0].cpu().numpy()
                #embeddings = self.wespeaker_model.extract_embedding_from_pcm(
                #    audio_tensor, sample_rate=self.sample_rate
                #)
                
                # Extract embedding vector
                #embedding = embeddings.cpu().numpy()
            
            duration = len(audio_data) / self.sample_rate
            
            # Calculate confidence from words, handling both WordTiming objects and strings
            word_confidences = []
            for w in voice_segment.words:
                if hasattr(w, 'confidence') and w.confidence is not None:
                    word_confidences.append(w.confidence)
            confidence = np.mean(word_confidences) if word_confidences else 0.95  # Default for reference audio
            
            if self.verbose:
                print(f"🔊 [TITANET] Created embedding (shape: {embedding.shape}, confidence: {confidence:.3f})")
            
            return TitaNetFingerprint(
                speaker_profile_id="",  # Will be set by caller
                embedding=embedding,
                confidence=confidence,
                duration=duration,
                timestamp=time.time()
            )
            
        except Exception as e:            
            print(f"❌ [TITANET] Error creating fingerprint: {e}")
            return None
    
    def _match_speaker(self, fingerprint: TitaNetFingerprint) -> Optional[str]:
        """Match a live fingerprint against stored reference fingerprints."""
        if not self.reference_fingerprints:
            return None
        
        if self.verbose:    
            print(f"🔊 [TITANET] Matching against {len(self.reference_fingerprints)} speaker profiles (threshold: {self.confidence_threshold:.3f})")
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, fingerprints in self.reference_fingerprints.items():
            if not fingerprints:
                continue
            
            # Calculate similarities against all reference fingerprints for this speaker
            similarities = []
            #print(f"similarities for speaker {speaker_id}:")
            for ref_fp in fingerprints:
                similarity = self._cosine_similarity(fingerprint.embedding, ref_fp.embedding)
                similarities.append(similarity)
                #print(f"Similarity {similarity}")
            
            # Use best similarity for this speaker
            speaker_similarity = max(similarities) if similarities else 0.0
            print(f"for speaker {speaker_id} we got closest similarity {speaker_similarity}")
            # Get speaker name
            speaker_name = speaker_id
            for profile_id, profile in self.speakers_config.profiles.items():
                if profile_id == speaker_id:
                    speaker_name = profile.name
                    break
            
            if self.verbose:
                print(f"🔊 [TITANET]   {speaker_name}: best similarity {speaker_similarity:.3f} (from {len(similarities)} references)")
            
            if speaker_similarity > best_similarity:
                best_similarity = speaker_similarity
                best_match = speaker_id
        print(f"Best similarity {best_similarity} with speaker id {speaker_id}")
        
        if best_similarity >= self.confidence_threshold:
            # Get speaker name for display
            matched_name = best_match
            for profile_id, profile in self.speakers_config.profiles.items():
                if profile_id == best_match:
                    matched_name = profile.name
                    break
            
            if self.verbose:
                print(f"🎯 [TITANET] Best match: {matched_name} (similarity: {best_similarity:.3f})")
            return best_match
        else:
            if self.verbose:
                print(f"❌ [TITANET] No match found (best similarity: {best_similarity:.3f} < threshold: {self.confidence_threshold:.3f})")
            return None
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)                      # numerical stability
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-12)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Ensure same shape
        if a.shape != b.shape:
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _add_reference_fingerprint(self, speaker_id: str, fingerprint: TitaNetFingerprint):
        """Add a new reference fingerprint for a speaker."""
        fingerprint.speaker_profile_id = speaker_id
        
        if speaker_id not in self.reference_fingerprints:
            self.reference_fingerprints[speaker_id] = []
        
        # Add fingerprint
        self.reference_fingerprints[speaker_id].append(fingerprint)
        
        # Limit number of fingerprints per speaker
        if len(self.reference_fingerprints[speaker_id]) > self.max_fingerprints_per_speaker:
            # Remove oldest fingerprint
            self.reference_fingerprints[speaker_id].pop(0)
        
        # Save to disk
        self._save_reference_fingerprints()
        
        speaker_name = speaker_id
        for profile_id, profile in self.speakers_config.profiles.items():
            if profile_id == speaker_id:
                speaker_name = profile.name
                break
        
        print(f"💾 [TITANET] Added fingerprint for {speaker_name} (total: {len(self.reference_fingerprints[speaker_id])})")
    
    def _handle_unknown_speaker(self, deepgram_speaker_id: int, fingerprint: TitaNetFingerprint) -> str:
        """Handle a new unknown speaker."""
        self.unknown_speaker_count += 1
        new_speaker_id = f"unknown_speaker_{self.unknown_speaker_count}"
        
        # Create new profile
        self.session_speakers[deepgram_speaker_id] = new_speaker_id
        
        # Add first fingerprint
        self._add_reference_fingerprint(new_speaker_id, fingerprint)
        
        return new_speaker_id
    
    def get_speaker_name(self, deepgram_speaker_id: int) -> Optional[str]:
        """Get the friendly name for a Deepgram speaker ID."""
        speaker_id = self.session_speakers.get(deepgram_speaker_id)
        if not speaker_id:
            return None
        
        # Check if it's a configured speaker
        for profile_id, profile in self.speakers_config.profiles.items():
            if profile_id == speaker_id:
                return profile.name
        
        # Unknown speaker - return friendly name
        if speaker_id.startswith("unknown_speaker_"):
            count = speaker_id.replace("unknown_speaker_", "")
            return f"User {count}"
        
        return speaker_id
    
    def reset_all(self):
        """Reset all fingerprints and speaker mappings. Returns count of cleared profiles."""
        count = len(self.reference_fingerprints)
        self.reference_fingerprints = {}
        self.session_speakers = {}
        self.unknown_speaker_count = 0
        self.audio_buffer.clear()
        # Delete the fingerprint file
        try:
            if Path(self.fingerprints_path).exists():
                Path(self.fingerprints_path).unlink()
        except Exception as e:
            print(f"⚠️ [TITANET] Failed to delete fingerprint file: {e}")
        print(f"🔄 [TITANET] Reset complete - cleared {count} speaker profiles")
        return count

    def _save_reference_fingerprints(self):
        """Save reference fingerprints to disk."""
        try:
            with open(self.fingerprints_path, 'wb') as f:
                pickle.dump(self.reference_fingerprints, f)
        except Exception as e:
            print(f"⚠️ [TITANET] Failed to save fingerprints: {e}")
    
    def _load_reference_fingerprints(self):
        """Load reference fingerprints from disk."""
        try:
            if Path(self.fingerprints_path).exists():
                with open(self.fingerprints_path, 'rb') as f:
                    self.reference_fingerprints = pickle.load(f)
                
                total_fingerprints = sum(len(fps) for fps in self.reference_fingerprints.values())
                print(f"📁 [TITANET] Loaded {total_fingerprints} fingerprints for {len(self.reference_fingerprints)} speakers from disk")

                # Update unknown_speaker_count to avoid ID collisions after restart
                for speaker_id in self.reference_fingerprints.keys():
                    if speaker_id.startswith("unknown_speaker_"):
                        try:
                            num = int(speaker_id.replace("unknown_speaker_", ""))
                            self.unknown_speaker_count = max(self.unknown_speaker_count, num)
                        except ValueError:
                            pass
                if self.unknown_speaker_count > 0:
                    print(f"🔢 [TITANET] Resuming unknown speaker count from {self.unknown_speaker_count}")
        except Exception as e:
            print(f"⚠️ [TITANET] Failed to load existing fingerprints: {e}")
            self.reference_fingerprints = {}
    
    def _load_reference_audio_files(self):
        """Load reference audio files and create initial fingerprints, with caching."""
        for profile_id, profile in self.speakers_config.profiles.items():
            if not profile.reference_audio:
                continue
            
            audio_path = Path(profile.reference_audio)
            if not audio_path.exists():
                print(f"⚠️ [TITANET] Reference audio not found: {audio_path}")
                continue
            
            # Check for cached embeddings
            cache_path = audio_path.with_suffix('.titanet_cache')
            use_cache = False
            
            if cache_path.exists():
                # Check if cache is newer than audio file
                audio_mtime = os.path.getmtime(audio_path)
                cache_mtime = os.path.getmtime(cache_path)
                
                if cache_mtime > audio_mtime:
                    try:
                        print(f"📦 [TITANET] Loading cached embeddings for {profile.name}")
                        with open(cache_path, 'rb') as f:
                            cached_fingerprints = pickle.load(f)
                        
                        # Add cached fingerprints
                        for fingerprint in cached_fingerprints:
                            self._add_reference_fingerprint(profile_id, fingerprint)
                        
                        print(f"✅ [TITANET] Loaded {len(cached_fingerprints)} cached fingerprints for {profile.name}")
                        use_cache = True
                        
                    except Exception as e:
                        print(f"⚠️ [TITANET] Failed to load cache for {profile.name}: {e}")
                        print(f"🔄 [TITANET] Will regenerate embeddings...")
            
            if not use_cache:
                print(f"🔊 [TITANET] Loading reference audio for {profile.name}: {audio_path}")
                
                try:
                    # Load audio file
                    audio_data, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                    duration = len(audio_data) / self.sample_rate
                    
                    print(f"🔊 [TITANET] Loaded {duration:.1f}s of audio at {sr}Hz")
                    
                    # Create fingerprints from chunks
                    chunk_duration = 4.0  # 4 second chunks
                    chunk_samples = int(chunk_duration * self.sample_rate)
                    overlap_samples = int(1.0 * self.sample_rate)  # 1 second overlap
                    
                    chunks_created = 0
                    new_fingerprints = []
                    
                    for start in range(0, len(audio_data) - chunk_samples, chunk_samples - overlap_samples):
                        end = start + chunk_samples
                        chunk = audio_data[start:end]
                        
                        if len(chunk) < chunk_samples:
                            continue
                        
                        # Create voice segment
                        voice_segment = VoiceSegment(
                            audio_data=chunk,
                            speaker_id=0,  # Dummy ID
                            start_time=start / self.sample_rate,
                            end_time=end / self.sample_rate,
                            words=["reference"],
                            sample_rate=self.sample_rate
                        )
                        
                        # Create fingerprint
                        fingerprint = self._create_titanet_fingerprint(voice_segment)
                        if fingerprint:
                            chunks_created += 1
                            new_fingerprints.append(fingerprint)
                            self._add_reference_fingerprint(profile_id, fingerprint)
                    
                    print(f"✅ [TITANET] Created {chunks_created} reference fingerprints for {profile.name}")
                    
                    # Cache the new fingerprints
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(new_fingerprints, f)
                        print(f"💾 [TITANET] Cached embeddings to {cache_path}")
                    except Exception as e:
                        print(f"⚠️ [TITANET] Failed to cache embeddings: {e}")
                    
                except Exception as e:
                    print(f"❌ [TITANET] Error loading reference audio for {profile.name}: {e}")

def create_word_timing_from_deepgram_word(word) -> WordTiming:
    """Convert Deepgram word object to WordTiming."""
    return WordTiming(
        word=getattr(word, 'word', ''),
        speaker_id=getattr(word, 'speaker', None),
        start_time=getattr(word, 'start', 0.0),
        end_time=getattr(word, 'end', 0.0),
        confidence=getattr(word, 'confidence', 0.0),
        utterance_start=0.0  # Will be set by caller
    )