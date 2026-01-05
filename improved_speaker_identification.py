#!/usr/bin/env python3
"""
Improved Speaker Identification System with False Positive Prevention
Addresses issues with unknown speakers being falsely detected as known speakers
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SpeakerProfile:
    """Speaker profile containing voice characteristics and metadata."""
    name: str
    speaker_id: str
    embeddings: List[List[float]]  # Multiple voice embeddings
    created_at: str
    last_seen: str
    session_count: int
    confidence_scores: List[float]
    
    def add_embedding(self, embedding: List[float], confidence: float = 1.0):
        """Add a new voice embedding to this speaker's profile."""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.last_seen = datetime.now().isoformat()
        
        # Keep only the best embeddings (limit to 10)
        if len(self.embeddings) > 10:
            # Remove the embedding with lowest confidence
            min_idx = self.confidence_scores.index(min(self.confidence_scores))
            self.embeddings.pop(min_idx)
            self.confidence_scores.pop(min_idx)
    
    def get_average_embedding(self) -> np.ndarray:
        """Get the average embedding for this speaker."""
        if not self.embeddings:
            return np.array([])
        return np.mean(self.embeddings, axis=0)

class SpeakerIdentifier:
    """Improved system for identifying speakers with false positive prevention."""
    
    def __init__(self, profiles_dir: str = "speaker_profiles", profiles_file: Optional[str] = None):
        """Initialize the improved speaker identification system."""
        if profiles_file:
            self.profiles_dir = Path(profiles_file).parent
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            self._profiles_file_override = Path(profiles_file)
        else:
            self.profiles_dir = Path(profiles_dir)
            self.profiles_dir.mkdir(exist_ok=True)
            self._profiles_file_override = None
        
        self.known_speakers: Dict[str, SpeakerProfile] = {}
        self.session_speakers: Dict[int, str] = {}  # Maps session speaker_id to known speaker_id
        self.unknown_count = 0
        
        # Improved similarity thresholds
        self.identification_threshold = 0.3    # Much higher threshold to prevent false positives
        self.registration_threshold = 0.4      # Higher threshold for adding to profiles
        self.margin_threshold = 0.01           # Best match must be this much better than second best
        self.min_confidence_diff = 0.05        # Minimum difference between best and second-best
        
        # Debug mode for monitoring similarities
        self.debug_mode = True
        
        self.load_speaker_profiles()
    
    def load_speaker_profiles(self):
        """Load existing speaker profiles from disk."""
        try:
            profiles_file = self._profiles_file_override or (self.profiles_dir / "speaker_profiles.json")
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for speaker_id, profile_data in profiles_data.items():
                    self.known_speakers[speaker_id] = SpeakerProfile(**profile_data)
                
                print(f"📚 Loaded {len(self.known_speakers)} known speaker profiles")
                if self.debug_mode:
                    print(f"🔧 Debug mode: High threshold = {self.identification_threshold}")
            else:
                print("📝 No existing speaker profiles found - starting fresh")
        except Exception as e:
            print(f"⚠️  Error loading speaker profiles: {e}")
    
    def save_speaker_profiles(self):
        """Save speaker profiles to disk."""
        try:
            profiles_file = self._profiles_file_override or (self.profiles_dir / "speaker_profiles.json")
            profiles_data = {}
            
            for speaker_id, profile in self.known_speakers.items():
                profiles_data[speaker_id] = asdict(profile)
            
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
            print(f"💾 Saved {len(self.known_speakers)} speaker profiles")
        except Exception as e:
            print(f"⚠️  Error saving speaker profiles: {e}")
    
    def extract_speaker_embedding(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract improved speaker embedding from audio segment.
        Uses more robust features to reduce false positives.
        """
        if len(audio_segment) == 0:
            return np.array([])
        
        # Compute FFT
        fft = np.fft.fft(audio_segment)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(magnitude), 1/sample_rate)
        
        features = []
        
        # More robust spectral features
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]
        freqs_half = freqs[:half_len]
        
        # Spectral centroid (weighted frequency mean)
        if np.sum(magnitude_half) > 0:
            spectral_centroid = np.sum(freqs_half * magnitude_half) / np.sum(magnitude_half)
        else:
            spectral_centroid = 0
        features.append(spectral_centroid)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(magnitude_half)
        if cumsum[-1] > 0:
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            spectral_rolloff = freqs_half[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            spectral_rolloff = 0
        features.append(spectral_rolloff)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
        zcr = zero_crossings / len(audio_segment) if len(audio_segment) > 0 else 0
        features.append(zcr)
        
        # Spectral bandwidth
        if np.sum(magnitude_half) > 0:
            bandwidth = np.sqrt(np.sum(((freqs_half - spectral_centroid) ** 2) * magnitude_half) / np.sum(magnitude_half))
        else:
            bandwidth = 0
        features.append(bandwidth)
        
        # Spectral skewness (measure of asymmetry)
        if np.sum(magnitude_half) > 0 and bandwidth > 0:
            skewness = np.sum(((freqs_half - spectral_centroid) ** 3) * magnitude_half) / (np.sum(magnitude_half) * (bandwidth ** 3))
        else:
            skewness = 0
        features.append(skewness)
        
        # Energy in different frequency bands
        bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]  # Hz
        for low, high in bands:
            band_mask = (freqs_half >= low) & (freqs_half <= high)
            band_energy = np.sum(magnitude_half[band_mask])
            features.append(band_energy)
        
        # MFCC-like features (improved)
        mel_filters = 13
        for i in range(mel_filters):
            start_idx = i * len(magnitude_half) // mel_filters
            end_idx = (i + 1) * len(magnitude_half) // mel_filters
            mel_energy = np.sum(magnitude_half[start_idx:end_idx])
            # Add logarithm for better discrimination
            features.append(np.log(mel_energy + 1e-8))
        
        # Statistical features of the raw audio
        features.extend([
            np.mean(audio_segment),
            np.std(audio_segment),
            np.max(audio_segment),
            np.min(audio_segment),
            np.median(audio_segment)
        ])
        
        return np.array(features)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate improved similarity between two speaker embeddings.
        Uses more conservative scoring to prevent false positives.
        """
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity (range -1 to 1)
        cosine_sim = np.dot(embedding1_norm, embedding2_norm)
        
        # More conservative conversion - only positive similarities count
        # This prevents random noise from getting high scores
        if cosine_sim <= 0:
            return 0.0
        
        # Scale positive similarities more conservatively
        # cosine_sim of 1.0 -> 1.0, cosine_sim of 0.5 -> 0.25
        similarity = cosine_sim ** 2  # Square to make it more conservative
        
        return similarity
    
    def identify_speaker(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """
        Identify a speaker with detailed matching information.
        Returns (speaker_name, confidence, debug_info) or (None, 0, debug_info) if unknown.
        """
        if len(self.known_speakers) == 0:
            return None, 0.0, {"reason": "no_known_speakers"}
        
        similarities = []
        speaker_names = []
        
        for speaker_id, profile in self.known_speakers.items():
            avg_embedding = profile.get_average_embedding()
            if len(avg_embedding) == 0:
                continue
                
            similarity = self.calculate_similarity(embedding, avg_embedding)
            print(f"Simliarity {speaker_id} {similarity}")
            similarities.append(similarity)
            speaker_names.append((speaker_id, profile.name))
        
        if not similarities:
            return None, 0.0, {"reason": "no_valid_embeddings"}
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        best_similarity = similarities[sorted_indices[0]]
        best_speaker_id, best_speaker_name = speaker_names[sorted_indices[0]]
        
        # Get second best for margin calculation
        second_best_similarity = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
        margin = best_similarity - second_best_similarity
        
        debug_info = {
            "best_similarity": best_similarity,
            "second_best_similarity": second_best_similarity,
            "margin": margin,
            "threshold": self.identification_threshold,
            "margin_threshold": self.margin_threshold,
            "all_similarities": dict(zip([name for _, name in speaker_names], similarities))
        }
        
        if self.debug_mode:
            print(f"🔍 Speaker matching debug:")
            print(f"   Best: {best_speaker_name} ({best_similarity:.3f})")
            print(f"   Second: {speaker_names[sorted_indices[1]][1] if len(similarities) > 1 else 'None'} ({second_best_similarity:.3f})")
            print(f"   Margin: {margin:.3f} (need ≥{self.margin_threshold:.3f})")
            print(f"   Threshold: {best_similarity:.3f} ≥ {self.identification_threshold:.3f}? {best_similarity >= self.identification_threshold}")
        
        # Check both similarity threshold and margin requirement
        if (best_similarity >= self.identification_threshold and 
            margin >= self.margin_threshold):
            debug_info["reason"] = "identified"
            return best_speaker_name, best_similarity, debug_info
        elif best_similarity >= self.identification_threshold:
            debug_info["reason"] = "insufficient_margin"
            print(f"⚠️  High similarity ({best_similarity:.3f}) but insufficient margin ({margin:.3f})")
        else:
            debug_info["reason"] = "low_similarity"
            print(f"❌ Low similarity ({best_similarity:.3f}) - treating as unknown speaker")
        
        return None, best_similarity, debug_info
    
    def register_speaker(self, name: str, embedding: np.ndarray, force: bool = False) -> str:
        """Register a new speaker with their voice embedding."""
        speaker_id = f"speaker_{len(self.known_speakers):03d}_{int(time.time())}"
        
        profile = SpeakerProfile(
            name=name,
            speaker_id=speaker_id,
            embeddings=[embedding.tolist()],
            created_at=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            session_count=1,
            confidence_scores=[1.0]
        )
        
        self.known_speakers[speaker_id] = profile
        self.save_speaker_profiles()
        
        print(f"✅ Registered new speaker: {name} (ID: {speaker_id})")
        return speaker_id
    
    def update_speaker_profile(self, speaker_id: str, embedding: np.ndarray, confidence: float):
        """Update an existing speaker's profile with new embedding."""
        if speaker_id in self.known_speakers:
            self.known_speakers[speaker_id].add_embedding(embedding.tolist(), confidence)
            self.known_speakers[speaker_id].session_count += 1
            self.save_speaker_profiles()
    
    def process_session_speaker(self, session_speaker_id: int, audio_segment: np.ndarray, 
                              sample_rate: int = 16000) -> Tuple[str, str, Dict]:
        """
        Process a speaker with detailed debug information.
        Returns (display_name, speaker_type, debug_info).
        """
        # Extract embedding from audio
        embedding = self.extract_speaker_embedding(audio_segment, sample_rate)
        
        # Check if we already processed this session speaker
        if session_speaker_id in self.session_speakers:
            known_id = self.session_speakers[session_speaker_id]
            if known_id in self.known_speakers:
                # Update profile with new embedding
                self.update_speaker_profile(known_id, embedding, 0.8)
                return self.known_speakers[known_id].name, "known", {"cached": True}
        
        # Try to identify the speaker
        identified_name, confidence, debug_info = self.identify_speaker(embedding)
        
        if identified_name:
            # Found a match - map this session speaker to known speaker
            for speaker_id, profile in self.known_speakers.items():
                if profile.name == identified_name:
                    self.session_speakers[session_speaker_id] = speaker_id
                    self.update_speaker_profile(speaker_id, embedding, confidence)
                    print(f"✅ Identified: Session Speaker {session_speaker_id} → {identified_name}")
                    return identified_name, "known", debug_info
        
        # Unknown speaker - assign temporary name
        self.unknown_count += 1
        unknown_name = f"Unknown Speaker {self.unknown_count}"
        print(f"❓ Unknown speaker detected: {unknown_name}")
        print(f"   Reason: {debug_info.get('reason', 'unknown')}")
        
        return unknown_name, "unknown", debug_info
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
        print(f"🔧 Debug mode: {'enabled' if enabled else 'disabled'}")
    
    def adjust_thresholds(self, identification_threshold: float = None, 
                         margin_threshold: float = None):
        """Adjust identification thresholds."""
        if identification_threshold is not None:
            self.identification_threshold = identification_threshold
            print(f"🎯 Identification threshold set to {identification_threshold}")
        
        if margin_threshold is not None:
            self.margin_threshold = margin_threshold
            print(f"📏 Margin threshold set to {margin_threshold}")
    
    def get_speaker_name(self, session_speaker_id: int) -> str:
        """Get the display name for a session speaker."""
        if session_speaker_id in self.session_speakers:
            known_id = self.session_speakers[session_speaker_id]
            if known_id in self.known_speakers:
                return self.known_speakers[known_id].name
        
        return f"Speaker {session_speaker_id}"
    
    def list_known_speakers(self):
        """List all known speakers with detailed information."""
        if not self.known_speakers:
            print("📝 No speakers registered yet")
            return
        
        print(f"\n👥 Known Speakers ({len(self.known_speakers)}):")
        for profile in self.known_speakers.values():
            print(f"  • {profile.name}")
            print(f"    - ID: {profile.speaker_id}")
            print(f"    - Registered: {profile.created_at[:10]}")
            print(f"    - Sessions: {profile.session_count}")
            print(f"    - Voice samples: {len(profile.embeddings)}")
        
        print(f"\n🎯 Current Settings:")
        print(f"  - Identification threshold: {self.identification_threshold}")
        print(f"  - Margin threshold: {self.margin_threshold}")
        print(f"  - Debug mode: {self.debug_mode}")

def main():
    """Test the improved speaker identification system."""
    system = ImprovedSpeakerIdentificationSystem()
    
    print("🎯 Improved Speaker Identification System")
    print("🔒 Enhanced false positive prevention")
    
    # Show existing speakers
    system.list_known_speakers()
    
    while True:
        print("\nOptions:")
        print("1. List known speakers")
        print("2. Adjust thresholds")
        print("3. Toggle debug mode")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            system.list_known_speakers()
        elif choice == "2":
            try:
                id_thresh = float(input(f"Identification threshold (current: {system.identification_threshold}): "))
                margin_thresh = float(input(f"Margin threshold (current: {system.margin_threshold}): "))
                system.adjust_thresholds(id_thresh, margin_thresh)
            except ValueError:
                print("Invalid input")
        elif choice == "3":
            system.set_debug_mode(not system.debug_mode)
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 