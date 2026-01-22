#!/usr/bin/env python3
"""
Audio Noise Gate with Attack, Hold, and Release
Proper gate implementation for cleaning up microphone input.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class GateConfig:
    """Configuration for the noise gate."""
    threshold_db: float = -40.0      # Gate opens above this level (dB)
    attack_ms: float = 1.0           # How fast gate opens (ms)
    hold_ms: float = 100.0           # Keep gate open after signal drops (ms)
    release_ms: float = 50.0         # How smoothly gate closes (ms)
    range_db: float = -80.0          # Attenuation when closed (dB), -inf for full mute
    lookahead_ms: float = 0.0        # Lookahead for transient preservation (ms)
    hysteresis_db: float = 3.0       # Hysteresis to prevent chattering (dB)
    enabled: bool = True


class NoiseGate:
    """
    Professional-style noise gate with attack, hold, and release.
    
    State machine:
    - CLOSED: Gate is closed, applying range_db attenuation
    - ATTACK: Gate is opening (signal exceeded threshold)
    - OPEN: Gate is fully open
    - HOLD: Signal dropped but waiting before release
    - RELEASE: Gate is closing smoothly
    """
    
    # States
    CLOSED = 0
    ATTACK = 1
    OPEN = 2
    HOLD = 3
    RELEASE = 4
    
    def __init__(self, config: GateConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Convert times to samples
        self._update_timing()
        
        # State
        self.state = self.CLOSED
        self.envelope = 0.0  # Current gain envelope (0-1)
        self.hold_counter = 0  # Samples remaining in hold
        
        # For RMS calculation
        self.rms_window_samples = int(sample_rate * 0.010)  # 10ms RMS window
        self.rms_buffer = np.zeros(self.rms_window_samples)
        self.rms_index = 0
        
        # Convert thresholds to linear
        self.threshold_linear = self._db_to_linear(config.threshold_db)
        self.threshold_close = self._db_to_linear(config.threshold_db - config.hysteresis_db)
        self.range_linear = self._db_to_linear(config.range_db)
        
        # Debug/monitoring
        self.last_rms_db = -100.0
        self.gate_open = False
    
    def _update_timing(self):
        """Update timing coefficients from config."""
        # Attack coefficient (how fast envelope rises)
        if self.config.attack_ms > 0:
            attack_samples = max(1, int(self.sample_rate * self.config.attack_ms / 1000))
            self.attack_coeff = 1.0 - np.exp(-2.2 / attack_samples)
        else:
            self.attack_coeff = 1.0
        
        # Release coefficient (how fast envelope falls)
        if self.config.release_ms > 0:
            release_samples = max(1, int(self.sample_rate * self.config.release_ms / 1000))
            self.release_coeff = 1.0 - np.exp(-2.2 / release_samples)
        else:
            self.release_coeff = 1.0
        
        # Hold time in samples
        self.hold_samples = int(self.sample_rate * self.config.hold_ms / 1000)
    
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear gain."""
        if db <= -100:
            return 0.0
        return 10 ** (db / 20.0)
    
    def _linear_to_db(self, linear: float) -> float:
        """Convert linear gain to dB."""
        if linear <= 0:
            return -100.0
        return 20.0 * np.log10(linear)
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS level of audio chunk."""
        return np.sqrt(np.mean(audio ** 2))
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through the noise gate.
        
        Args:
            audio: Input audio as float32 numpy array (-1 to 1)
            
        Returns:
            Gated audio as float32 numpy array
        """
        if not self.config.enabled:
            return audio
        
        # Calculate RMS level
        rms = self._calculate_rms(audio)
        self.last_rms_db = self._linear_to_db(rms)
        
        # State machine
        if self.state == self.CLOSED:
            if rms > self.threshold_linear:
                self.state = self.ATTACK
                self.gate_open = True
        
        elif self.state == self.ATTACK:
            if rms < self.threshold_close:
                # Signal dropped during attack - go to hold
                self.state = self.HOLD
                self.hold_counter = self.hold_samples
            else:
                # Envelope rising
                self.envelope += self.attack_coeff * (1.0 - self.envelope)
                if self.envelope >= 0.99:
                    self.envelope = 1.0
                    self.state = self.OPEN
        
        elif self.state == self.OPEN:
            if rms < self.threshold_close:
                self.state = self.HOLD
                self.hold_counter = self.hold_samples
        
        elif self.state == self.HOLD:
            if rms > self.threshold_linear:
                # Signal came back, go to attack to ramp up
                self.state = self.ATTACK
            else:
                self.hold_counter -= len(audio)
                if self.hold_counter <= 0:
                    self.state = self.RELEASE
        
        elif self.state == self.RELEASE:
            if rms > self.threshold_linear:
                # Signal came back, re-open
                self.state = self.ATTACK
            else:
                # Envelope falling
                target = self.range_linear
                self.envelope += self.release_coeff * (target - self.envelope)
                if self.envelope <= target + 0.01:
                    self.envelope = target
                    self.state = self.CLOSED
                    self.gate_open = False
        
        # Apply envelope
        return audio * self.envelope
    
    def reset(self):
        """Reset gate state."""
        self.state = self.CLOSED
        self.envelope = self.range_linear
        self.hold_counter = 0
        self.gate_open = False
    
    def get_status(self) -> dict:
        """Get current gate status for monitoring."""
        state_names = ['CLOSED', 'ATTACK', 'OPEN', 'HOLD', 'RELEASE']
        return {
            'state': state_names[self.state],
            'envelope_db': self._linear_to_db(self.envelope),
            'input_db': self.last_rms_db,
            'threshold_db': self.config.threshold_db,
            'gate_open': self.gate_open
        }


class ExpansionGate(NoiseGate):
    """
    Expansion gate - softer than hard gate.
    Gradually reduces gain below threshold rather than hard cut.
    """
    
    def __init__(self, config: GateConfig, sample_rate: int = 16000, ratio: float = 4.0):
        super().__init__(config, sample_rate)
        self.ratio = ratio  # Expansion ratio (4:1 means 4dB reduction per 1dB below threshold)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process with soft expansion rather than hard gating."""
        if not self.config.enabled:
            return audio
        
        rms = self._calculate_rms(audio)
        self.last_rms_db = self._linear_to_db(rms)
        
        if rms > self.threshold_linear:
            # Above threshold - full gain
            target_gain = 1.0
            self.gate_open = True
        else:
            # Below threshold - apply expansion
            db_below = self.config.threshold_db - self.last_rms_db
            gain_reduction_db = db_below * (self.ratio - 1) / self.ratio
            target_gain = max(self.range_linear, self._db_to_linear(-gain_reduction_db))
            self.gate_open = target_gain > 0.5
        
        # Smooth envelope
        if target_gain > self.envelope:
            self.envelope += self.attack_coeff * (target_gain - self.envelope)
        else:
            self.envelope += self.release_coeff * (target_gain - self.envelope)
        
        return audio * self.envelope


# Convenience function for creating a gate from config dict
def create_gate_from_config(config_dict: dict, sample_rate: int = 16000) -> NoiseGate:
    """Create a noise gate from a configuration dictionary."""
    gate_config = GateConfig(
        threshold_db=config_dict.get('threshold_db', -40.0),
        attack_ms=config_dict.get('attack_ms', 1.0),
        hold_ms=config_dict.get('hold_ms', 100.0),
        release_ms=config_dict.get('release_ms', 50.0),
        range_db=config_dict.get('range_db', -80.0),
        hysteresis_db=config_dict.get('hysteresis_db', 3.0),
        enabled=config_dict.get('enabled', True)
    )
    
    gate_type = config_dict.get('type', 'hard')
    if gate_type == 'expansion':
        ratio = config_dict.get('ratio', 4.0)
        return ExpansionGate(gate_config, sample_rate, ratio)
    else:
        return NoiseGate(gate_config, sample_rate)


if __name__ == "__main__":
    # Test the gate
    import time
    
    config = GateConfig(
        threshold_db=-30.0,
        attack_ms=2.0,
        hold_ms=150.0,
        release_ms=100.0,
        range_db=-60.0
    )
    
    gate = NoiseGate(config, sample_rate=16000)
    
    # Simulate audio chunks
    chunk_size = 1024
    sample_rate = 16000
    
    print("Testing noise gate...")
    print(f"Threshold: {config.threshold_db} dB")
    print(f"Attack: {config.attack_ms} ms, Hold: {config.hold_ms} ms, Release: {config.release_ms} ms")
    print()
    
    # Generate test signal: silence -> loud -> silence
    for i in range(50):
        if i < 10:
            # Silence
            audio = np.random.randn(chunk_size).astype(np.float32) * 0.001
        elif i < 30:
            # Loud signal
            audio = np.random.randn(chunk_size).astype(np.float32) * 0.3
        else:
            # Silence again
            audio = np.random.randn(chunk_size).astype(np.float32) * 0.001
        
        output = gate.process(audio)
        status = gate.get_status()
        
        input_rms = np.sqrt(np.mean(audio ** 2))
        output_rms = np.sqrt(np.mean(output ** 2))
        
        print(f"Chunk {i:2d}: Input {status['input_db']:6.1f} dB | "
              f"State: {status['state']:8s} | "
              f"Envelope: {status['envelope_db']:6.1f} dB | "
              f"Output RMS: {output_rms:.4f}")
