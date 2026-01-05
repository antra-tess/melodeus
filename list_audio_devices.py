#!/usr/bin/env python3
"""
List all available audio devices on the system.
This helps configure the output_device_name in config.yaml
"""

import math
import sys
from typing import Iterable, List, Tuple

from audio_aec import AudioEngine


DEVICE_TYPE_LABELS = {
    "default_duplex": "DEFAULT INPUT & OUTPUT",
    "default_input": "DEFAULT INPUT",
    "default_output": "DEFAULT OUTPUT",
    "duplex": "INPUT & OUTPUT",
    "input": "INPUT-ONLY",
    "output": "OUTPUT-ONLY",
}


def _format_sample_rates(rates: Iterable[float]) -> str:
    """Return a friendly label for the list of supported sample rates."""
    normalized: List[str] = []
    seen = set()
    for rate in rates:
        try:
            value = float(rate)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        if abs(value - round(value)) < 1e-6:
            display = str(int(round(value)))
        else:
            display = f"{value:.2f}".rstrip("0").rstrip(".")
        if display not in seen:
            seen.add(display)
            normalized.append(display)
    normalized.sort(key=lambda item: float(item))
    return ", ".join(normalized) if normalized else "Not reported"


def list_all_audio_devices():
    """List all audio devices (both input and output) using audio_aec."""
    engine = AudioEngine()

    try:
        devices = engine.list_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to list audio devices: {exc}") from exc

    try:
        host_info = engine.get_host_info()
    except Exception:
        host_info = {}

    default_input_name = host_info.get("default_input")
    default_output_name = host_info.get("default_output")

    print("\n" + "=" * 60)
    print("🎤 AUDIO DEVICES")
    print("=" * 60)

    if host_info:
        backend = host_info.get("backend", "Unknown backend")
        version = host_info.get("version_text") or host_info.get("version")
        print(f"Backend: {backend}" + (f" ({version})" if version else ""))
        if default_input_name:
            print(f"Default input: '{default_input_name}'")
        if default_output_name:
            print(f"Default output: '{default_output_name}'")

    for idx, entry in enumerate(devices, start=1):
        name, device_type, is_input, is_output, input_rates, output_rates = _coerce_device_entry(entry)

        capabilities = []
        if is_input:
            capabilities.append("INPUT")
        if is_output:
            capabilities.append("OUTPUT")
        if not capabilities:
            capabilities.append("UNKNOWN")

        status_labels: List[str] = []
        if device_type in DEVICE_TYPE_LABELS:
            status_labels.append(DEVICE_TYPE_LABELS[device_type])
        if default_input_name and name == default_input_name and "DEFAULT INPUT" not in status_labels:
            status_labels.append("DEFAULT INPUT")
        if default_output_name and name == default_output_name and "DEFAULT OUTPUT" not in status_labels:
            status_labels.append("DEFAULT OUTPUT")

        print(f"\n📍 Device {idx}: '{name}'")
        print(f"   Type: {', '.join(capabilities)}")
        if status_labels:
            print(f"   Status: {', '.join(status_labels)} ⭐")

        if is_input:
            print(f"   Supported input sample rates: {_format_sample_rates(input_rates)}")
        if is_output:
            print(f"   Supported output sample rates: {_format_sample_rates(output_rates)}")

    print("\n" + "=" * 60)
    print("💡 HOW TO USE:")
    print("=" * 60)
    print("1. Find your desired INPUT and OUTPUT devices from the list above")
    print("2. Note part of their names (device names are matched using partial matching)")
    print("3. In config.yaml, configure devices:")
    print("\n📥 For INPUT devices (microphones) - under 'stt:':")
    print('   input_device_name: "<partial_device_name>"')
    print("\n📤 For OUTPUT devices (speakers) - under 'tts:':")
    print('   output_device_name: "<partial_device_name>"')
    print("\n📋 Example config.yaml:")
    print("   stt:")
    print('     input_device_name: "Scarlett 18i20"    # Matches audio interface mic')
    print("     # or")
    print('     input_device_name: "MacBook Pro"       # Matches built-in mic')
    print("   tts:")
    print('     output_device_name: "Loopback Audio"   # Matches virtual audio device')
    print("     # or")
    print('     output_device_name: "Speakers"         # Matches any device with \'Speakers\'')
    print("\n💡 Use 'null' for default devices")
    print("\n")


def _coerce_device_entry(entry: Tuple) -> Tuple[str, str, bool, bool, Tuple[float, ...], Tuple[float, ...]]:
    """
    Normalize the device entry coming from audio_aec to avoid unpacking surprises.

    Older builds may omit some fields, so we coerce them to sensible defaults.
    """
    name = entry[0] if len(entry) > 0 else "Unknown device"
    device_type = entry[1] if len(entry) > 1 else "duplex"
    is_input = bool(entry[2]) if len(entry) > 2 else False
    is_output = bool(entry[3]) if len(entry) > 3 else False
    input_rates = tuple(entry[4]) if len(entry) > 4 else tuple()
    output_rates = tuple(entry[5]) if len(entry) > 5 else tuple()
    return name, device_type, is_input, is_output, input_rates, output_rates


if __name__ == "__main__":
    try:
        list_all_audio_devices()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
