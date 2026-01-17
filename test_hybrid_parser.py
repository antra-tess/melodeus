#!/usr/bin/env python3
"""
Test script for hybrid context parsing approach.

The hybrid approach:
1. Known AI speakers from config - always accept
2. Speaker lines must appear after a blank line (paragraph boundary)
3. Frequency threshold for unknown speakers (2+)
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class ParsedMessage:
    speaker: str
    content: str
    role: str  # 'user' or 'assistant'
    line_number: int


class HybridContextParser:
    """Parser that uses hybrid approach for speaker detection."""

    # Common human speaker aliases - always accept these
    COMMON_HUMAN_ALIASES = {'H', 'Human', 'User'}

    def __init__(self, known_ai_speakers: Set[str] = None, known_human_speakers: Set[str] = None):
        """
        Initialize parser with known speakers.

        Args:
            known_ai_speakers: Set of AI speaker names from config (characters)
            known_human_speakers: Set of human speaker names from config whitelist
        """
        self.known_ai_speakers = known_ai_speakers or set()
        self.known_human_speakers = known_human_speakers or set()

    def _first_pass_collect_speakers(self, content: str) -> Counter:
        """
        First pass: collect potential speakers at line starts.

        A speaker must be at the start of a line (not mid-line).
        """
        speaker_counts = Counter()
        lines = content.split('\n')

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Match "Speaker: content" at line start
            match = re.match(r'^([^:]+):\s*(.*)$', stripped)
            if match:
                potential_speaker = match.group(1).strip()
                # Basic validation: not too long
                if len(potential_speaker) <= 50:
                    speaker_counts[potential_speaker] += 1

        return speaker_counts

    def _determine_valid_speakers(self, speaker_counts: Counter) -> Set[str]:
        """Determine valid speakers using hybrid approach."""
        valid_speakers = set()

        for speaker, count in speaker_counts.items():
            # Always accept known AI speakers
            if speaker in self.known_ai_speakers:
                valid_speakers.add(speaker)
                continue

            # Always accept known human speakers from whitelist
            if speaker in self.known_human_speakers:
                valid_speakers.add(speaker)
                continue

            # Always accept common human aliases
            if speaker in self.COMMON_HUMAN_ALIASES:
                valid_speakers.add(speaker)
                continue

            # Accept unknown speakers that appear 2+ times
            if count >= 2:
                valid_speakers.add(speaker)

        return valid_speakers

    def parse(self, content: str) -> Tuple[List[ParsedMessage], Set[str], Counter]:
        """
        Parse content using hybrid approach.

        Returns:
            Tuple of (list of ParsedMessage, set of valid speakers, speaker counts)
        """
        # First pass: collect speaker frequencies at paragraph boundaries
        speaker_counts = self._first_pass_collect_speakers(content)

        # Determine valid speakers
        valid_speakers = self._determine_valid_speakers(speaker_counts)

        # Second pass: parse messages
        messages = []
        current_speaker = None
        current_content = []
        current_line_start = 0

        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if not stripped:
                if current_speaker:
                    current_content.append('')
                continue

            # Check for speaker change at line start
            match = re.match(r'^([^:]+):\s*(.*)$', stripped)
            if match:
                potential_speaker = match.group(1).strip()
                message_content = match.group(2).strip()

                if potential_speaker in valid_speakers:
                    # Save previous message
                    if current_speaker and current_content:
                        role = 'assistant' if current_speaker in self.known_ai_speakers else 'user'
                        messages.append(ParsedMessage(
                            speaker=current_speaker,
                            content='\n'.join(current_content).strip(),
                            role=role,
                            line_number=current_line_start
                        ))

                    # Start new message
                    current_speaker = potential_speaker
                    current_content = [message_content] if message_content else []
                    current_line_start = line_num
                    continue

            # Content line - add to current message
            if current_speaker:
                current_content.append(stripped)

        # Save final message
        if current_speaker and current_content:
            role = 'assistant' if current_speaker in self.known_ai_speakers else 'user'
            messages.append(ParsedMessage(
                speaker=current_speaker,
                content='\n'.join(current_content).strip(),
                role=role,
                line_number=current_line_start
            ))

        return messages, valid_speakers, speaker_counts


def test_edge_cases():
    """Test edge cases with synthetic content."""
    print("=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    # Realistic chat log - no artificial "Note:" appearing twice at line starts
    test_content = """Claude: Hello, how can I help you today?

H: I have a question about this code:
Note: this is important
Example: def foo():
    return 42

Claude: I see! The code shows a function that:
1. Takes no arguments
2. Returns 42

Note that the return value is an integer.

H: What about this pattern:
Step 1: Initialize
Step 2: Process

Claude: Those are steps in a process, not speakers.

Unknown User: Can I join?

Claude: Of course! Welcome.

Unknown User: Thanks! I appreciate the help.

Claude: You're welcome."""

    known_ai = {"Claude", "Opus", "Sonnet"}
    parser = HybridContextParser(known_ai_speakers=known_ai)
    messages, valid_speakers, counts = parser.parse(test_content)

    print(f"\nParagraph-boundary speaker counts: {dict(counts)}")
    print(f"Valid speakers: {sorted(valid_speakers)}")
    print(f"Expected: Claude, H, Unknown User")
    print(f"Should NOT include: Note, Example, Step 1, Step 2")

    print("\nParsed messages:")
    for i, msg in enumerate(messages):
        preview = msg.content[:60].replace('\n', ' ')
        if len(msg.content) > 60:
            preview += '...'
        print(f"  {i+1}. [{msg.role}] {msg.speaker} (line {msg.line_number}): {preview}")

    # Verify
    speakers_found = {msg.speaker for msg in messages}

    errors = []
    if "Claude" not in speakers_found:
        errors.append("Claude should be detected")
    if "H" not in speakers_found:
        errors.append("H should be detected")
    if "Unknown User" not in speakers_found:
        errors.append("Unknown User should be detected")
    # Single-occurrence patterns should NOT be detected
    if "Note" in speakers_found:
        errors.append("Note should NOT be detected (only 1 occurrence)")
    if "Example" in speakers_found:
        errors.append("Example should NOT be detected (only 1 occurrence)")
    if "Step 1" in speakers_found:
        errors.append("Step 1 should NOT be detected (only 1 occurrence)")

    if errors:
        print(f"\n❌ Errors:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("\n✅ All edge case assertions passed!")

    return len(errors) == 0


def test_with_file(file_path: str, known_ai_speakers: Set[str], known_human_speakers: Set[str] = None):
    """Test the parser with a specific file."""
    print(f"\n{'='*60}")
    print(f"Testing: {file_path}")
    print('='*60)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    parser = HybridContextParser(
        known_ai_speakers=known_ai_speakers,
        known_human_speakers=known_human_speakers or set()
    )
    messages, valid_speakers, counts = parser.parse(content)

    print(f"\nParagraph-boundary speaker counts (top 15):")
    for speaker, count in counts.most_common(15):
        marker = "✓" if speaker in valid_speakers else " "
        ai_marker = "(AI)" if speaker in known_ai_speakers else ""
        print(f"  {marker} {speaker}: {count} {ai_marker}")

    print(f"\nValid speakers: {sorted(valid_speakers)}")
    print(f"Total messages parsed: {len(messages)}")

    # Show first few messages
    print("\nFirst 5 messages:")
    for i, msg in enumerate(messages[:5]):
        preview = msg.content[:80].replace('\n', ' ')
        if len(msg.content) > 80:
            preview += '...'
        print(f"  {i+1}. [{msg.role}] {msg.speaker}: {preview}")


def main():
    """Run tests on available context files."""

    # Known AI speakers from config (typical setup)
    known_ai_speakers = {
        "Claude", "Opus", "Sonnet", "Haiku",
        "Claude 3 Opus", "Claude 3 Sonnet", "Claude 3.5 Sonnet",
        "Opus 4.5", "Sonnet45", "Claude-Bedrock",
        "Claude Opus 4", "Claude Opus 4.1",
    }

    # Test edge cases first
    success = test_edge_cases()

    # Find and test available context files
    base_dir = Path(__file__).parent

    # Test manykin.md (no whitelist needed - all speakers appear 2+ times)
    file_path = base_dir / "manykin.md"
    if file_path.exists():
        test_with_file(str(file_path), known_ai_speakers)

    # Test input_prompt.txt WITHOUT whitelist (misses single-occurrence speakers)
    file_path = base_dir / "input_prompt.txt"
    if file_path.exists():
        print("\n" + "="*60)
        print("WITHOUT whitelist (single-occurrence speakers missed):")
        test_with_file(str(file_path), known_ai_speakers)

        # Test WITH whitelist for single-occurrence speakers
        print("\n" + "="*60)
        print("WITH whitelist (rainparadox, .0xg now included):")
        human_whitelist = {"rainparadox", ".0xg", "repligate"}
        test_with_file(str(file_path), known_ai_speakers, human_whitelist)


if __name__ == "__main__":
    main()
