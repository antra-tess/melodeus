sa#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

reconstruction_dir = Path(__file__).parent / 'funeralia_reconstruction'

# Load manifest
with open(reconstruction_dir / 'manifest.json') as f:
    manifest = json.load(f)

# Load AI responses
with open(reconstruction_dir / 'ai_responses.json') as f:
    ai_responses = json.load(f)

print(f'Human turns: {len(manifest["turns"])}')
print(f'AI responses: {len(ai_responses)}')

# Create events list
events = []

for turn in manifest['turns']:
    ts = datetime.fromisoformat(turn['start'])
    events.append(('human', ts, turn['file'], turn['duration']))

for resp in ai_responses:
    ts = datetime.fromisoformat(resp['timestamp'])
    text = resp.get('text', '')
    words = len(text.split())
    duration = max(1.0, words / 2.5)
    events.append(('ai', ts, text[:50], duration))

events.sort(key=lambda x: x[1])
print(f'Total events: {len(events)}')

# Show first 15 interleaved
print('\nFirst 15 events (interleaved):')
for i, (etype, ts, data, dur) in enumerate(events[:15]):
    marker = "🎤" if etype == "human" else "🤖"
    print(f'  {marker} {ts.strftime("%H:%M:%S")} | {dur:5.1f}s | {str(data)[:40]}')

