#!/bin/bash
cd /Users/llms/Documents/TalkToLLM/melodeus
# Use config.yaml directly (no preset override)
export PYTHONPATH="/Users/llms/Documents/TalkToLLM/mel-aec/python:$PYTHONPATH"

# Kill any zombie processes from previous runs
echo "Cleaning up old processes..."
pkill -9 -f "unified_voice_conversation_config.py" 2>/dev/null
lsof -ti :8795 | xargs kill -9 2>/dev/null  # WebSocket server
lsof -ti :8080 | xargs kill -9 2>/dev/null  # HTTP server
lsof -ti :11235 | xargs kill -9 2>/dev/null # Flic button listener
sleep 1

# Start HTTP server for UI in background
python3 -m http.server 8080 &
HTTP_PID=$!

# Start main app
./venv/bin/python unified_voice_conversation_config.py

# Clean up HTTP server when main app exits
kill $HTTP_PID 2>/dev/null
