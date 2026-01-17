#!/usr/bin/env python3
"""
Auto-deploy poller for melodeus.

Polls a webhook relay server for new pushes, pulls changes, and restarts services.

Usage:
    python auto_deploy.py --webhook-url https://your-app.railway.app

Environment variables:
    WEBHOOK_URL - URL of the webhook relay server
    POLL_INTERVAL - Seconds between polls (default: 2)
"""

import os
import sys
import time
import argparse
import subprocess
import requests
from pathlib import Path

# Files that trigger specific service restarts
DMX_FILES = {"dmx_osc_bridge.py", "dmx_config.yaml"}
MELODEUS_FILES = {
    "unified_voice_conversation_config.py",
    "async_tts_module.py",
    "async_stt_module.py",
    "character_system.py",
    "config.yaml",
    "config_loader.py",
    "context_manager.py",
    "websocket_ui_server.py",
    "tool_parser.py",
    "tools.py",
}


def get_current_sha():
    """Get current local HEAD SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    return result.stdout.strip() if result.returncode == 0 else None


def get_changed_files(old_sha: str, new_sha: str) -> set:
    """Get files changed between two commits."""
    result = subprocess.run(
        ["git", "diff", "--name-only", old_sha, new_sha],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    if result.returncode == 0:
        return set(result.stdout.strip().split("\n"))
    return set()


def pull_changes():
    """Pull latest changes from origin/main."""
    print("📥 Pulling changes...")
    result = subprocess.run(
        ["git", "pull", "origin", "main"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"❌ Pull failed: {result.stderr}")
        return False


def restart_service(service_name: str):
    """Restart a terminal-sessions service via the MCP server."""
    # We'll use a simple approach: kill and restart via subprocess
    # This assumes the services are managed by terminal-sessions MCP
    print(f"🔄 Restarting {service_name}...")

    # For now, we'll just print - actual restart logic depends on your setup
    # You could use: subprocess.run(["pkill", "-f", pattern])
    # Then start the service again

    if service_name == "dmx-bridge":
        # Kill existing
        subprocess.run(["pkill", "-f", "dmx_osc_bridge.py"], capture_output=True)
        time.sleep(1)
        # Start new (in background)
        subprocess.Popen(
            ["./venv/bin/python", "dmx_osc_bridge.py"],
            cwd=Path(__file__).parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        print(f"✅ {service_name} restarted")

    elif service_name == "melodeus":
        # Kill existing
        subprocess.run(["pkill", "-f", "unified_voice_conversation_config.py"], capture_output=True)
        time.sleep(1)
        # Also kill ports it uses
        subprocess.run("lsof -ti :8795 :11235 | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
        time.sleep(1)
        # Start new (in background)
        subprocess.Popen(
            ["./venv/bin/python", "unified_voice_conversation_config.py"],
            cwd=Path(__file__).parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        print(f"✅ {service_name} restarted")


def check_and_deploy(webhook_url: str, last_sha: str) -> str:
    """Check for updates and deploy if needed. Returns new SHA."""
    try:
        response = requests.get(f"{webhook_url}/latest", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"⚠️  Failed to poll webhook: {e}")
        return last_sha

    remote_sha = data.get("sha")

    if not remote_sha:
        return last_sha  # No push recorded yet

    if remote_sha == last_sha:
        return last_sha  # No changes

    print(f"\n🆕 New push detected: {remote_sha[:7]}")
    print(f"   Pusher: {data.get('pusher')}")
    print(f"   Ref: {data.get('ref')}")

    # Pull changes
    old_sha = get_current_sha()
    if not pull_changes():
        return last_sha

    new_sha = get_current_sha()

    # Determine what to restart based on changed files
    changed = get_changed_files(old_sha, new_sha) if old_sha else set()
    print(f"📝 Changed files: {changed}")

    restart_dmx = bool(changed & DMX_FILES)
    restart_melodeus = bool(changed & MELODEUS_FILES)

    # If we can't determine, restart both
    if not changed or (not restart_dmx and not restart_melodeus):
        restart_dmx = True
        restart_melodeus = True

    if restart_dmx:
        restart_service("dmx-bridge")
    if restart_melodeus:
        restart_service("melodeus")

    print(f"✅ Deploy complete!\n")
    return new_sha


def main():
    parser = argparse.ArgumentParser(description="Auto-deploy poller for melodeus")
    parser.add_argument(
        "--webhook-url",
        default=os.environ.get("WEBHOOK_URL"),
        help="URL of the webhook relay server"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("POLL_INTERVAL", "2")),
        help="Seconds between polls (default: 2)"
    )
    args = parser.parse_args()

    if not args.webhook_url:
        print("❌ Error: --webhook-url or WEBHOOK_URL environment variable required")
        sys.exit(1)

    print(f"🚀 Auto-deploy poller starting")
    print(f"   Webhook URL: {args.webhook_url}")
    print(f"   Poll interval: {args.poll_interval}s")
    print(f"   Press Ctrl+C to stop\n")

    last_sha = get_current_sha()
    print(f"📍 Current SHA: {last_sha[:7] if last_sha else 'unknown'}\n")

    try:
        while True:
            last_sha = check_and_deploy(args.webhook_url, last_sha)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n👋 Stopping auto-deploy poller")


if __name__ == "__main__":
    main()
