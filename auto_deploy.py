#!/usr/bin/env python3
"""
Auto-deploy poller for melodeus.

Polls a webhook relay server for new pushes and manual triggers.
Handles both code updates (melodeus) and config updates (melodeus-config).

Usage:
    python auto_deploy.py --webhook-url https://your-app.railway.app

Environment variables:
    WEBHOOK_URL - URL of the webhook relay server
    POLL_INTERVAL - Seconds between polls (default: 2)
    CONFIG_REPO - Config repo name (default: antra-tess/melodeus-config)
    CODE_REPO - Code repo name (default: antra-tess/melodeus)
"""

import os
import sys
import time
import argparse
import subprocess
import requests
from pathlib import Path

# Repository configuration
CODE_REPO = os.environ.get("CODE_REPO", "antra-tess/melodeus")
CONFIG_REPO = os.environ.get("CONFIG_REPO", "antra-tess/melodeus-config")

# Directories
MELODEUS_DIR = Path(__file__).parent
CONFIG_DIR = MELODEUS_DIR.parent / "melodeus-config"

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

# Config files to sync
CONFIG_FILES = {
    "dmx_config.yaml",
    "config.yaml",
}

# Config directories to sync
CONFIG_DIRS = {
    "presets",
    "context_states",
}


def get_current_sha(repo_dir: Path):
    """Get current local HEAD SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_dir
    )
    return result.stdout.strip() if result.returncode == 0 else None


def get_changed_files(old_sha: str, new_sha: str, repo_dir: Path) -> set:
    """Get files changed between two commits."""
    result = subprocess.run(
        ["git", "diff", "--name-only", old_sha, new_sha],
        capture_output=True,
        text=True,
        cwd=repo_dir
    )
    if result.returncode == 0:
        return set(result.stdout.strip().split("\n"))
    return set()


def pull_changes(repo_dir: Path):
    """Pull latest changes from origin/main."""
    print(f"📥 Pulling changes in {repo_dir.name}...")
    result = subprocess.run(
        ["git", "pull", "origin", "main"],
        capture_output=True,
        text=True,
        cwd=repo_dir
    )
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"❌ Pull failed: {result.stderr}")
        return False


def sync_configs():
    """Sync config files and directories from config repo to melodeus.
    
    Returns set of synced file names for determining which services to restart.
    """
    import shutil

    if not CONFIG_DIR.exists():
        print(f"⚠️  Config directory not found: {CONFIG_DIR}")
        return set()

    print("⚙️  Syncing config files...")
    synced = set()

    # Sync individual files
    for config_file in CONFIG_FILES:
        src = CONFIG_DIR / config_file
        dst = MELODEUS_DIR / config_file
        if src.exists():
            content = src.read_text()
            dst.write_text(content)
            synced.add(config_file)
            print(f"   ✓ {config_file}")

    # Sync directories
    for config_dir in CONFIG_DIRS:
        src = CONFIG_DIR / config_dir
        dst = MELODEUS_DIR / config_dir
        if src.exists() and src.is_dir():
            # Remove existing and copy fresh
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            synced.add(f"{config_dir}/")
            print(f"   ✓ {config_dir}/")

    if synced:
        print(f"✅ Synced {len(synced)} config items")
    return synced


def restart_service(service_name: str):
    """Restart a service."""
    print(f"🔄 Restarting {service_name}...")

    if service_name == "dmx-bridge":
        subprocess.run(["pkill", "-f", "dmx_osc_bridge.py"], capture_output=True)
        time.sleep(1)
        subprocess.Popen(
            ["./venv/bin/python", "dmx_osc_bridge.py"],
            cwd=MELODEUS_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        print(f"✅ {service_name} restarted")

    elif service_name == "melodeus":
        subprocess.run(["pkill", "-f", "unified_voice_conversation_config.py"], capture_output=True)
        time.sleep(1)
        subprocess.run("lsof -ti :8795 :11235 | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
        time.sleep(1)
        subprocess.Popen(
            ["./venv/bin/python", "unified_voice_conversation_config.py"],
            cwd=MELODEUS_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        print(f"✅ {service_name} restarted")


def handle_code_update(webhook_url: str, last_sha: str) -> str:
    """Handle code repository updates. Returns new SHA."""
    try:
        response = requests.get(f"{webhook_url}/latest/{CODE_REPO}", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return last_sha

    remote_sha = data.get("sha")
    trigger = data.get("trigger")

    # Check for manual triggers
    if trigger:
        action = trigger.get("action")
        print(f"\n🎯 Manual trigger: {action}")

        # Acknowledge the trigger
        try:
            requests.post(f"{webhook_url}/ack/{CODE_REPO}", timeout=5)
        except:
            pass

        if action == "full":
            pull_changes(MELODEUS_DIR)
            restart_service("dmx-bridge")
            restart_service("melodeus")
            print("✅ Full restart complete!\n")
            return get_current_sha(MELODEUS_DIR) or last_sha

        elif action == "pull":
            pull_changes(MELODEUS_DIR)
            print("✅ Pull complete (no restart)\n")
            return get_current_sha(MELODEUS_DIR) or last_sha

        elif action == "config":
            synced = sync_configs()
            # Restart services based on what was synced
            if "dmx_config.yaml" in synced:
                restart_service("dmx-bridge")
            if "config.yaml" in synced or "presets/" in synced or "context_states/" in synced:
                restart_service("melodeus")
            print("✅ Config sync complete!\n")
            return last_sha

    # Check for new pushes
    if not remote_sha or remote_sha == last_sha:
        return last_sha

    print(f"\n🆕 New code push detected: {remote_sha[:7]}")
    print(f"   Pusher: {data.get('pusher')}")
    print(f"   Ref: {data.get('ref')}")

    old_sha = get_current_sha(MELODEUS_DIR)
    if not pull_changes(MELODEUS_DIR):
        return last_sha

    new_sha = get_current_sha(MELODEUS_DIR)

    # Determine what to restart
    changed = get_changed_files(old_sha, new_sha, MELODEUS_DIR) if old_sha else set()
    print(f"📝 Changed files: {changed}")

    restart_dmx = bool(changed & DMX_FILES)
    restart_melodeus = bool(changed & MELODEUS_FILES)

    if not changed or (not restart_dmx and not restart_melodeus):
        restart_dmx = True
        restart_melodeus = True

    if restart_dmx:
        restart_service("dmx-bridge")
    if restart_melodeus:
        restart_service("melodeus")

    print(f"✅ Code deploy complete!\n")
    return new_sha


def handle_config_update(webhook_url: str, last_sha: str) -> str:
    """Handle config repository updates. Returns new SHA."""
    if not CONFIG_DIR.exists():
        return last_sha

    try:
        response = requests.get(f"{webhook_url}/latest/{CONFIG_REPO}", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return last_sha

    remote_sha = data.get("sha")
    trigger = data.get("trigger")

    # Check for manual config triggers
    if trigger:
        action = trigger.get("action")
        print(f"\n🎯 Config manual trigger: {action}")

        try:
            requests.post(f"{webhook_url}/ack/{CONFIG_REPO}", timeout=5)
        except:
            pass

        pull_changes(CONFIG_DIR)
        synced = sync_configs()
        
        # Restart services based on what was synced
        if "dmx_config.yaml" in synced:
            restart_service("dmx-bridge")
        if "config.yaml" in synced or "presets/" in synced or "context_states/" in synced:
            restart_service("melodeus")
        
        print("✅ Config update complete!\n")
        return get_current_sha(CONFIG_DIR) or last_sha

    # Check for new config pushes
    if not remote_sha or remote_sha == last_sha:
        return last_sha

    print(f"\n🆕 New config push detected: {remote_sha[:7]}")
    print(f"   Pusher: {data.get('pusher')}")

    old_sha = get_current_sha(CONFIG_DIR)
    if not pull_changes(CONFIG_DIR):
        return last_sha

    new_sha = get_current_sha(CONFIG_DIR)
    
    # Check what changed in config repo
    changed = get_changed_files(old_sha, new_sha, CONFIG_DIR) if old_sha else set()
    print(f"📝 Changed config files: {changed}")
    
    synced = sync_configs()

    # Restart services based on what changed
    if "dmx_config.yaml" in changed or "dmx_config.yaml" in synced:
        restart_service("dmx-bridge")
    if any(f in changed for f in ["config.yaml", "presets/", "context_states/"]) or \
       any(f.startswith("presets/") or f.startswith("context_states/") for f in changed) or \
       "config.yaml" in synced:
        restart_service("melodeus")

    print(f"✅ Config deploy complete!\n")
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
    print(f"   Code repo: {CODE_REPO}")
    print(f"   Config repo: {CONFIG_REPO}")
    print(f"   Config dir: {CONFIG_DIR}")
    print(f"   Press Ctrl+C to stop\n")

    code_sha = get_current_sha(MELODEUS_DIR)
    config_sha = get_current_sha(CONFIG_DIR) if CONFIG_DIR.exists() else None

    print(f"📍 Code SHA: {code_sha[:7] if code_sha else 'unknown'}")
    print(f"📍 Config SHA: {config_sha[:7] if config_sha else 'not cloned'}\n")

    try:
        while True:
            code_sha = handle_code_update(args.webhook_url, code_sha)
            if CONFIG_DIR.exists():
                config_sha = handle_config_update(args.webhook_url, config_sha)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n👋 Stopping auto-deploy poller")


if __name__ == "__main__":
    main()
