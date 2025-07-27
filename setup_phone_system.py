#!/usr/bin/env python3
"""
Setup Script for Phone Voice Participation System
Prepares the system for the Claude Sonnet Funeral event.
"""

import subprocess
import sys
from pathlib import Path
import shutil

def check_command(command):
    """Check if a command is available."""
    return shutil.which(command) is not None

def install_requirements():
    """Install Python requirements."""
    print("📦 Installing Python requirements...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_phone.txt'], 
                      check=True)
        print("✅ Python requirements installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python requirements")
        return False
    
    return True

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    print("🎵 Checking for ffmpeg...")
    
    if check_command('ffmpeg'):
        print("✅ ffmpeg is installed")
        return True
    else:
        print("❌ ffmpeg not found")
        print("   Install with:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/")
        return False

def setup_webapp_directory():
    """Ensure webapp directory is set up correctly."""
    print("🌐 Setting up webapp directory...")
    
    webapp_dir = Path("webapp")
    webapp_dir.mkdir(exist_ok=True)
    
    # Check if index.html exists
    index_file = webapp_dir / "index.html"
    if index_file.exists():
        print("✅ Webapp files found")
        return True
    else:
        print("❌ webapp/index.html not found")
        print("   Make sure webapp files are in the webapp/ directory")
        return False

def check_config():
    """Check if config.yaml has required API keys."""
    print("⚙️ Checking configuration...")
    
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        api_keys = config.get('api_keys', {})
        required_keys = ['deepgram', 'elevenlabs', 'anthropic']
        
        missing_keys = []
        for key in required_keys:
            if not api_keys.get(key) or api_keys[key] == f'your_{key}_api_key_here':
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing API keys: {', '.join(missing_keys)}")
            print("   Update config.yaml with your API keys")
            return False
        else:
            print("✅ API keys configured")
            return True
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def main():
    """Main setup function."""
    print("🎭 Claude Sonnet Funeral - Phone Participation Setup")
    print("=" * 50)
    
    all_good = True
    
    # Check Python requirements
    if not install_requirements():
        all_good = False
    
    # Check ffmpeg
    if not check_ffmpeg():
        all_good = False
    
    # Setup webapp
    if not setup_webapp_directory():
        all_good = False
    
    # Check config
    if not check_config():
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 Setup complete! Ready to run the phone participation system.")
        print("\n🚀 To start the system:")
        print("   python webapp_server.py --public-host YOUR_IP_ADDRESS")
        print("\n📱 People will be able to scan the QR code and join via phone!")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 