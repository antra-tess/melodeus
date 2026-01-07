#!/usr/bin/env python3
"""
Context Manager for Voice Conversation System
Handles multiple conversation contexts with persistent state management
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import shutil


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user", "assistant", "system", or "tool"
    content: str
    timestamp: datetime
    status: str = "completed"  # "completed", "interrupted", "pending"
    character: Optional[str] = None  # For multi-character conversations
    speaker_id: Optional[str] = None  # From STT
    speaker_name: Optional[str] = None  # From voice fingerprinting
    metadata: Dict[str, Any] = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ContextConfig:
    """Configuration for a conversation context."""
    name: str
    history_file: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    character_histories: Optional[Dict[str, str]] = None  # Maps character name to their history file
    

class ConversationContext:
    """Represents a single conversation context with its state."""
    
    def __init__(self, config: ContextConfig, state_dir: Path):
        self.config = config
        self.state_dir = state_dir
        self.state_file = state_dir / f"{config.name}_state.json"
        
        # Original history loaded from file (immutable)
        self.original_history: List[ConversationTurn] = []
        
        # Character-specific histories for this context (immutable)
        self.character_histories: Dict[str, List[ConversationTurn]] = {}
        
        # Current conversation state (mutable)
        self.current_history: List[ConversationTurn] = []
        
        # Additional context state
        self.metadata: Dict[str, Any] = config.metadata or {}
        self.last_save_time: Optional[datetime] = None
        self.is_modified = False
        
    def load_state(self) -> bool:
        """Load persistent state from file."""
        # Preserve character histories that were loaded
        preserved_char_histories = self.character_histories.copy()
        
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # Reconstruct conversation history from saved state
                self.current_history = []
                for turn_data in state_data.get('history', []):
                    turn = ConversationTurn(
                        role=turn_data['role'],
                        content=turn_data['content'],
                        timestamp=datetime.fromisoformat(turn_data['timestamp']),
                        status=turn_data.get('status', 'completed'),
                        character=turn_data.get('character'),
                        speaker_name=turn_data.get('speaker_name'),
                        metadata=turn_data.get('metadata', {})
                    )
                    self.current_history.append(turn)
                
                self.metadata = state_data.get('metadata', {})
                self.last_save_time = datetime.fromisoformat(state_data['last_save'])
                
                # Restore preserved character histories
                self.character_histories = preserved_char_histories
                
                print(f"✅ Loaded state for context '{self.config.name}': {len(self.current_history)} turns")
                if self.character_histories:
                    print(f"   📚 Preserved character histories for: {list(self.character_histories.keys())}")
                return True
            else:
                # No saved state, use original history
                self.current_history = self.original_history.copy()
                # Restore preserved character histories
                self.character_histories = preserved_char_histories
                print(f"📋 No saved state for context '{self.config.name}', using original history")
                return False
                
        except Exception as e:
            print(f"❌ Error loading state for context '{self.config.name}': {e}")
            self.current_history = self.original_history.copy()
            # Restore preserved character histories
            self.character_histories = preserved_char_histories
            return False
    
    def save_state(self) -> bool:
        """Save current state to file."""
        try:
            # Ensure state directory exists
            self.state_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare state data
            history_data = []
            for turn in self.current_history:
                # Skip deleted messages
                if getattr(turn, 'status', None) == 'deleted':
                    continue
                # Handle timestamp - could be datetime or float
                if isinstance(turn.timestamp, datetime):
                    timestamp_str = turn.timestamp.isoformat()
                elif isinstance(turn.timestamp, (int, float)):
                    # Convert Unix timestamp to datetime
                    timestamp_str = datetime.fromtimestamp(turn.timestamp).isoformat()
                else:
                    # Fallback to current time if timestamp is invalid
                    timestamp_str = datetime.now().isoformat()
                
                turn_data = {
                    'role': turn.role,
                    'content': turn.content,
                    'timestamp': timestamp_str,
                    'status': turn.status,
                }
                if turn.character:
                    turn_data['character'] = turn.character
                if hasattr(turn, 'speaker_name') and turn.speaker_name:
                    turn_data['speaker_name'] = turn.speaker_name
                if hasattr(turn, 'metadata') and turn.metadata:
                    turn_data['metadata'] = turn.metadata
                history_data.append(turn_data)
            
            state_data = {
                'context_name': self.config.name,
                'history': history_data,
                'metadata': self.metadata,
                'last_save': datetime.now().isoformat(),
                'original_history_file': self.config.history_file
            }
            
            # Write to temporary file first
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
            self.last_save_time = datetime.now()
            self.is_modified = False
            
            print(f"💾 Saved state for context '{self.config.name}': {len(self.current_history)} turns")
            return True
            
        except Exception as e:
            print(f"❌ Error saving state for context '{self.config.name}': {e}")
            return False
    
    def reset_to_original(self):
        """Reset current history to original loaded history."""
        self.current_history = self.original_history.copy()
        self.is_modified = True
        print(f"🔄 Reset context '{self.config.name}' to original history")
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new turn to the conversation."""
        self.current_history.append(turn)
        self.is_modified = True
    
    def update_turn(self, index: int, turn: ConversationTurn):
        """Update an existing turn."""
        if 0 <= index < len(self.current_history):
            self.current_history[index] = turn
            self.is_modified = True
    
    def remove_turn(self, index: int):
        """Remove a turn from the conversation."""
        if 0 <= index < len(self.current_history):
            self.current_history.pop(index)
            self.is_modified = True


class ContextManager:
    """Manages multiple conversation contexts."""
    
    def __init__(self, contexts_config: List[Dict[str, Any]], state_dir: str = "./context_states"):
        self.state_dir = Path(state_dir)
        self.contexts: Dict[str, ConversationContext] = {}
        self.active_context_name: Optional[str] = None
        
        # Auto-save configuration
        self.auto_save_enabled = True
        self.auto_save_interval = 30  # seconds
        self.auto_save_task: Optional[asyncio.Task] = None
        
        # Initialize contexts from config
        for ctx_config in contexts_config:
            config = ContextConfig(
                name=ctx_config['name'],
                history_file=ctx_config['history_file'],
                description=ctx_config.get('description'),
                metadata=ctx_config.get('metadata', {}),
                character_histories=ctx_config.get('character_histories')
            )
            context = ConversationContext(config, self.state_dir)
            self.contexts[config.name] = context
        
        # Set first context as active if available
        if self.contexts:
            self.active_context_name = list(self.contexts.keys())[0]
            print(f"🎯 Active context: '{self.active_context_name}'")
    
    def _parse_character_history_file(self, file_path: str, character_name: str) -> List[Dict[str, str]]:
        """Parse a character-specific history file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            messages = []
            lines = content.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                
                # Simple parsing: look for "Speaker: content" pattern
                if ': ' in line:
                    speaker, message = line.split(': ', 1)
                    
                    # Validate that this is actually a speaker name (not just any text with a colon)
                    # Speaker names should be relatively short and not contain certain patterns
                    is_valid_speaker = (
                        len(speaker) <= 50 and  # Not too long
                        not speaker.lower().startswith(('most', 'but ', 'and ', 'the ', 'this ', 'that ',
                                                      'when ', 'where ', 'what ', 'why ', 'how ',
                                                      'read ', 'edit ', 'note ', 'warning ')) and
                        not any(char in speaker for char in ['(', ')', '[', ']', '{', '}', '"', "'"])
                    )
                    
                    if not is_valid_speaker:
                        # Not a speaker line, treat as content
                        i += 1
                        continue
                    
                    # Collect multi-line messages
                    full_message = [message]
                    i += 1
                    
                    # Continue collecting lines until we hit another speaker or end
                    while i < len(lines):
                        next_line = lines[i].strip()
                        # Check if this line starts with a speaker pattern (any word followed by colon)
                        if not next_line or (': ' in next_line and 
                                           next_line.split(': ', 1)[0].strip() and
                                           not next_line.startswith((' ', '\t'))):  # Not indented
                            break
                        full_message.append(next_line)
                        i += 1
                    
                    # Determine role
                    if speaker.lower() in ['user', 'human', 'h']:
                        role = 'user'
                    else:
                        role = 'assistant'
                    
                    content = '\n'.join(full_message).strip()
                    messages.append({
                        'role': role,
                        'content': content
                    })
                    
                    # Debug logging
                    preview = content[:100] + '...' if len(content) > 100 else content
                    print(f"   📝 Parsed {role} message from {speaker}: {preview}")
                else:
                    i += 1
            
            return messages
            
        except Exception as e:
            print(f"❌ Error parsing character history file {file_path}: {e}")
            return []
    
    def get_active_context(self) -> Optional[ConversationContext]:
        """Get the currently active context."""
        if self.active_context_name:
            return self.contexts.get(self.active_context_name)
        return None
    
    def switch_context(self, context_name: str) -> bool:
        """Switch to a different context."""
        print(f"🔄 ContextManager.switch_context called with: '{context_name}'")
        print(f"📋 Available contexts: {list(self.contexts.keys())}")
                
        if context_name not in self.contexts:
            print(f"❌ Context '{context_name}' not found")
            return False
        
        # Save current context if modified
        current_context = self.get_active_context()
        if current_context and current_context.is_modified:
            print(f"💾 Saving current context '{self.active_context_name}' before switch")
            current_context.save_state()
        
        self.active_context_name = context_name
        print(f"🎯 Switched to context: '{context_name}'")
        return True
    
    def reset_active_context(self):
        """Reset the active context to its original history."""
        context = self.get_active_context()
        if context:
            context.reset_to_original()
            # Save immediately after reset
            context.save_state()
    
    def load_original_histories(self, parse_history_func):
        """Load original histories for all contexts using the provided parse function."""
        for context in self.contexts.values():
            if context.config.history_file and Path(context.config.history_file).exists():
                print(f"📖 Loading original history for '{context.config.name}' from {context.config.history_file}")
                
                # Parse history file
                messages = parse_history_func(context.config.history_file)
                
                # Convert to ConversationTurn objects
                for msg in messages:
                    # Extract character name if present
                    character_name = msg.get("_speaker_name")
                    
                    turn = ConversationTurn(
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=datetime.now(),  # Original timestamps not available
                        status="completed",
                        character=character_name
                    )
                    context.original_history.append(turn)
                
                print(f"   ✅ Loaded {len(context.original_history)} turns")
            else:
                print(f"   ⚠️ No history file for context '{context.config.name}'")
            
            print(f"🔍 DEBUG: Checking character_histories for context '{context.config.name}': {context.config.character_histories}")
            
            # Load character-specific histories for this context
            if context.config.character_histories:
                print(f"📚 Loading character histories for context '{context.config.name}'")
                for char_name, char_history_file in context.config.character_histories.items():
                    if Path(char_history_file).exists():
                        print(f"   📖 Loading {char_name}'s history from {char_history_file}")
                        
                        # Parse character history file
                        char_messages = self._parse_character_history_file(char_history_file, char_name)
                        
                        # Convert to ConversationTurn objects
                        char_turns = []
                        for msg in char_messages:
                            turn = ConversationTurn(
                                role=msg["role"],
                                content=msg["content"],
                                timestamp=datetime.now(),
                                status="completed",
                                character=char_name if msg["role"] == "assistant" else None
                            )
                            char_turns.append(turn)
                        
                        context.character_histories[char_name] = char_turns
                        print(f"      ✅ Loaded {len(char_turns)} turns for {char_name}")
                    else:
                        print(f"      ⚠️ Character history file not found: {char_history_file}")
    
    def load_all_states(self):
        """Load saved states for all contexts."""
        for context in self.contexts.values():
            context.load_state()
    
    def save_all_states(self):
        """Save states for all modified contexts."""
        for context in self.contexts.values():
            if context.is_modified:
                context.save_state()
    
    async def start_auto_save(self):
        """Start the auto-save background task."""
        if self.auto_save_enabled and not self.auto_save_task:
            self.auto_save_task = asyncio.create_task(self._auto_save_loop())
            print(f"🔄 Auto-save started (interval: {self.auto_save_interval}s)")
    
    async def stop_auto_save(self):
        """Stop the auto-save background task."""
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
            self.auto_save_task = None
            print("🛑 Auto-save stopped")
    
    async def _auto_save_loop(self):
        """Background task that periodically saves modified contexts."""
        try:
            while True:
                await asyncio.sleep(self.auto_save_interval)
                
                # Save any modified contexts
                saved_count = 0
                for context in self.contexts.values():
                    if context.is_modified:
                        if context.save_state():
                            saved_count += 1
                
                if saved_count > 0:
                    print(f"🔄 Auto-save: Saved {saved_count} modified context(s)")
                    
        except asyncio.CancelledError:
            # Final save before exiting
            self.save_all_states()
            raise
    
    def get_context_list(self) -> List[Dict[str, Any]]:
        """Get list of all contexts with their metadata."""
        context_list = []
        for name, context in self.contexts.items():
            context_info = {
                'name': name,
                'description': context.config.description,
                'is_active': name == self.active_context_name,
                'is_modified': context.is_modified,
                'turn_count': len(context.current_history),
                'last_save': context.last_save_time.isoformat() if context.last_save_time else None
            }
            context_list.append(context_info)
        return context_list
    
    def export_context(self, context_name: str, export_path: str) -> bool:
        """Export a context's current state to a file."""
        context = self.contexts.get(context_name)
        if not context:
            return False
        
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write in the same format as conversation logs
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(f"# Exported Context: {context_name}\n")
                f.write(f"# Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for turn in context.current_history:
                    speaker = turn.character or ("User" if turn.role == "user" else "Assistant")
                    f.write(f"{speaker}: {turn.content}\n\n")
            
            print(f"📤 Exported context '{context_name}' to {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting context: {e}")
            return False