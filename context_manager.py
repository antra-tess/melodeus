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
    system_active_message: Optional[str] = None  # Message to inject after reset point when context becomes active
    

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

        # Flag to indicate system_active_message should be injected
        # Set to True when context is first loaded or reset
        self.needs_system_message = True
        
    def load_state(self) -> bool:
        """Load persistent state from file."""
        # Preserve character histories that were loaded
        preserved_char_histories = self.character_histories.copy()

        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                # Reconstruct conversation history from saved state
                self.current_history = self._deserialize_history(state_data.get('history', []))

                self.metadata = state_data.get('metadata', {})
                self.last_save_time = datetime.fromisoformat(state_data['last_save'])

                # For dynamic contexts, also load original_history (the reset point)
                if self.metadata.get('is_dynamic') and 'original_history' in state_data:
                    self.original_history = self._deserialize_history(state_data['original_history'])

                # Restore preserved character histories
                self.character_histories = preserved_char_histories

                # Check if system message already exists in loaded history
                # If not, we need to inject it (e.g., for tool instructions)
                has_system_message = any(
                    getattr(turn, 'speaker_name', None) == 'System'
                    for turn in self.current_history
                )
                self.needs_system_message = not has_system_message
                print(f"✅ Loaded state for context '{self.config.name}': {len(self.current_history)} turns")
                if self.needs_system_message:
                    print(f"   📢 Will inject system message (none found in history)")
                if self.metadata.get('is_dynamic'):
                    print(f"   📌 Reset point: {len(self.original_history)} turns")
                if self.character_histories:
                    print(f"   📚 Preserved character histories for: {list(self.character_histories.keys())}")
                return True
            else:
                # No saved state, use original history - need to inject system message
                self.current_history = self.original_history.copy()
                self.needs_system_message = True
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
    
    def _serialize_history(self, history: List['ConversationTurn']) -> List[Dict]:
        """Serialize a list of ConversationTurn objects to JSON-compatible dicts."""
        history_data = []
        for turn in history:
            # Skip deleted messages
            if getattr(turn, 'status', None) == 'deleted':
                continue
            # Handle timestamp - could be datetime or float
            if isinstance(turn.timestamp, datetime):
                timestamp_str = turn.timestamp.isoformat()
            elif isinstance(turn.timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(turn.timestamp).isoformat()
            else:
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
        return history_data

    def _deserialize_history(self, history_data: List[Dict]) -> List['ConversationTurn']:
        """Deserialize JSON history data to ConversationTurn objects."""
        history = []
        for turn_data in history_data:
            turn = ConversationTurn(
                role=turn_data['role'],
                content=turn_data['content'],
                timestamp=datetime.fromisoformat(turn_data['timestamp']),
                status=turn_data.get('status', 'completed'),
                character=turn_data.get('character'),
                speaker_name=turn_data.get('speaker_name'),
                metadata=turn_data.get('metadata', {})
            )
            history.append(turn)
        return history

    def save_state(self) -> bool:
        """Save current state to file."""
        try:
            # Ensure state directory exists
            self.state_dir.mkdir(parents=True, exist_ok=True)

            # Serialize current history
            history_data = self._serialize_history(self.current_history)

            state_data = {
                'context_name': self.config.name,
                'history': history_data,
                'metadata': self.metadata,
                'last_save': datetime.now().isoformat(),
                'original_history_file': self.config.history_file
            }

            # For dynamic contexts, also save original_history (the reset point)
            if self.metadata.get('is_dynamic'):
                state_data['original_history'] = self._serialize_history(self.original_history)
            
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
    
    def reset_to_original(self) -> bool:
        """Reset current history to original loaded history.

        Returns:
            True if reset was performed, False if original history is empty
        """
        if not self.original_history:
            print(f"⚠️ Cannot reset context '{self.config.name}': no original history loaded (history file may not exist)")
            return False

        self.current_history = self.original_history.copy()
        self.is_modified = True
        self.needs_system_message = True  # Re-inject system message after reset
        print(f"🔄 Reset context '{self.config.name}' to original history ({len(self.original_history)} turns)")
        return True

    def inject_system_message(self, tool_instructions: Optional[str] = None):
        """Inject the system_active_message into current history.

        Called after context is activated to inject the configured message
        plus any tool instructions.
        """
        if not self.needs_system_message:
            return

        # Build the message content
        message_parts = []

        # Add configured system_active_message if present
        if self.config.system_active_message:
            message_parts.append(self.config.system_active_message)

        # Add tool instructions if provided
        if tool_instructions:
            message_parts.append(tool_instructions)

        if not message_parts:
            self.needs_system_message = False
            return

        # Create the system message turn
        system_turn = ConversationTurn(
            role="user",
            content="System: " + "\n\n".join(message_parts),
            timestamp=datetime.now(),
            status="completed",
            character=None,
            speaker_name="System"
        )

        # Insert after original history (at the current position after reset)
        self.current_history.append(system_turn)
        self.is_modified = True
        self.needs_system_message = False
        print(f"📢 Injected system message into context '{self.config.name}'")
    
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

    # File for storing dynamically created contexts
    DYNAMIC_CONTEXTS_FILE = "dynamic_contexts.json"

    def __init__(self, contexts_config: List[Dict[str, Any]], state_dir: str = "./context_states"):
        self.state_dir = Path(state_dir)
        self.contexts: Dict[str, ConversationContext] = {}
        self.active_context_name: Optional[str] = None
        self.dynamic_contexts_file = self.state_dir / self.DYNAMIC_CONTEXTS_FILE

        # Auto-save configuration
        self.auto_save_enabled = True
        self.auto_save_interval = 30  # seconds
        self.auto_save_task: Optional[asyncio.Task] = None

        # Initialize contexts from config
        for ctx_config in contexts_config:
            print(f"🔧 DEBUG: Creating context '{ctx_config['name']}' with system_active_message: {repr(ctx_config.get('system_active_message'))[:80] if ctx_config.get('system_active_message') else 'None'}")
            config = ContextConfig(
                name=ctx_config['name'],
                history_file=ctx_config['history_file'],
                description=ctx_config.get('description'),
                metadata=ctx_config.get('metadata', {}),
                character_histories=ctx_config.get('character_histories'),
                system_active_message=ctx_config.get('system_active_message')
            )
            context = ConversationContext(config, self.state_dir)
            self.contexts[config.name] = context

        # Load dynamically created contexts
        self._load_dynamic_contexts()

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
            # Only save if reset succeeded (original history exists)
            if context.reset_to_original():
                context.save_state()
            else:
                print(f"⚠️ Reset failed - not saving empty state")
    
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

    def _load_dynamic_contexts(self):
        """Load dynamically created contexts from storage."""
        try:
            if not self.dynamic_contexts_file.exists():
                return

            with open(self.dynamic_contexts_file, 'r', encoding='utf-8') as f:
                dynamic_configs = json.load(f)

            for ctx_data in dynamic_configs:
                name = ctx_data['name']
                if name in self.contexts:
                    print(f"⚠️ Dynamic context '{name}' already exists, skipping")
                    continue

                config = ContextConfig(
                    name=name,
                    history_file=ctx_data.get('history_file', ''),
                    description=ctx_data.get('description'),
                    metadata=ctx_data.get('metadata', {}),
                    character_histories=None  # Dynamic contexts don't inherit character histories
                )
                context = ConversationContext(config, self.state_dir)

                # For dynamic contexts, original_history is stored in state file
                # and loaded via load_state(), not from history_file
                self.contexts[name] = context
                print(f"📂 Loaded dynamic context: '{name}'")

        except Exception as e:
            print(f"❌ Error loading dynamic contexts: {e}")

    def _save_dynamic_contexts(self):
        """Save dynamic context definitions to storage."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)

            # Collect all dynamic contexts (those with 'is_dynamic' in metadata)
            dynamic_configs = []
            for name, context in self.contexts.items():
                if context.metadata.get('is_dynamic'):
                    dynamic_configs.append({
                        'name': name,
                        'history_file': context.config.history_file,
                        'description': context.config.description,
                        'metadata': context.metadata,
                        'forked_from': context.metadata.get('forked_from'),
                        'forked_at': context.metadata.get('forked_at')
                    })

            with open(self.dynamic_contexts_file, 'w', encoding='utf-8') as f:
                json.dump(dynamic_configs, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {len(dynamic_configs)} dynamic context(s)")

        except Exception as e:
            print(f"❌ Error saving dynamic contexts: {e}")

    def _generate_unique_name(self, base_name: str) -> str:
        """Generate a unique context name by appending/incrementing a suffix."""
        # Check if base_name already has a numeric suffix
        import re
        match = re.match(r'^(.+)_(\d+)$', base_name)
        if match:
            base = match.group(1)
            start_num = int(match.group(2)) + 1
        else:
            base = base_name
            start_num = 2

        # Find the next available number
        for i in range(start_num, 1000):
            new_name = f"{base}_{i}"
            if new_name not in self.contexts:
                return new_name

        # Fallback with timestamp
        return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def duplicate_context(self, source_name: str = None, new_name: str = None) -> Optional[str]:
        """Duplicate a context, using current state as the new context's original history.

        Args:
            source_name: Name of context to duplicate (default: active context)
            new_name: Name for new context (default: auto-generated)

        Returns:
            Name of new context, or None if failed
        """
        # Use active context if not specified
        if source_name is None:
            source_name = self.active_context_name

        if source_name not in self.contexts:
            print(f"❌ Source context '{source_name}' not found")
            return None

        source_context = self.contexts[source_name]

        # Generate unique name if not provided
        if new_name is None:
            new_name = self._generate_unique_name(source_name)

        if new_name in self.contexts:
            print(f"❌ Context '{new_name}' already exists")
            return None

        # Create new context config
        config = ContextConfig(
            name=new_name,
            history_file='',  # Dynamic contexts don't have history files
            description=f"Forked from '{source_name}' at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            metadata={'is_dynamic': True, 'forked_from': source_name, 'forked_at': datetime.now().isoformat()},
            character_histories=None  # Don't inherit character histories
        )

        # Create new context
        new_context = ConversationContext(config, self.state_dir)

        # Copy current history as BOTH original and current
        # This means reset will return to this state
        new_context.original_history = [
            ConversationTurn(
                role=turn.role,
                content=turn.content,
                timestamp=turn.timestamp,
                status=turn.status,
                character=turn.character,
                speaker_id=turn.speaker_id,
                speaker_name=turn.speaker_name,
                metadata=turn.metadata.copy() if turn.metadata else {}
            )
            for turn in source_context.current_history
        ]
        new_context.current_history = new_context.original_history.copy()
        new_context.metadata = config.metadata

        # Add to contexts
        self.contexts[new_name] = new_context

        # Save the new context's state
        new_context.save_state()

        # Save dynamic contexts registry
        self._save_dynamic_contexts()

        print(f"✅ Created new context '{new_name}' from '{source_name}' ({len(new_context.current_history)} turns)")
        return new_name

    def delete_dynamic_context(self, context_name: str) -> bool:
        """Delete a dynamically created context.

        Args:
            context_name: Name of dynamic context to delete

        Returns:
            True if deleted, False otherwise
        """
        if context_name not in self.contexts:
            print(f"❌ Context '{context_name}' not found")
            return False

        context = self.contexts[context_name]

        # Only allow deletion of dynamic contexts
        if not context.metadata.get('is_dynamic'):
            print(f"❌ Cannot delete non-dynamic context '{context_name}'")
            return False

        # Don't allow deletion of active context
        if context_name == self.active_context_name:
            print(f"❌ Cannot delete active context '{context_name}'")
            return False

        # Remove state file
        try:
            if context.state_file.exists():
                context.state_file.unlink()
        except Exception as e:
            print(f"⚠️ Could not delete state file: {e}")

        # Remove from contexts
        del self.contexts[context_name]

        # Update dynamic contexts registry
        self._save_dynamic_contexts()

        print(f"🗑️ Deleted dynamic context '{context_name}'")
        return True