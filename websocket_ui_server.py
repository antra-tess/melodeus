"""WebSocket server for UI communication with voice system."""
import asyncio
import websockets
import json
import time
import uuid
import os
import sys
import subprocess
from datetime import datetime
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class UIMessage:
    """Base class for UI messages."""
    type: str
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class VoiceUIServer:
    """WebSocket server for voice conversation UI."""
    
    def __init__(self, conversation=None, host='localhost', port=8795):
        self.conversation = conversation
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.logger = logging.getLogger(__name__)
        
        # Track current state for new clients
        interruptions_enabled = False
        director_enabled = False
        director_mode = "off"
        stt_start_enabled = False
        mute_while_speaking = True
        if conversation and hasattr(conversation, 'config'):
            interruptions_enabled = conversation.config.conversation.interruptions_enabled
            director_enabled = conversation.config.conversation.director_enabled
            director_mode = getattr(conversation.config.conversation, 'director_mode',
                "director" if director_enabled else "off")
            stt_start_enabled = getattr(conversation.config.conversation, 'stt_start_enabled', False)
            mute_while_speaking = getattr(conversation.config.conversation, 'mute_while_speaking', True)

        self.current_state = {
            "current_speaker": None,
            "is_speaking": False,
            "is_processing": False,
            "pending_speaker": None,
            "thinking_sound": False,
            "stt_active": stt_start_enabled,  # Reflects actual STT state
            "conversation_active": False,
            "interruptions_enabled": interruptions_enabled,
            "director_enabled": director_enabled,
            "director_mode": director_mode,  # "off", "same_model", "director"
            "last_ai_speaker": None,  # For same_model mode - which AI will respond next
            "mute_while_speaking": mute_while_speaking  # Mute input while AI is speaking
        }
        
    async def start(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client, self.host, self.port
        )
        print(f"🌐 WebSocket UI server started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("🌐 WebSocket UI server stopped")
            
    async def handle_client(self, websocket):
        """Handle a new client connection."""
        # Register client
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"🔌 UI client connected from {client_addr}")
        
        try:
            # Send initial state
            await self.send_state_sync(websocket)
            
            # Send contexts list if available
            if self.conversation:
                contexts = self.conversation.get_context_list()
                msg = UIMessage("contexts_list", {"contexts": contexts})
                await websocket.send(msg.to_json())
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"🔌 UI client disconnected from {client_addr}")
            
    async def handle_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming message from UI client."""
        msg_type = message.get("type")
        data = message.get("data", {})
        
        if not self.conversation:
            await self.send_error(websocket, "Voice system not connected")
            return
            
        try:
            if msg_type == "force_interrupt":
                await self.handle_force_interrupt()
                
            elif msg_type == "stt_control":
                action = data.get("action")
                await self.handle_stt_control(action)
                
            elif msg_type == "select_speaker":
                speaker = data.get("speaker")
                await self.handle_select_speaker(speaker)
                
            elif msg_type == "trigger_speaker":
                speaker = data.get("speaker")
                await self.handle_trigger_speaker(speaker)
                
            elif msg_type == "trigger_prepared_statement":
                speaker = data.get("speaker")
                statement_name = data.get("statement_name")
                await self.handle_trigger_prepared_statement(speaker, statement_name)
                
            elif msg_type == "sync_request":
                await self.send_state_sync(websocket)
                
            elif msg_type == "ui_ready":
                # Client is ready, send conversation history
                client_id = data.get("client_id", "unknown")
                print(f"✅ UI client ready: {client_id}")
                await self.send_conversation_history(websocket)
                await self.send_current_state(websocket)
                
            elif msg_type == "edit_message":
                # Handle message edit
                msg_id = data.get("message_id")
                new_text = data.get("new_text")
                await self.handle_edit_message(msg_id, new_text)
                
            elif msg_type == "delete_message":
                # Handle message deletion
                msg_id = data.get("message_id")
                await self.handle_delete_message(msg_id)

            elif msg_type == "reset_fingerprints":
                # Reset voice fingerprint database
                await self.handle_reset_fingerprints()

            elif msg_type == "set_learning_mode":
                # Toggle learning mode for voice fingerprinting
                enabled = data.get("enabled", True)
                await self.handle_set_learning_mode(enabled)

            elif msg_type == "set_fingerprint_threshold":
                # Update confidence threshold for voice fingerprinting
                threshold = data.get("threshold", 0.5)
                await self.handle_set_fingerprint_threshold(threshold)

            elif msg_type == "rename_speaker":
                # Rename a learned speaker
                old_id = data.get("old_id")
                new_name = data.get("new_name")
                await self.handle_rename_speaker(old_id, new_name)

            elif msg_type == "merge_speakers":
                # Merge multiple speakers into one
                speaker_ids = data.get("speaker_ids", [])
                target_name = data.get("target_name")
                await self.handle_merge_speakers(speaker_ids, target_name)

            elif msg_type == "delete_speaker":
                # Delete a single speaker from fingerprint database
                speaker_id = data.get("speaker_id")
                await self.handle_delete_speaker(speaker_id)

            elif msg_type == "send_text_message":
                # Handle text message from UI
                speaker_name = data.get("speaker_name", "USER")
                text = data.get("text", "")
                await self.handle_text_message(speaker_name, text)
                
            elif msg_type == "toggle_interruptions":
                # Toggle interruptions on/off
                enabled = data.get("enabled", False)
                await self.handle_toggle_interruptions(enabled)
                
            elif msg_type == "toggle_director":
                # Toggle director on/off (legacy)
                enabled = data.get("enabled", False)
                await self.handle_toggle_director(enabled)
                
            elif msg_type == "set_director_mode":
                # Set director mode: "off", "same_model", "director"
                mode = data.get("mode", "off")
                await self.handle_set_director_mode(mode)
            
            elif msg_type == "set_last_ai_speaker":
                # Set which AI character will respond next (for same_model mode)
                character = data.get("character")
                await self.handle_set_last_ai_speaker(character)
                
            elif msg_type == "get_contexts":
                # Get list of available contexts
                await self.handle_get_contexts(websocket)
                
            elif msg_type == "switch_context":
                # Switch to a different context
                context_name = data.get("context_name")
                print(f"🔄 Received switch_context request: {context_name}")
                if context_name:
                    await self.handle_switch_context(context_name)
                
            elif msg_type == "reset_context":
                # Reset current context to original history
                await self.handle_reset_context()

            elif msg_type == "duplicate_context":
                # Duplicate current context as a new context
                new_name = data.get("new_name")  # Optional custom name
                await self.handle_duplicate_context(new_name)

            elif msg_type == "delete_context":
                # Delete a dynamic context
                context_name = data.get("context_name")
                if context_name:
                    await self.handle_delete_context(context_name)

            elif msg_type == "restart":
                # Restart the melodeus server
                await self.handle_restart()

            elif msg_type == "set_mute_while_speaking":
                # Toggle mute while AI is speaking
                enabled = data.get("enabled", True)
                await self.handle_set_mute_while_speaking(enabled)

            elif msg_type == "toggle_context_mode":
                # Toggle between local and discord context modes
                await self.handle_toggle_context_mode()

            elif msg_type == "save_discord_context":
                # Save current Discord context as a local file context
                new_name = data.get("new_name")
                await self.handle_save_discord_context(new_name)

            else:
                await self.send_error(websocket, f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling {msg_type}: {e}")
            await self.send_error(websocket, str(e))
            
    async def handle_force_interrupt(self):
        """Handle force interrupt request."""
        if not self.conversation:
            return
            
        print("🛑 Force interrupt requested by UI")
        
        # Stop TTS and clear audio queue
        if hasattr(self.conversation, 'tts'):
            await self.conversation.tts.interrupt()
            
        # Mark current processing as interrupted if needed
        if hasattr(self.conversation.state, 'current_processing_turn') and self.conversation.state.current_processing_turn:
            self.conversation.state.current_processing_turn.status = "interrupted"
            
        # Cancel current LLM task
        if hasattr(self.conversation.state, 'current_llm_task'):
            if self.conversation.state.current_llm_task and not self.conversation.state.current_llm_task.done():
                self.conversation.state.current_llm_task.cancel()
                # Wait for cancellation
                try:
                    await asyncio.wait_for(self.conversation.state.current_llm_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self.conversation.state.current_llm_task = None
                
        # Clear all processing state
        if hasattr(self.conversation.state, 'is_speaking'):
            self.conversation.state.is_speaking = False
        if hasattr(self.conversation.state, 'is_processing_llm'):
            self.conversation.state.is_processing_llm = False
        if hasattr(self.conversation.state, 'current_processing_turn'):
            self.conversation.state.current_processing_turn = None
            
        print("🔄 All AI processing interrupted and state cleared")
                
        # Stop thinking sound
        if hasattr(self.conversation, 'thinking_sound'):
            await self.conversation.thinking_sound.stop()
        
        # If TTS was interrupted, send UI update with truncated text
        if hasattr(self.conversation, 'tts') and hasattr(self.conversation.tts, 'current_session'):
            session = self.conversation.tts.current_session
            if session:
                # Get the truncated text from whisper
                truncated_text = self.conversation.tts.get_spoken_text_heuristic().strip()
                if truncated_text and hasattr(self.conversation.state, 'current_ui_session_id'):
                    # Use the stored UI session ID
                    ui_session_id = self.conversation.state.current_ui_session_id
                    
                    # Send correction to UI
                    await self.broadcast(UIMessage(
                        type="ai_stream_correction",
                        data={
                            "session_id": ui_session_id,
                            "corrected_text": truncated_text,
                            "was_interrupted": True
                        }
                    ))
                    print(f"🔄 Sent UI correction for interrupted message: {len(truncated_text)} chars")
        
        # Increment processing generation to cancel any ongoing processing
        if hasattr(self.conversation, '_processing_generation'):
            self.conversation._processing_generation += 1
            
        # Also increment director generation to cancel any director requests
        if hasattr(self.conversation, '_director_generation'):
            self.conversation._director_generation += 1
            
        # Cancel any pending processing task
        if hasattr(self.conversation, '_processing_task'):
            if self.conversation._processing_task and not self.conversation._processing_task.done():
                self.conversation._processing_task.cancel()
        
        # Update and broadcast state
        self.current_state.update({
            "current_speaker": None,
            "is_speaking": False,
            "is_processing": False,
            "thinking_sound": False
        })
        
        await self.broadcast_speaker_status()
        
    async def handle_stt_control(self, action: str):
        """Handle STT pause/resume."""
        if not self.conversation or not hasattr(self.conversation, 'stt'):
            return

        if action == "pause":
            await self.conversation.stt.pause()
            self.current_state["stt_active"] = False
            self.conversation.state.stt_enabled = False
            print("⏸️ STT paused by UI")
        elif action == "resume":
            await self.conversation.stt.resume()
            self.current_state["stt_active"] = True
            self.conversation.state.stt_enabled = True
            print("▶️ STT resumed by UI")

        await self.broadcast_system_status()
        
    async def handle_select_speaker(self, speaker: str):
        """Handle manual speaker selection."""
        if not self.conversation:
            return
            
        print(f"🎭 UI requested speaker: {speaker}")
        # TODO: Implement manual speaker selection
        # This would bypass director and force a specific speaker
        
    async def handle_trigger_speaker(self, speaker: str):
        """Handle triggering a specific speaker to respond."""
        if not self.conversation:
            return

        print(f"🎯 handle_trigger_speaker: {speaker}, discord_mode={self.conversation.is_discord_mode()}")

        # Check if in Discord mode - route to Discord instead of local LLM
        if self.conversation.is_discord_mode():
            # Get the character's discord_user_id
            char_config = self.conversation.character_manager.get_character_config(speaker)
            discord_user_id = None
            if char_config:
                discord_user_id = getattr(char_config, 'discord_user_id', None)

            if discord_user_id:
                print(f"🎯 Triggering Discord bot for {speaker} (ID: {discord_user_id})")
                # Post a mention to Discord to trigger the bot
                if self.conversation.relay_manager and self.conversation.relay_manager._webhook_poster:
                    # Get channel from relay config
                    channel_id = None
                    if self.conversation.config.relay and self.conversation.config.relay.channels:
                        channel_id = self.conversation.config.relay.channels[0]
                    if channel_id:
                        await self.conversation.relay_manager._webhook_poster.post_mention(
                            user_id=discord_user_id,
                            channel_id=channel_id
                        )
                    else:
                        print("⚠️ No Discord channel configured")
            else:
                print(f"⚠️ Character {speaker} has no discord_user_id configured")
            return

        # Validate that the speaker exists (for local mode)
        if not self.conversation.character_manager.get_character_config(speaker):
            print(f"❌ Cannot trigger unknown character: {speaker}")
            await self.broadcast_error(f"Unknown character: {speaker}")
            return

        await self.conversation._get_llm_output(speaker)
        return
            
        # Prevent duplicate triggers
        import time
        current_time = time.time()
        
        if hasattr(self, '_last_trigger_time') and hasattr(self, '_last_trigger_speaker'):
            if (self._last_trigger_speaker == speaker and 
                current_time - self._last_trigger_time < 1.0):  # Within 1 second
                print(f"⚠️ Ignoring duplicate trigger for {speaker} (too soon)")
                return
        
        self._last_trigger_speaker = speaker
        self._last_trigger_time = current_time
            
        print(f"🎯 UI triggered speaker: {speaker}")
        
        # First check if we need to interrupt ongoing processes
        interrupted = False
        
        # Check if TTS is currently playing
        if hasattr(self.conversation, 'tts') and self.conversation.tts.is_currently_playing():
            print("🚫 Cancelling previous TTS playback")
            # Stop TTS and get the spoken content
            await self.conversation.tts.interrupt()
            interrupted = True
            
        # Check if LLM is processing
        if hasattr(self.conversation.state, 'is_processing_llm') and self.conversation.state.is_processing_llm:
            print("🚫 Cancelling previous LLM generation")
            # Mark current processing as interrupted
            if hasattr(self.conversation.state, 'current_processing_turn') and self.conversation.state.current_processing_turn:
                self.conversation.state.current_processing_turn.status = "interrupted"
            interrupted = True
            
        # Cancel any ongoing LLM task
        if hasattr(self.conversation.state, 'current_llm_task') and self.conversation.state.current_llm_task and not self.conversation.state.current_llm_task.done():
            print("🚫 Cancelling previous processing task")
            self.conversation.state.current_llm_task.cancel()
            # Wait for cancellation
            try:
                await asyncio.wait_for(self.conversation.state.current_llm_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self.conversation.state.current_llm_task = None
            
        # Clear state if we interrupted
        if interrupted:
            self.conversation.state.is_speaking = False
            self.conversation.state.is_processing_llm = False
            self.conversation.state.current_processing_turn = None
            print(f"🔄 All AI processing interrupted and state cleared")
        
        # Set the next speaker globally
        async with self.conversation.state.speaker_lock:
            self.conversation.state.next_speaker = speaker
            print(f"📢 Set global next speaker: {speaker} (no system message added)")
        
        # Mark any pending user utterances as completed before processing
        pending_count = 0
        for turn in self.conversation.state.conversation_history:
            if turn.role == "user" and turn.status == "pending":
                turn.status = "completed"
                pending_count += 1
        if pending_count > 0:
            print(f"✅ Marked {pending_count} pending user utterances as completed")
        
        # If we interrupted, wait a bit for state to clear
        if interrupted:
            await asyncio.sleep(0.1)
        
        # Store the task to prevent duplicate processing
        if hasattr(self, '_trigger_task') and self._trigger_task and not self._trigger_task.done():
            print("⚠️ Previous trigger task still running, skipping")
            return
            
        # Create a reference turn for metadata without adding to history
        from unified_voice_conversation_config import ConversationTurn
        from datetime import datetime
        
        reference_turn = ConversationTurn(
            role="system",
            content="",  # Empty content
            timestamp=datetime.now(),
            status="completed",
            metadata={
                "is_manual_trigger": True,
                "triggered_speaker": speaker
            }
        )
        
        # Directly process with character LLM using empty input
        self._trigger_task = asyncio.create_task(
            self.conversation._process_with_character_llm("", reference_turn)
        )
        
        # Broadcast status update
        await self.broadcast_speaker_status(
            is_processing=True,
            pending_speaker=speaker
        )
    
    async def handle_trigger_prepared_statement(self, speaker: str, statement_name: str):
        """Handle triggering a prepared statement for a specific speaker."""
        if not self.conversation:
            return
            
        print(f"📜 UI triggered prepared statement '{statement_name}' for speaker: {speaker}")
        
        # Check if we need to interrupt ongoing processes (similar to regular trigger)
        interrupted = False
        
        # Check if TTS is currently playing
        if hasattr(self.conversation, 'tts') and self.conversation.tts.is_currently_playing():
            print("🚫 Cancelling previous TTS playback")
            await self.conversation.tts.interrupt()
            interrupted = True
            
        # Check if LLM is processing
        if hasattr(self.conversation.state, 'is_processing_llm') and self.conversation.state.is_processing_llm:
            print("🚫 Cancelling previous LLM generation")
            if hasattr(self.conversation.state, 'current_processing_turn') and self.conversation.state.current_processing_turn:
                self.conversation.state.current_processing_turn.status = "interrupted"
            interrupted = True
            
        # Cancel any ongoing LLM task
        if hasattr(self.conversation.state, 'current_llm_task') and self.conversation.state.current_llm_task and not self.conversation.state.current_llm_task.done():
            print("🚫 Cancelling previous processing task")
            self.conversation.state.current_llm_task.cancel()
            try:
                await asyncio.wait_for(self.conversation.state.current_llm_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self.conversation.state.current_llm_task = None
            
        # Clear state if we interrupted
        if interrupted:
            self.conversation.state.is_speaking = False
            self.conversation.state.is_processing_llm = False
            self.conversation.state.current_processing_turn = None
            print(f"🔄 All AI processing interrupted and state cleared")
        
        # Set the next speaker globally
        async with self.conversation.state.speaker_lock:
            self.conversation.state.next_speaker = speaker
            print(f"📢 Set global next speaker: {speaker} (for prepared statement)")
        
        # If we interrupted, wait a bit for state to clear
        if interrupted:
            await asyncio.sleep(0.1)
        
        # Create a reference turn for metadata
        from unified_voice_conversation_config import ConversationTurn
        from datetime import datetime
        
        reference_turn = ConversationTurn(
            role="system",
            content="",
            timestamp=datetime.now(),
            status="completed",
            metadata={
                "is_prepared_statement": True,
                "statement_name": statement_name,
                "triggered_speaker": speaker
            }
        )
        
        # Process with character LLM, passing the prepared statement name
        task = asyncio.create_task(
            self.conversation._process_with_character_llm("", reference_turn, prepared_statement_name=statement_name)
        )
        
        # Broadcast status update
        await self.broadcast_speaker_status(
            is_processing=True,
            pending_speaker=speaker
        )
        
    async def send_state_sync(self, websocket):
        """Send current state to a client."""
        # Send speaker status
        await self.send_to_client(websocket, UIMessage(
            type="speaker_status",
            data=self.current_state
        ))
        
        # Send system status
        await self.send_to_client(websocket, UIMessage(
            type="system_status",
            data={
                "stt_active": self.current_state["stt_active"],
                "tts_active": bool(self.current_state["is_speaking"]),
                "conversation_active": self.current_state["conversation_active"],
                "current_generation": getattr(self.conversation, '_processing_generation', 0) if self.conversation else 0,
                "director_generation": getattr(self.conversation, '_director_generation', 0) if self.conversation else 0
            }
        ))
        
        # Send available characters
        if self.conversation and hasattr(self.conversation, 'character_manager') and self.conversation.character_manager:
            characters = []
            for char_name in self.conversation.character_manager.characters.keys():
                char_config = self.conversation.character_manager.get_character_config(char_name)
                char_data = {
                    "name": char_name,
                    "model": char_config.llm_model if char_config else "unknown",
                    "active": self.conversation.character_manager.active_character == char_name
                }
                
                # Add prepared statements if available
                if char_config and hasattr(char_config, 'prepared_statements') and char_config.prepared_statements:
                    char_data["prepared_statements"] = list(char_config.prepared_statements.keys())
                    print(f"   📜 {char_name} has prepared statements: {char_data['prepared_statements']}")
                
                characters.append(char_data)
            
            await self.send_to_client(websocket, UIMessage(
                type="available_characters",
                data={"characters": characters}
            ))

        # Send context mode state
        if self.conversation:
            context_mode = self.conversation.get_context_mode()
            await self.send_to_client(websocket, UIMessage(
                type="context_mode_change",
                data={"mode": context_mode}
            ))

    async def send_to_client(self, client: websockets.WebSocketServerProtocol, message: UIMessage):
        """Send message to specific client."""
        try:
            await client.send(message.to_json())
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error sending to client: {e}")
            
    async def broadcast(self, message: UIMessage):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, message) for client in self.clients],
                return_exceptions=True
            )
            
    async def send_error(self, client, error_message: str, severity="error"):
        """Send error message to client."""
        error_msg = UIMessage(
            type="error",
            data={
                "severity": severity,
                "message": error_message,
                "timestamp": time.time()
            }
        )
        
        if client is None:
            # Broadcast to all clients if no specific client
            await self.broadcast(error_msg)
        else:
            await self.send_to_client(client, error_msg)
        
    # Convenience methods for common broadcasts
    
    async def broadcast_speaker_status(self, current_speaker=None, is_speaking=None, 
                                      is_processing=None, pending_speaker=None, 
                                      thinking_sound=None):
        """Broadcast speaker status update."""
        # Update internal state
        if current_speaker is not None:
            self.current_state["current_speaker"] = current_speaker
        if is_speaking is not None:
            self.current_state["is_speaking"] = is_speaking
        if is_processing is not None:
            self.current_state["is_processing"] = is_processing
        if pending_speaker is not None:
            self.current_state["pending_speaker"] = pending_speaker
        if thinking_sound is not None:
            self.current_state["thinking_sound"] = thinking_sound
            
        await self.broadcast(UIMessage(
            type="speaker_status",
            data=self.current_state
        ))
        
    async def broadcast_transcription(self, speaker: str, text: str,
                                    is_final: bool = False, is_interim: bool = False, is_edit: bool = False,
                                    message_id = None, has_audio: bool = None):
        """Broadcast transcription update."""
        message_data = {
            "speaker": speaker,
            "text": text,
            "is_final": is_final,
            "is_interim": is_interim,
            "timestamp": time.time()
        }

        # Add message ID for final messages so they can be edited/deleted
        if is_final and not is_interim:
            message_data["message_id"] = str(uuid.uuid4()) if not message_id else message_id
            # Check if audio archiving is enabled (has_audio flag for replay button)
            if has_audio is None and self.conversation:
                has_audio = getattr(self.conversation.config.conversation, 'audio_archive_enabled', False)
            message_data["has_audio"] = has_audio if has_audio else False
        
        if is_edit:
            # Broadcast the edit to all clients
            update_data = {
                "id": message_id,
                "new_text": text,
                "edited": True
            }
            await self.broadcast(UIMessage("message_edited", update_data))
        else:
            await self.broadcast(UIMessage(
                type="transcription",
                data=message_data
            ))
        
    async def broadcast_ai_stream(self, speaker: str, text: str,
                                 is_complete: bool = False, session_id: str = "",
                                 message_id: str = None, has_audio: bool = None):
        """Broadcast AI response stream."""
        message_data = {
            "speaker": speaker,
            "text": text,
            "is_complete": is_complete,
            "session_id": session_id,
            "timestamp": time.time()
        }

        # Add message ID for completed messages so they can be edited/deleted
        # Use provided message_id if available, otherwise generate UUID (fallback)
        if is_complete:
            message_data["message_id"] = message_id if message_id else str(uuid.uuid4())
            # Check if audio archiving is enabled (has_audio flag for replay button)
            if has_audio is None and self.conversation:
                has_audio = getattr(self.conversation.config.conversation, 'audio_archive_enabled', False)
            message_data["has_audio"] = has_audio if has_audio else False

        await self.broadcast(UIMessage(
            type="ai_stream",
            data=message_data
        ))
    
    async def broadcast_tts_progress(self, speaker: str, spoken_text: str, char_index: int, session_id: str = ""):
        """Broadcast TTS playback progress (text revealed so far)."""
        await self.broadcast(UIMessage(
            type="tts_progress",
            data={
                "speaker": speaker,
                "spoken_text": spoken_text,
                "char_index": char_index,
                "session_id": session_id,
                "timestamp": time.time()
            }
        ))
        
    async def broadcast_system_status(self):
        """Broadcast system status update."""
        await self.broadcast(UIMessage(
            type="system_status",
            data={
                "stt_active": self.current_state["stt_active"],
                "tts_active": bool(self.current_state["is_speaking"]),
                "conversation_active": self.current_state["conversation_active"],
                "current_generation": getattr(self.conversation, '_processing_generation', 0) if self.conversation else 0,
                "director_generation": getattr(self.conversation, '_director_generation', 0) if self.conversation else 0,
                "timestamp": time.time()
            }
        ))
    
    async def broadcast_audio_levels(self, input_level: float = 0.0, output_level: float = 0.0):
        """Broadcast audio input/output levels for UI meters."""
        await self.broadcast(UIMessage(
            type="audio_levels",
            data={
                "input": input_level,
                "output": output_level
            }
        ))
        
    async def broadcast_conversation_update(self, turn_data: Dict[str, Any]):
        """Broadcast conversation history update."""
        await self.broadcast(UIMessage(
            type="conversation_update",
            data={"turn": turn_data}
        ))
    
    async def send_conversation_history(self, websocket):
        """Send full conversation history to a client."""
        if not self.conversation:
            return
            
        history = []
        for i, turn in enumerate(self.conversation.state.conversation_history):
            # Skip deleted messages
            if getattr(turn, 'status', None) == 'deleted':
                continue
            # Convert each turn to a format suitable for UI
            # Handle timestamp serialization
            timestamp = getattr(turn, 'timestamp', time.time())
            # Convert datetime to timestamp if needed
            if hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()
            elif isinstance(timestamp, str):
                # If it's already a string, keep it
                pass
            elif not isinstance(timestamp, (int, float)):
                # Fallback to current time if timestamp is invalid
                timestamp = time.time()
                
            turn_data = {
                "id": getattr(turn, 'id', None) or f"msg_{i}",  # Use turn.id if set, else fallback
                "role": turn.role,
                "content": turn.content,
                "speaker_name": getattr(turn, 'speaker_name', None),
                "character": getattr(turn, 'character', None),
                "timestamp": timestamp,
                "status": getattr(turn, 'status', 'completed'),
                "editable": True  # All messages are editable/deletable
            }
            history.append(turn_data)
        
        msg = UIMessage("conversation_history", {"history": history})
        await websocket.send(msg.to_json())
        print(f"📜 Sent {len(history)} history messages to client")
    
    async def send_current_state(self, websocket):
        """Send current system state to a client."""
        msg = UIMessage("state_sync", self.current_state)
        await websocket.send(msg.to_json())
        
        # Also send learned speakers
        await self.send_learned_speakers(websocket)
    
    async def send_learned_speakers(self, websocket):
        """Send list of learned speakers to a client."""
        if not self.conversation:
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    speakers = fingerprinter.get_known_speakers()
                    msg = UIMessage("learned_speakers", {"speakers": speakers})
                    await websocket.send(msg.to_json())
                    # Also send fingerprint settings
                    await self.send_fingerprint_settings(websocket)
        except Exception as e:
            print(f"⚠️ Error sending learned speakers: {e}")

    async def send_fingerprint_settings(self, websocket):
        """Send fingerprint settings to a client."""
        if not self.conversation:
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    msg = UIMessage("fingerprint_settings", {
                        "learning_mode": fingerprinter.learning_mode,
                        "threshold": fingerprinter.confidence_threshold
                    })
                    await websocket.send(msg.to_json())
        except Exception as e:
            print(f"⚠️ Error sending fingerprint settings: {e}")
    
    async def handle_edit_message(self, msg_id: str, new_text: str):
        """Handle message edit request."""
        if not self.conversation or not msg_id or not new_text:
            return

        try:
            # Extract index from msg_id (format: "msg_123")
            index = int(msg_id.split('_')[1])

            if 0 <= index < len(self.conversation.state.conversation_history):
                turn = self.conversation.state.conversation_history[index]
                # Update the message content for any role
                old_text = turn.content
                turn.content = new_text
                print(f"✏️ Edited message {msg_id} ({turn.role}): '{old_text}' → '{new_text}'")

                # If this turn has Discord message info, edit it there too
                if turn.metadata and turn.metadata.get("discord_message_id"):
                    if hasattr(self.conversation, 'relay_manager') and self.conversation.relay_manager:
                        webhook_poster = self.conversation.relay_manager._webhook_poster
                        if webhook_poster:
                            success = await webhook_poster.edit_webhook_message(
                                message_id=turn.metadata["discord_message_id"],
                                webhook_id=turn.metadata["discord_webhook_id"],
                                webhook_token=turn.metadata["discord_webhook_token"],
                                new_content=new_text
                            )
                            if success:
                                print(f"   ✅ Also edited Discord message {turn.metadata['discord_message_id']}")
                            else:
                                print(f"   ⚠️ Failed to edit Discord message (local edit kept)")

                # Update conversation log file if it exists
                if hasattr(self.conversation, '_log_conversation_turn'):
                    # Re-log the edited turn
                    self.conversation._log_conversation_turn(turn.role, new_text)

                # Broadcast the edit to all clients
                update_data = {
                    "id": msg_id,
                    "new_text": new_text,
                    "edited": True
                }
                await self.broadcast(UIMessage("message_edited", update_data))

                # Save immediately to persist the edit
                if hasattr(self.conversation, '_save_conversation_state'):
                    self.conversation._save_conversation_state()
            else:
                await self.send_error(None, f"Invalid message ID: {msg_id}")

        except Exception as e:
            self.logger.error(f"Error editing message: {e}")
            await self.send_error(None, str(e))
    
    async def handle_delete_message(self, msg_id: str):
        """Handle message deletion request."""
        if not self.conversation or not msg_id:
            return
            
        try:
            # Extract index from msg_id
            parts = msg_id.split('_')
            if len(parts) != 2 or parts[0] != 'msg':
                print(f"⚠️ Invalid message ID format: {msg_id}")
                await self.send_error(None, f"Invalid message ID format: {msg_id}")
                return
                
            index = int(parts[1])
            history_length = len(self.conversation.state.conversation_history)
            
            print(f"🔍 Attempting to delete message {msg_id}: index={index}, history_length={history_length}")
            
            if 0 <= index < history_length:
                # Mark as deleted but don't remove (to preserve indices)
                turn = self.conversation.state.conversation_history[index]
                turn.status = "deleted"
                print(f"🗑️ Deleted message {msg_id}")

                # Broadcast deletion to all clients
                await self.broadcast(UIMessage("message_deleted", {"id": msg_id}))

                # Save immediately to persist the deletion
                if hasattr(self.conversation, '_save_conversation_state'):
                    self.conversation._save_conversation_state()
            else:
                print(f"⚠️ Index {index} out of range for history length {history_length}")
                await self.send_error(None, f"Invalid message index: {index} (history has {history_length} messages)")
                
        except Exception as e:
            self.logger.error(f"Error deleting message: {e}")
            await self.send_error(None, str(e))

    async def handle_reset_fingerprints(self):
        """Reset voice fingerprint database."""
        if not self.conversation:
            return

        try:
            # Get the fingerprinter from STT module
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    count = fingerprinter.reset_all()
                    print(f"🔄 Reset fingerprints via UI - cleared {count} profiles")
                    await self.broadcast(UIMessage("fingerprints_reset", {"cleared_count": count}))
                else:
                    print("⚠️ No voice fingerprinter available")
            else:
                print("⚠️ Voice fingerprinting not enabled")
        except Exception as e:
            self.logger.error(f"Error resetting fingerprints: {e}")
            await self.send_error(None, str(e))

    async def handle_set_learning_mode(self, enabled: bool):
        """Toggle learning mode for voice fingerprinting."""
        if not self.conversation:
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    fingerprinter.learning_mode = enabled
                    print(f"🎓 Learning mode {'enabled' if enabled else 'disabled'} via UI")
                    await self.broadcast(UIMessage("fingerprint_settings", {
                        "learning_mode": enabled,
                        "threshold": fingerprinter.confidence_threshold
                    }))
        except Exception as e:
            self.logger.error(f"Error setting learning mode: {e}")
            await self.send_error(None, str(e))

    async def handle_set_fingerprint_threshold(self, threshold: float):
        """Update confidence threshold for voice fingerprinting."""
        if not self.conversation:
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    fingerprinter.confidence_threshold = threshold
                    print(f"🎚️ Fingerprint threshold set to {threshold:.2f} via UI")
                    await self.broadcast(UIMessage("fingerprint_settings", {
                        "learning_mode": fingerprinter.learning_mode,
                        "threshold": threshold
                    }))
        except Exception as e:
            self.logger.error(f"Error setting threshold: {e}")
            await self.send_error(None, str(e))

    async def handle_rename_speaker(self, old_id: str, new_name: str):
        """Rename a learned speaker (e.g., 'User 1' -> 'antra_tessera')."""
        if not self.conversation:
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    success = fingerprinter.rename_speaker(old_id, new_name)
                    if success:
                        # Convert internal ID to display name for history matching
                        display_old_name = old_id
                        if old_id.startswith("unknown_speaker_"):
                            display_old_name = "User " + old_id.replace("unknown_speaker_", "")
                        
                        # Update conversation history
                        updated_count = 0
                        if hasattr(self.conversation, 'state') and hasattr(self.conversation.state, 'conversation_history'):
                            for turn in self.conversation.state.conversation_history:
                                if hasattr(turn, 'speaker_name') and turn.speaker_name:
                                    if turn.speaker_name.lower() == display_old_name.lower():
                                        turn.speaker_name = new_name
                                        updated_count += 1
                        
                        print(f"📝 Updated {updated_count} history entries: '{display_old_name}' -> '{new_name}'")
                        
                        # Update detected speakers set
                        if hasattr(self.conversation, 'detected_speakers'):
                            if display_old_name in self.conversation.detected_speakers:
                                self.conversation.detected_speakers.discard(display_old_name)
                                self.conversation.detected_speakers.add(new_name)
                        
                        # Get updated list of speakers
                        speakers = fingerprinter.get_known_speakers()
                        await self.broadcast(UIMessage("speaker_renamed", {
                            "old_id": old_id,
                            "new_name": new_name,
                            "known_speakers": speakers
                        }))
                    else:
                        await self.send_error(None, f"Failed to rename speaker '{old_id}'")
                else:
                    print("⚠️ No voice fingerprinter available")
            else:
                print("⚠️ Voice fingerprinting not enabled")
        except Exception as e:
            self.logger.error(f"Error renaming speaker: {e}")
            await self.send_error(None, str(e))

    async def handle_merge_speakers(self, speaker_ids: list, target_name: str):
        """Merge multiple learned speakers into one."""
        if not self.conversation or not speaker_ids or len(speaker_ids) < 2 or not target_name:
            await self.send_error(None, "Need at least 2 speakers and a target name to merge")
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    success = fingerprinter.merge_speakers(speaker_ids, target_name)
                    if success:
                        # Convert internal IDs to display names for history matching
                        display_names = []
                        for sid in speaker_ids:
                            if sid.startswith("unknown_speaker_"):
                                display_names.append("User " + sid.replace("unknown_speaker_", ""))
                            elif sid.startswith("User "):
                                display_names.append(sid)
                            else:
                                display_names.append(sid)
                        
                        # Update conversation history for all merged speakers
                        updated_count = 0
                        if hasattr(self.conversation, 'state') and hasattr(self.conversation.state, 'conversation_history'):
                            for turn in self.conversation.state.conversation_history:
                                if hasattr(turn, 'speaker_name') and turn.speaker_name:
                                    if turn.speaker_name.lower() in [dn.lower() for dn in display_names]:
                                        turn.speaker_name = target_name
                                        updated_count += 1
                        
                        print(f"📝 Updated {updated_count} history entries to '{target_name}'")
                        
                        # Update detected speakers set
                        if hasattr(self.conversation, 'detected_speakers'):
                            for dn in display_names:
                                self.conversation.detected_speakers.discard(dn)
                            self.conversation.detected_speakers.add(target_name)
                        
                        # Get updated list of speakers
                        speakers = fingerprinter.get_known_speakers()
                        await self.broadcast(UIMessage("speakers_merged", {
                            "merged_ids": speaker_ids,
                            "target_name": target_name,
                            "known_speakers": speakers
                        }))
                    else:
                        await self.send_error(None, f"Failed to merge speakers")
                else:
                    print("⚠️ No voice fingerprinter available")
            else:
                print("⚠️ Voice fingerprinting not enabled")
        except Exception as e:
            self.logger.error(f"Error merging speakers: {e}")
            await self.send_error(None, str(e))

    async def handle_delete_speaker(self, speaker_id: str):
        """Delete a single speaker from the fingerprint database."""
        if not self.conversation or not speaker_id:
            await self.send_error(None, "No speaker ID provided")
            return

        try:
            if hasattr(self.conversation, 'stt') and hasattr(self.conversation.stt, 'voice_fingerprinter'):
                fingerprinter = self.conversation.stt.voice_fingerprinter
                if fingerprinter:
                    success = fingerprinter.delete_speaker(speaker_id)
                    if success:
                        # Get updated list of speakers
                        speakers = fingerprinter.get_known_speakers()
                        await self.broadcast(UIMessage("speaker_deleted", {
                            "speaker_id": speaker_id,
                            "known_speakers": speakers
                        }))
                        print(f"🗑️ Deleted speaker: {speaker_id}")
                    else:
                        await self.send_error(None, f"Failed to delete speaker: {speaker_id}")
                else:
                    print("⚠️ No voice fingerprinter available")
            else:
                print("⚠️ Voice fingerprinting not enabled")
        except Exception as e:
            self.logger.error(f"Error deleting speaker: {e}")
            await self.send_error(None, str(e))

    async def handle_text_message(self, speaker_name: str, text: str):
        """Handle text message from UI."""
        if not self.conversation or not text.strip():
            return

        print(f"💬 Text message from {speaker_name}: {text}")

        # Create a synthetic transcription result
        from dataclasses import dataclass

        @dataclass
        class SyntheticTranscriptionResult:
            text: str
            words: list
            is_final: bool = True
            is_complete_utterance: bool = True
            speaker_name: str = None
            speaker_id: int = None
            timestamp: float = None
            confidence: float = 1.0
            raw_data: dict = None
            message_id: str = None  # For edit tracking
            is_edit: bool = False  # Whether this is editing an existing message
            audio_window_start: int = None  # For audio archiving (not used for text input)
            audio_window_end: int = None  # For audio archiving (not used for text input)

        # Create a result that looks like it came from STT
        result = SyntheticTranscriptionResult(
            text=text,
            words=[],  # Empty words list for text input
            speaker_name=speaker_name,
            speaker_id=None,  # Will be handled by conversation system
            timestamp=time.time(),
            confidence=1.0,
            raw_data={}
        )

        # Add the speaker to detected_speakers so they become a stop sequence
        if hasattr(self.conversation, 'detected_speakers') and speaker_name:
            self.conversation.detected_speakers.add(speaker_name)
            print(f"📝 Added '{speaker_name}' to detected speakers for stop sequences")

        # Process through the conversation system
        if hasattr(self.conversation, '_on_utterance_complete'):
            await self.conversation._on_utterance_complete(result)
        
        # Broadcast the transcription to all UI clients
        # don't need this because it's already broadcasted in on_utterance_complete
        #await self.broadcast_transcription(
        #    speaker=speaker_name,
        #    text=text,
        #    is_final=True
        #)
    
    async def handle_toggle_interruptions(self, enabled: bool):
        """Handle toggling interruptions on/off."""
        print(f"{'🟢' if enabled else '🔴'} Voice interruptions {'enabled' if enabled else 'disabled'}")
        
        # Update our state
        self.current_state["interruptions_enabled"] = enabled
        
        # Update the conversation config if available
        if self.conversation and hasattr(self.conversation, 'config'):
            self.conversation.config.conversation.interruptions_enabled = enabled
        
        # Broadcast the update to all clients
        await self.broadcast(UIMessage(
            type="interruptions_toggled",
            data={"enabled": enabled}
        ))
        
        # Also send a state sync to update UI
        await self.broadcast(UIMessage(
            type="state_sync",
            data=self.current_state
        ))

    async def handle_toggle_context_mode(self):
        """Toggle between local and discord context modes."""
        print(f"🔄 handle_toggle_context_mode called")
        if not self.conversation:
            print("❌ Cannot toggle context mode: conversation not initialized")
            return

        print(f"🔄 relay_manager = {self.conversation.relay_manager}")
        # Toggle the mode
        new_mode = await self.conversation.toggle_context_mode()
        print(f"🔄 Context mode toggled to: {new_mode}")

        # Broadcast the update to all clients
        await self.broadcast(UIMessage(
            type="context_mode_change",
            data={"mode": new_mode}
        ))

    async def handle_get_contexts(self, websocket):
        """Get list of available contexts."""
        if not self.conversation:
            await self.send_error(websocket, "Conversation not initialized")
            return
        
        contexts = self.conversation.get_context_list()
        msg = UIMessage("contexts_list", {"contexts": contexts})
        await websocket.send(msg.to_json())
        print(f"📋 Sent {len(contexts)} contexts to client")
    
    async def handle_switch_context(self, context_name: str):
        """Switch to a different context."""
        if not self.conversation:
            return
        
        try:
            success = await self.conversation.switch_context(context_name)
            if success:
                # Broadcast context switch to all clients
                msg = UIMessage("context_switched", {"context_name": context_name})
                await self.broadcast(msg)
                
                # Send updated conversation history to all clients
                for ws in self.clients:
                    await self.send_conversation_history(ws)
                
                print(f"🔄 Switched to context: {context_name}")
            else:
                await self.broadcast_error(f"Failed to switch to context: {context_name}")
                
        except Exception as e:
            self.logger.error(f"Error switching context: {e}")
            await self.broadcast_error(str(e))
    
    async def handle_reset_context(self):
        """Reset current context to original history."""
        if not self.conversation:
            return
        
        try:
            await self.conversation.reset_context()
            
            # Broadcast context reset to all clients
            msg = UIMessage("context_reset", {})
            await self.broadcast(msg)
            
            # Send updated conversation history to all clients
            for ws in self.clients:
                await self.send_conversation_history(ws)
            
            print("🔄 Reset context to original history")

        except Exception as e:
            self.logger.error(f"Error resetting context: {e}")
            await self.broadcast_error(str(e))

    async def handle_duplicate_context(self, new_name: str = None):
        """Duplicate current context as a new context."""
        if not self.conversation or not self.conversation.context_manager:
            return

        try:
            # Duplicate the active context
            result_name = self.conversation.context_manager.duplicate_context(new_name=new_name)

            if result_name:
                # Broadcast context list update to all clients
                contexts = self.conversation.context_manager.get_context_list()
                msg = UIMessage("contexts_list", {"contexts": contexts})
                await self.broadcast(msg)

                # Notify about the new context
                msg = UIMessage("context_duplicated", {
                    "new_context": result_name,
                    "message": f"Created new context '{result_name}'"
                })
                await self.broadcast(msg)

                print(f"✅ Duplicated context as '{result_name}'")
            else:
                await self.broadcast_error("Failed to duplicate context")

        except Exception as e:
            self.logger.error(f"Error duplicating context: {e}")
            await self.broadcast_error(str(e))

    async def handle_delete_context(self, context_name: str):
        """Delete a dynamic context."""
        if not self.conversation or not self.conversation.context_manager:
            return

        try:
            success = self.conversation.context_manager.delete_dynamic_context(context_name)

            if success:
                # Broadcast context list update to all clients
                contexts = self.conversation.context_manager.get_context_list()
                msg = UIMessage("contexts_list", {"contexts": contexts})
                await self.broadcast(msg)

                msg = UIMessage("context_deleted", {
                    "context_name": context_name,
                    "message": f"Deleted context '{context_name}'"
                })
                await self.broadcast(msg)

                print(f"🗑️ Deleted context '{context_name}'")
            else:
                await self.broadcast_error(f"Cannot delete context '{context_name}'")

        except Exception as e:
            self.logger.error(f"Error deleting context: {e}")
            await self.broadcast_error(str(e))

    async def handle_save_discord_context(self, new_name: str = None):
        """Save current Discord context as a local file context."""
        if not self.conversation or not self.conversation.context_manager:
            await self.broadcast_error("Context manager not available")
            return

        try:
            # Get the active context
            active_context = self.conversation.context_manager.get_active_context()
            if not active_context:
                await self.broadcast_error("No active context")
                return

            # Check if it's a Discord context
            if not active_context.is_discord_context():
                await self.broadcast_error("Current context is not a Discord context")
                return

            # Save as local file context
            result_name = self.conversation.context_manager.save_discord_context_as_file(
                discord_context_name=active_context.config.name,
                new_name=new_name
            )

            if result_name:
                # Broadcast context list update to all clients
                contexts = self.conversation.get_context_list()
                msg = UIMessage("contexts_list", {"contexts": contexts})
                await self.broadcast(msg)

                # Notify about the new context
                msg = UIMessage("discord_context_saved", {
                    "new_context": result_name,
                    "message": f"Saved Discord context as '{result_name}'"
                })
                await self.broadcast(msg)

                print(f"💾 Saved Discord context as '{result_name}'")
            else:
                await self.broadcast_error("Failed to save Discord context")

        except Exception as e:
            self.logger.error(f"Error saving Discord context: {e}")
            await self.broadcast_error(str(e))

    async def handle_restart(self):
        """Restart the melodeus server by spawning a new process and exiting."""
        print("🔄 Restart requested from UI...")

        # Notify all clients
        msg = UIMessage("server_restarting", {"message": "Server is restarting..."})
        await self.broadcast(msg)

        # Give clients a moment to receive the message
        await asyncio.sleep(0.5)

        # Get the directory where melodeus lives
        melodeus_dir = os.path.dirname(os.path.abspath(__file__))
        start_script = os.path.join(melodeus_dir, "start_melodeus.sh")

        if os.path.exists(start_script):
            # Spawn a detached process that waits then restarts
            restart_cmd = f'sleep 2 && cd "{melodeus_dir}" && ./start_melodeus.sh'
            subprocess.Popen(
                restart_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"🔄 Spawned restart process, exiting in 1 second...")
        else:
            print(f"⚠️ Start script not found: {start_script}")
            await self.broadcast_error("Restart script not found")
            return

        # Give the spawn a moment, then exit
        await asyncio.sleep(1)
        print("🔄 Exiting for restart...")
        os._exit(0)

    async def handle_toggle_director(self, enabled: bool):
        """Handle toggling director on/off (legacy - prefer set_director_mode)."""
        # Convert to new mode system
        mode = "director" if enabled else "off"
        await self.handle_set_director_mode(mode)
    
    async def handle_set_director_mode(self, mode: str):
        """Handle setting director mode: 'off', 'same_model', or 'director'."""
        if mode not in ("off", "same_model", "director"):
            print(f"⚠️ Invalid director mode: {mode}")
            return
            
        mode_icons = {"off": "🚫", "same_model": "🔄", "director": "🎭"}
        mode_names = {"off": "Off", "same_model": "Same Model", "director": "Director"}
        print(f"{mode_icons.get(mode, '❓')} Director mode: {mode_names.get(mode, mode)}")
        
        # Update our state
        self.current_state["director_mode"] = mode
        self.current_state["director_enabled"] = (mode != "off")  # Legacy compatibility
        
        # Update the conversation config if available
        if self.conversation and hasattr(self.conversation, 'config'):
            self.conversation.config.conversation.director_mode = mode
            self.conversation.config.conversation.director_enabled = (mode != "off")
            # Save global settings to persist across restarts
            if hasattr(self.conversation, '_save_global_settings'):
                self.conversation._save_global_settings()

        # Broadcast the update to all clients
        await self.broadcast(UIMessage(
            type="director_mode_changed",
            data={"mode": mode}
        ))
        
        # Also broadcast legacy toggle for compatibility
        await self.broadcast(UIMessage(
            type="director_toggled",
            data={"enabled": mode != "off"}
        ))
        
        # Also send a state sync to update UI
        await self.broadcast(UIMessage(
            type="state_sync",
            data=self.current_state
        ))
    
    async def handle_set_last_ai_speaker(self, character: Optional[str]):
        """Set which AI character will respond next (used in same_model mode)."""
        if character:
            print(f"🎤 Next AI speaker set to: {character}")
        else:
            print("🎤 Next AI speaker cleared")
        
        # Update our state
        self.current_state["last_ai_speaker"] = character
        
        # Update the conversation state if available
        if self.conversation and hasattr(self.conversation, 'state'):
            self.conversation.state.last_ai_speaker = character
            self.conversation.state.next_speaker = character  # Also set next_speaker
        
        # Broadcast the update to all clients
        await self.broadcast(UIMessage(
            type="last_ai_speaker_changed",
            data={"character": character}
        ))
        
        # Also send a state sync to update UI
        await self.broadcast(UIMessage(
            type="state_sync",
            data=self.current_state
        ))

    async def handle_set_mute_while_speaking(self, enabled: bool):
        """Toggle mute while AI is speaking setting."""
        if not self.conversation or not hasattr(self.conversation, 'state'):
            return

        self.conversation.state.mute_while_speaking = enabled
        self.current_state["mute_while_speaking"] = enabled
        print(f"🔇 Mute while speaking: {'enabled' if enabled else 'disabled'}")

        # Broadcast the update to all clients
        await self.broadcast(UIMessage(
            type="mute_while_speaking_changed",
            data={"enabled": enabled}
        ))


# Example usage
if __name__ == "__main__":
    async def test_server():
        server = VoiceUIServer()
        await server.start()
        
        # Keep server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            await server.stop()
            
    asyncio.run(test_server())