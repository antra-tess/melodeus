"""
Flic Button Listener for Melodeus

Listens for TCP connections from Flic Hub and routes button events
to interrupt or trigger specific characters.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any


@dataclass
class FlicConfig:
    """Configuration for Flic button listener."""
    enabled: bool = False
    port: int = 11235
    buttons: Dict[str, Any] = None

    def __post_init__(self):
        if self.buttons is None:
            self.buttons = {
                "default": {
                    "single_click": "trigger:Sonnet45",
                    "double_click": "interrupt",
                    "hold": "interrupt"
                }
            }


class FlicButtonListener:
    """
    TCP server that listens for Flic Hub button events.

    Protocol: Newline-delimited JSON
    Events from hub:
      - {"type": "hubInfo", "serial": "...", "firmware": "..."}
      - {"type": "buttonInfo", "button": {...}}
      - {"isSingleClick": true, "isDoubleClick": false, "isHold": false, "bdaddr": "XX:XX:XX:XX:XX:XX", ...}
    """

    def __init__(self, config: FlicConfig, on_interrupt: Callable, on_trigger: Callable[[str], None]):
        """
        Initialize the Flic button listener.

        Args:
            config: FlicConfig with port and button mappings
            on_interrupt: Callback when interrupt action is triggered
            on_trigger: Callback when character trigger action is triggered, receives character name
        """
        self.config = config
        self.on_interrupt = on_interrupt
        self.on_trigger = on_trigger
        self.server = None
        self.connected_clients = []
        self.known_buttons = {}  # bdaddr -> button info

    async def start(self):
        """Start the TCP server."""
        if not self.config.enabled:
            print("Flic button listener disabled in config")
            return

        self.server = await asyncio.start_server(
            self._handle_client,
            '0.0.0.0',
            self.config.port
        )
        print(f"🎮 Flic button listener started on port {self.config.port}")

    async def stop(self):
        """Stop the TCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("🎮 Flic button listener stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a connected Flic Hub."""
        addr = writer.get_extra_info('peername')
        print(f"🎮 Flic Hub connected from {addr}")
        self.connected_clients.append(writer)

        try:
            buffer = ""
            while True:
                data = await reader.read(4096)
                if not data:
                    break

                buffer += data.decode('utf-8')

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        await self._process_message(line.strip())

        except Exception as e:
            print(f"🎮 Flic connection error: {e}")
        finally:
            print(f"🎮 Flic Hub disconnected from {addr}")
            self.connected_clients.remove(writer)
            writer.close()

    async def _process_message(self, message: str):
        """Process a JSON message from the Flic Hub."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            print(f"🎮 Invalid JSON from Flic: {e}")
            return

        msg_type = data.get('type')

        if msg_type == 'hubInfo':
            print(f"🎮 Hub info: serial={data.get('serial')}, firmware={data.get('firmware')}")

        elif msg_type == 'buttonInfo':
            button = data.get('button', {})
            bdaddr = button.get('bdaddr', 'unknown')
            name = button.get('name', 'unnamed')
            self.known_buttons[bdaddr] = button
            print(f"🎮 Button registered: {name} ({bdaddr})")

        elif 'isSingleClick' in data or 'isDoubleClick' in data or 'isHold' in data:
            # This is a button event
            await self._handle_button_event(data)

        else:
            print(f"🎮 Unknown Flic message: {data}")

    async def _handle_button_event(self, event: dict):
        """Handle a button click/hold event."""
        bdaddr = event.get('bdaddr', 'unknown')

        # Determine click type
        if event.get('isHold'):
            click_type = 'hold'
        elif event.get('isDoubleClick'):
            click_type = 'double_click'
        elif event.get('isSingleClick'):
            click_type = 'single_click'
        else:
            print(f"🎮 Unknown click type: {event}")
            return

        button_name = self.known_buttons.get(bdaddr, {}).get('name', bdaddr)
        print(f"🎮 Button event: {button_name} -> {click_type}")

        # Look up action for this button/click type
        action = self._get_action(bdaddr, click_type)

        if action:
            await self._execute_action(action, button_name)

    def _get_action(self, bdaddr: str, click_type: str) -> Optional[str]:
        """Get the action for a button/click type combination."""
        buttons_config = self.config.buttons or {}

        # First check for button-specific mapping
        if bdaddr in buttons_config:
            button_config = buttons_config[bdaddr]
            if isinstance(button_config, str):
                # Simple mapping: any click type triggers this action
                return button_config
            elif isinstance(button_config, dict):
                return button_config.get(click_type)

        # Fall back to default mapping
        default_config = buttons_config.get('default', {})
        if isinstance(default_config, dict):
            return default_config.get(click_type)

        return None

    async def _execute_action(self, action: str, button_name: str):
        """Execute a button action."""
        print(f"🎮 Executing action: {action}")

        if action == 'interrupt':
            print(f"🎮 INTERRUPT triggered by {button_name}")
            if self.on_interrupt:
                try:
                    result = self.on_interrupt()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"🎮 Error executing interrupt: {e}")

        elif action.startswith('trigger:'):
            character = action[8:]  # Remove "trigger:" prefix
            print(f"🎮 TRIGGER {character} by {button_name}")
            if self.on_trigger:
                try:
                    result = self.on_trigger(character)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"🎮 Error triggering {character}: {e}")

        elif action.startswith('prompt:'):
            prompt_text = action[7:]  # Remove "prompt:" prefix
            print(f"🎮 PROMPT '{prompt_text}' by {button_name}")
            # TODO: Implement prompt injection

        else:
            print(f"🎮 Unknown action: {action}")


def load_flic_config(config_dict: dict) -> FlicConfig:
    """Load FlicConfig from a config dictionary."""
    flic_data = config_dict.get('flic', {})
    return FlicConfig(
        enabled=flic_data.get('enabled', False),
        port=flic_data.get('port', 11235),
        buttons=flic_data.get('buttons', None)
    )
