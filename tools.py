#!/usr/bin/env python3
"""
Configurable Tools for Voice AI System
Allows defining custom tools that can be executed during conversation
"""

import asyncio
import subprocess
import os
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from async_tts_module import ToolCall, ToolResult


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize tool with optional configuration."""
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """
        Execute the tool with given content.
        
        Args:
            content: The content extracted from the tool XML tags
            context: Optional context from the conversation
            
        Returns:
            ToolResult with should_interrupt flag and optional content
        """
        pass


class CommandTool(Tool):
    """Execute system commands."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a system command."""
        try:
            # Check if command is allowed
            allowed_commands = self.config.get('allowed_commands', [])
            if allowed_commands:
                command_parts = content.split()
                if command_parts and command_parts[0] not in allowed_commands:
                    return ToolResult(
                        should_interrupt=False,
                        content=f"Command '{command_parts[0]}' not allowed"
                    )
            
            # Execute command with timeout
            timeout = self.config.get('timeout', 30)
            
            # Run command asynchronously
            proc = await asyncio.create_subprocess_shell(
                content,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.get('working_directory', os.getcwd())
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                stderr_text = stderr.decode('utf-8', errors='replace').strip()
                
                # Combine output
                output = stdout_text
                if stderr_text:
                    output += f"\nError: {stderr_text}"
                
                # Check if we should interrupt based on config
                interrupt_on_error = self.config.get('interrupt_on_error', False)
                should_interrupt = interrupt_on_error and proc.returncode != 0
                
                return ToolResult(
                    should_interrupt=should_interrupt,
                    content=output if output else f"Command completed with code {proc.returncode}"
                )
                
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    should_interrupt=True,
                    content=f"Command timed out after {timeout} seconds"
                )
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Error executing command: {str(e)}"
            )


class SearchTool(Tool):
    """Perform search operations."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a search."""
        try:
            # Get search provider from config
            provider = self.config.get('provider', 'files')
            
            if provider == 'files':
                # Search in local files
                search_paths = self.config.get('search_paths', ['.'])
                max_results = self.config.get('max_results', 5)
                
                results = []
                for path in search_paths:
                    # Simple grep-like search
                    proc = await asyncio.create_subprocess_exec(
                        'grep', '-r', '-i', '-n', content, path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await proc.communicate()
                    
                    if stdout:
                        lines = stdout.decode('utf-8', errors='replace').strip().split('\n')
                        results.extend(lines[:max_results - len(results)])
                        
                        if len(results) >= max_results:
                            break
                
                if results:
                    result_text = f"Found {len(results)} matches:\n" + "\n".join(results)
                    return ToolResult(
                        should_interrupt=self.config.get('interrupt_on_results', True),
                        content=result_text
                    )
                else:
                    return ToolResult(
                        should_interrupt=False,
                        content=f"No matches found for '{content}'"
                    )
                    
            elif provider == 'web':
                # Placeholder for web search
                return ToolResult(
                    should_interrupt=True,
                    content=f"Web search for '{content}' not yet implemented"
                )
                
            else:
                return ToolResult(
                    should_interrupt=False,
                    content=f"Unknown search provider: {provider}"
                )
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Search error: {str(e)}"
            )


class CalculationTool(Tool):
    """Perform calculations."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a calculation."""
        try:
            # Define safe math operations
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow,
                # Math functions
                'sin': __import__('math').sin,
                'cos': __import__('math').cos,
                'tan': __import__('math').tan,
                'sqrt': __import__('math').sqrt,
                'log': __import__('math').log,
                'pi': __import__('math').pi,
                'e': __import__('math').e,
            }
            
            # Evaluate expression safely
            result = eval(content, {"__builtins__": {}}, safe_dict)
            
            # Format result based on config
            precision = self.config.get('precision', 2)
            if isinstance(result, float):
                result_str = f"{result:.{precision}f}"
            else:
                result_str = str(result)
            
            return ToolResult(
                should_interrupt=False,  # Calculations usually don't interrupt
                content=f"= {result_str}"
            )
            
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Calculation error: {str(e)}"
            )


class CustomTool(Tool):
    """Custom tool that can be configured with a Python function."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.function = None
        
        # Load function from config
        if 'function' in self.config:
            # This could be a module path like "my_module.my_function"
            func_path = self.config['function']
            try:
                module_name, func_name = func_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[func_name])
                self.function = getattr(module, func_name)
            except Exception as e:
                print(f"Error loading custom function '{func_path}': {e}")
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute custom function."""
        if not self.function:
            return ToolResult(
                should_interrupt=False,
                content="Custom function not configured"
            )
        
        try:
            # Call function (convert to async if needed)
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(content, context, self.config)
            else:
                result = await asyncio.to_thread(
                    self.function, content, context, self.config
                )
            
            # Handle different return types
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, tuple) and len(result) == 2:
                return ToolResult(should_interrupt=result[0], content=result[1])
            elif isinstance(result, str):
                return ToolResult(should_interrupt=False, content=result)
            else:
                return ToolResult(should_interrupt=False, content=str(result))
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Custom tool error: {str(e)}"
            )

class OSCTool(Tool):
    """Tool that sends OSC messages with pre-configured address and prefix args.

    Config options:
        address: OSC address path (e.g., "/avatar/expression")
        description: Human-readable description for the model
        prefix_args: Args to prepend (e.g., avatar name)
        param_name: Name of the parameter the model provides (default: "value")
        param_description: Description of the parameter for the model
    """

    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Send an OSC message with the configured address and provided value."""
        import json

        # Get OSC client from context
        osc_client = context.get('osc_client') if context else None

        if not osc_client:
            print("⚠️ OSCTool: No OSC client available")
            return ToolResult(should_interrupt=False, content="OSC not configured")

        try:
            # Get configured address and prefix args
            address = self.config.get('address', '/default')
            prefix_args = self.config.get('prefix_args', [])

            # Ensure prefix_args is a list
            if not isinstance(prefix_args, list):
                prefix_args = [prefix_args]

            # Parse the value from content (could be JSON or simple string)
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                value = content.strip()

            # Build the full args list: prefix_args + value(s)
            # If value is a list (like RGB), flatten it into args
            if isinstance(value, list):
                args = prefix_args + value
            else:
                args = prefix_args + [value]

            # Send the OSC message (flatten the list for python-osc)
            osc_client.send_message(address, args)
            print(f"📡 OSCTool: Sent {address} {args}")
            return ToolResult(should_interrupt=False, content=f"Done")

        except Exception as e:
            print(f"❌ OSCTool: Error: {e}")
            return ToolResult(should_interrupt=False, content=f"Error: {str(e)}")
    

class ToolRegistry:
    """Registry for managing available tools."""
    
    # Built-in tool types
    TOOL_TYPES = {
        'command': CommandTool,
        'search': SearchTool,
        'calculation': CalculationTool,
        'custom': CustomTool,
        'osc': OSCTool,
    }
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, name: str, tool_type: str, config: Dict[str, Any] = None):
        """
        Register a new tool.
        
        Args:
            name: XML tag name for the tool (e.g., 'cmd', 'search')
            tool_type: Type of tool ('command', 'search', 'calculation', 'custom')
            config: Tool-specific configuration
        """
        if tool_type not in self.TOOL_TYPES:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        tool_class = self.TOOL_TYPES[tool_type]
        self.tools[name] = tool_class(config)
        print(f"✅ Registered tool '{name}' of type '{tool_type}'")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_tool_definitions(self) -> list:
        """Get tool definitions for system prompt injection.

        Returns a list of dicts with name, description, and parameters for each tool.
        """
        from tool_parser import ToolDefinition

        definitions = []
        for name, tool in self.tools.items():
            # Try to get description from tool docstring or config
            description = tool.__class__.__doc__ or f"Execute {name} tool"
            description = description.strip().split('\n')[0]  # First line only

            # Build parameter schema based on tool type
            parameters = {}
            if isinstance(tool, CommandTool):
                parameters = {
                    'command': {
                        'type': 'string',
                        'description': 'The command to execute',
                        'required': True
                    }
                }
            elif isinstance(tool, SearchTool):
                parameters = {
                    'query': {
                        'type': 'string',
                        'description': 'The search query',
                        'required': True
                    }
                }
            elif isinstance(tool, CalculationTool):
                parameters = {
                    'expression': {
                        'type': 'string',
                        'description': 'The mathematical expression to evaluate',
                        'required': True
                    }
                }
            elif isinstance(tool, OSCTool):
                # OSC tools use config for parameter name and description
                param_name = tool.config.get('param_name', 'value')
                param_desc = tool.config.get('param_description', 'The value to send')
                parameters = {
                    param_name: {
                        'type': 'string',
                        'description': param_desc,
                        'required': True
                    }
                }
                # Use configured description if available
                if tool.config.get('description'):
                    description = tool.config['description']
            else:
                # Generic parameter for custom tools
                parameters = {
                    'content': {
                        'type': 'string',
                        'description': 'The input content for the tool',
                        'required': True
                    }
                }

            definitions.append(ToolDefinition(
                name=name,
                description=description,
                parameters=parameters
            ))

        return definitions
    
    async def execute_tool(self, tool_name_or_call, input_params: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Any:
        """Execute a tool call.

        Supports two calling conventions:
        1. New format: execute_tool("tool_name", {"param": "value"})
        2. Legacy format: execute_tool(ToolCall(tag_name="...", content="..."))
        """
        # Handle both old and new API
        if isinstance(tool_name_or_call, str):
            # New format: tool_name as string, input as dict
            tool_name = tool_name_or_call
            tool = self.get_tool(tool_name)
            if not tool:
                return f"Unknown tool: {tool_name}"

            # Convert input dict to content string for tools that expect it
            # Tools may expect different formats, so we try to be flexible
            if input_params:
                # If there's a single 'content' param, use it directly
                if 'content' in input_params and len(input_params) == 1:
                    content = str(input_params['content'])
                else:
                    # Otherwise, serialize the params
                    import json
                    content = json.dumps(input_params)
            else:
                content = ""

            result = await tool.execute(content, context)
            # Return just the content for the new format
            return result.content if result.content else ""
        else:
            # Legacy format: ToolCall object
            tool_call = tool_name_or_call
            tool = self.get_tool(tool_call.tag_name)

            if not tool:
                return ToolResult(
                    should_interrupt=False,
                    content=f"Unknown tool: {tool_call.tag_name}"
                )

            # Extract content between tags
            import re
            pattern = f'<{tool_call.tag_name}[^>]*>(.*?)</{tool_call.tag_name}>'
            match = re.search(pattern, tool_call.content, re.DOTALL)

            if match:
                content = match.group(1).strip()
                return await tool.execute(content, context)
            else:
                return ToolResult(
                    should_interrupt=False,
                    content=f"Could not parse tool content"
                )
    
    def load_from_config(self, tools_config: Dict[str, Dict[str, Any]]):
        """
        Load tools from configuration dictionary.
        
        Args:
            tools_config: Dictionary mapping tool names to their configurations
        """
        for name, config in tools_config.items():
            tool_type = config.get('type', 'custom')
            self.register_tool(name, tool_type, config)


# Example custom tool function
async def example_weather_tool(content: str, context: Dict[str, Any], config: Dict[str, Any]) -> ToolResult:
    """Example custom tool for weather queries."""
    # This is just an example - in real use, this would call a weather API
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        if city.lower() in content.lower():
            return ToolResult(
                should_interrupt=True,
                content=f"The weather in {city} is sunny with a temperature of 22°C"
            )
    
    return ToolResult(
        should_interrupt=False,
        content="I couldn't determine which city you're asking about"
    )


# Convenience function for creating a registry from config
def create_tool_registry(tools_config: Dict[str, Dict[str, Any]] = None) -> ToolRegistry:
    """Create a tool registry from configuration."""
    registry = ToolRegistry()
    
    if tools_config:
        registry.load_from_config(tools_config)
    
    return registry