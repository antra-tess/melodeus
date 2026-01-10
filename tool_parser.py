#!/usr/bin/env python3
"""
Tool parsing utilities for XML-based tool calls.
Ported from membrane (TypeScript) to Python.

Format:
<function_calls>
<invoke name="tool_name">
<parameter name="param_name">value</parameter>
</invoke>
</function_calls>
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    id: str
    name: str
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ParsedToolCalls:
    """Result of parsing tool calls from text."""
    calls: List[ToolCall]
    before_text: str  # Text before the function_calls block
    after_text: str   # Text after the function_calls block
    full_match: str   # The complete matched function_calls block


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, description, required, enum}


# ============================================================================
# Constants (assembled to avoid triggering stop sequences)
# ============================================================================

FUNC_CALLS_OPEN = '<' + 'function_calls>'
FUNC_CALLS_CLOSE = '</' + 'function_calls>'
INVOKE_OPEN = '<' + 'invoke name="'
INVOKE_CLOSE = '</' + 'invoke>'
PARAM_OPEN = '<' + 'parameter name="'
PARAM_CLOSE = '</' + 'parameter>'
FUNC_RESULTS_OPEN = '<' + 'function_results>'
FUNC_RESULTS_CLOSE = '</' + 'function_results>'


# ============================================================================
# Regex Patterns
# ============================================================================

FUNCTION_CALLS_PATTERN = re.compile(
    r'<function_calls>([\s\S]*?)</function_calls>',
    re.DOTALL
)

# Fallback pattern for unclosed function_calls blocks (model didn't output closing tag)
FUNCTION_CALLS_UNCLOSED_PATTERN = re.compile(
    r'<function_calls>([\s\S]*)</invoke>',
    re.DOTALL
)

INVOKE_PATTERN = re.compile(
    r'<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>',
    re.DOTALL
)

PARAMETER_PATTERN = re.compile(
    r'<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>',
    re.DOTALL
)


# ============================================================================
# Tool ID Generation
# ============================================================================

_tool_id_counter = 0

def generate_tool_id() -> str:
    """Generate a unique tool ID."""
    global _tool_id_counter
    _tool_id_counter += 1
    return f"tool_{int(time.time())}_{_tool_id_counter}"


# ============================================================================
# XML Escaping
# ============================================================================

def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))


def unescape_xml(text: str) -> str:
    """Unescape XML entities."""
    return (text
        .replace('&apos;', "'")
        .replace('&quot;', '"')
        .replace('&gt;', '>')
        .replace('&lt;', '<')
        .replace('&amp;', '&'))


# ============================================================================
# Tool Call Parsing
# ============================================================================

def parse_tool_calls(text: str) -> Optional[ParsedToolCalls]:
    """
    Parse tool calls from text containing XML function_calls blocks.

    Returns None if no tool calls found.
    """
    match = FUNCTION_CALLS_PATTERN.search(text)

    # Try fallback pattern if main pattern doesn't match (unclosed block)
    if not match:
        match = FUNCTION_CALLS_UNCLOSED_PATTERN.search(text)
        if match:
            # For unclosed blocks, we need to reconstruct the full_match to include </invoke>
            inner_content = match.group(1) or ''
            # Find where the last </invoke> ends
            full_match = match.group(0)
            match_index = match.start()
            before_text = text[:match_index]
            after_text = text[match_index + len(full_match):]

            calls: List[ToolCall] = []
            for invoke_match in INVOKE_PATTERN.finditer(inner_content + '</invoke>'):
                tool_name = invoke_match.group(1) or ''
                invoke_content = invoke_match.group(2) or ''
                input_params: Dict[str, Any] = {}
                for param_match in PARAMETER_PATTERN.finditer(invoke_content):
                    param_name = param_match.group(1) or ''
                    param_value = param_match.group(2) or ''
                    try:
                        input_params[param_name] = json.loads(param_value)
                    except (json.JSONDecodeError, ValueError):
                        input_params[param_name] = param_value.strip()
                calls.append(ToolCall(id=generate_tool_id(), name=tool_name, input=input_params))

            return ParsedToolCalls(calls=calls, before_text=before_text, after_text=after_text, full_match=full_match)

    if not match:
        return None

    full_match = match.group(0)
    inner_content = match.group(1) or ''
    match_index = match.start()

    before_text = text[:match_index]
    after_text = text[match_index + len(full_match):]

    calls: List[ToolCall] = []

    # Parse invocations
    for invoke_match in INVOKE_PATTERN.finditer(inner_content):
        tool_name = invoke_match.group(1) or ''
        invoke_content = invoke_match.group(2) or ''

        # Parse parameters
        input_params: Dict[str, Any] = {}
        for param_match in PARAMETER_PATTERN.finditer(invoke_content):
            param_name = param_match.group(1) or ''
            param_value = param_match.group(2) or ''

            # Try to parse as JSON, fall back to string
            try:
                input_params[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                input_params[param_name] = param_value.strip()

        calls.append(ToolCall(
            id=generate_tool_id(),
            name=tool_name,
            input=input_params
        ))

    return ParsedToolCalls(
        calls=calls,
        before_text=before_text,
        after_text=after_text,
        full_match=full_match
    )


def has_unclosed_tool_block(text: str) -> bool:
    """
    Check if text contains an unclosed function_calls block.
    Used for false-positive stop sequence detection.
    """
    open_count = text.count('<function_calls>')
    close_count = text.count('</function_calls>')
    return open_count > close_count


def ends_with_partial_tool_block(text: str) -> bool:
    """Check if text ends with a partial/unclosed tool block."""
    # Check for partial opening tags
    if re.search(r'<function_calls[^>]*$', text):
        return True
    if re.search(r'<invoke[^>]*$', text):
        return True
    if re.search(r'<parameter[^>]*$', text):
        return True

    # Check for unclosed block
    return has_unclosed_tool_block(text)


# ============================================================================
# Tool Result Formatting
# ============================================================================

def format_tool_results(results: List[ToolResult]) -> str:
    """Format tool results as XML for injection back into conversation."""
    parts = [FUNC_RESULTS_OPEN]

    for result in results:
        if result.is_error:
            parts.append(f'<error tool_use_id="{result.tool_use_id}">')
            parts.append(escape_xml(result.content))
            parts.append('</error>')
        else:
            parts.append(f'<result tool_use_id="{result.tool_use_id}">')
            parts.append(escape_xml(result.content))
            parts.append('</result>')

    parts.append(FUNC_RESULTS_CLOSE)
    return '\n'.join(parts)


def format_tool_result(result: ToolResult) -> str:
    """Format a single tool result."""
    return format_tool_results([result])


# ============================================================================
# Tool Definition Formatting (for system prompt injection)
# ============================================================================

def format_tool_definitions(tools: List[ToolDefinition]) -> str:
    """Format tool definitions as XML for system prompt."""
    parts = ['<tools>']

    for tool in tools:
        parts.append(f'<tool name="{escape_xml(tool.name)}">')
        parts.append(f'<description>{escape_xml(tool.description)}</description>')
        parts.append('<parameters>')

        for param_name, param in tool.parameters.items():
            attrs = [f'name="{escape_xml(param_name)}"', f'type="{param.get("type", "string")}"']
            if param.get('required'):
                attrs.append('required="true"')
            if param.get('enum'):
                attrs.append(f'enum="{",".join(param["enum"])}"')

            parts.append(f'<parameter {" ".join(attrs)}>')
            if param.get('description'):
                parts.append(escape_xml(param['description']))
            parts.append('</parameter>')

        parts.append('</parameters>')
        parts.append('</tool>')

    parts.append('</tools>')
    return '\n'.join(parts)


def get_tool_instructions(tools: List[ToolDefinition]) -> str:
    """
    Get tool usage instructions for system prompt injection.
    Shows the exact XML format needed to call tools.
    """
    # Build concrete examples for each tool
    examples = []
    for tool in tools:
        param_lines = []
        for param_name, param in tool.parameters.items():
            example_value = param.get('example', f'{param_name}_value')
            param_lines.append(f'<parameter name="{param_name}">{example_value}</parameter>')
        
        example = f"""<function_calls>
<invoke name="{tool.name}">
{chr(10).join(param_lines)}
</invoke>
</function_calls>"""
        examples.append(f"To use {tool.name}:\n{example}")

    return f"""To use a tool, output this exact XML format:

<function_calls>
<invoke name="TOOL_NAME">
<parameter name="PARAM_NAME">value</parameter>
</invoke>
</function_calls>

For array parameters, use JSON inside the parameter tags:

<function_calls>
<invoke name="set_color">
<parameter name="rgb">[255, 0, 0]</parameter>
</invoke>
</function_calls>

{chr(10).join(examples)}"""


def inject_tools_into_system(system: str, tools: List[ToolDefinition]) -> str:
    """Inject tool definitions and instructions into system prompt."""
    if not tools:
        return system

    tools_section = f"""
<available_tools>
{format_tool_definitions(tools)}
</available_tools>

{get_tool_instructions(tools)}
"""
    return system + '\n\n' + tools_section


# ============================================================================
# Stop Sequences & Turn Delimiters
# ============================================================================

# End-of-turn token - marks the boundary between conversation turns
# Used to prevent model from simulating other participants
EOT_TOKEN = "<|eot|>"


def get_eot_token() -> str:
    """Get the end-of-turn token for message boundaries."""
    return EOT_TOKEN


def get_tool_stop_sequence() -> str:
    """Get the stop sequence for tool calls."""
    return '</function_calls>'


def build_stop_sequences(
    participants: List[str],
    assistant_name: str = 'Claude',
    max_participants: int = 10,
    additional_sequences: Optional[List[str]] = None,
    include_eot: bool = True
) -> List[str]:
    """
    Build stop sequences for prefill mode.
    Includes participant-based stops, EOT token, and tool completion stop.
    """
    sequences = []

    # EOT token - highest priority stop (prevents participant simulation)
    if include_eot:
        sequences.append(get_eot_token())

    # Participant-based stops (excluding assistant)
    seen = set()
    for participant in participants:
        if participant != assistant_name and participant not in seen and len(seen) < max_participants:
            sequences.append(f'\n{participant}:')
            seen.add(participant)

    # Tool completion stop
    sequences.append(get_tool_stop_sequence())

    # Additional sequences
    if additional_sequences:
        sequences.extend(additional_sequences)

    return sequences


# ============================================================================
# Convenience Functions
# ============================================================================

def create_tool_definition(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None
) -> ToolDefinition:
    """Create a tool definition with sensible defaults."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters or {}
    )


def tool_calls_to_legacy_format(calls: List[ToolCall]) -> List[Dict[str, Any]]:
    """Convert parsed tool calls to legacy melodeus format."""
    return [
        {
            'tag_name': call.name,
            'content': f'<{call.name}>{json.dumps(call.input)}</{call.name}>',
            'id': call.id,
            'input': call.input
        }
        for call in calls
    ]
