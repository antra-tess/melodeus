#!/usr/bin/env python3
"""Token counting utilities for managing prompt lengths."""

import tiktoken
from typing import List, Dict, Any, Union

# Model to encoding mapping
MODEL_ENCODINGS = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "chatgpt-4o-latest": "cl100k_base",
    
    # Anthropic models (approximate using cl100k_base)
    "claude-3-opus-20240229": "cl100k_base",
    "claude-3-5-sonnet-20241022": "cl100k_base",
    "claude-3-haiku-20240307": "cl100k_base",
    
    # Groq models (approximate)
    "llama-3.1-8b-instant": "cl100k_base",
    "moonshotai/kimi-k2-instruct": "cl100k_base",
}

def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the appropriate tiktoken encoding for a model."""
    encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)

def count_message_tokens(message: Dict[str, Any], model: str = "gpt-4") -> int:
    """Count tokens in a single message.
    
    Args:
        message: Message dict with 'role' and 'content'
        model: Model name for encoding selection
        
    Returns:
        Token count
    """
    encoding = get_encoding_for_model(model)
    
    # Base tokens for message structure (role + formatting)
    # Approximate overhead per message
    tokens = 4  # <|im_start|>role\n content<|im_end|>\n
    
    # Add role tokens
    tokens += len(encoding.encode(message.get("role", "")))
    
    # Count content tokens
    content = message.get("content", "")
    
    if isinstance(content, str):
        tokens += len(encoding.encode(content))
    elif isinstance(content, list):
        # Handle structured content (text + images)
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    tokens += len(encoding.encode(item.get("text", "")))
                elif item.get("type") == "image":
                    # Images have significant token cost
                    # Rough approximation based on Anthropic/OpenAI pricing
                    tokens += 1000  # Base cost for an image
    
    return tokens

def count_messages_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4") -> int:
    """Count total tokens in a list of messages.
    
    Args:
        messages: List of message dicts
        model: Model name for encoding selection
        
    Returns:
        Total token count
    """
    total = 0
    for message in messages:
        total += count_message_tokens(message, model)
    
    # Add base prompt tokens
    total += 3  # Every reply is primed with <|im_start|>assistant
    
    return total

def truncate_messages_to_fit(
    messages: List[Dict[str, Any]], 
    max_tokens: int, 
    model: str = "gpt-4",
    keep_system: bool = True
) -> List[Dict[str, Any]]:
    """Truncate messages to fit within token limit.
    
    Keeps the system message (if present) and as many recent messages as possible.
    
    Args:
        messages: List of message dicts
        max_tokens: Maximum token count
        model: Model name for encoding selection
        keep_system: Whether to always keep the system message
        
    Returns:
        Truncated list of messages
    """
    if not messages:
        return []
    
    # Separate system message if present
    system_message = None
    other_messages = messages
    
    if keep_system and messages and messages[0].get("role") == "system":
        system_message = messages[0]
        other_messages = messages[1:]
    
    # Start with system message if we have one
    result = []
    current_tokens = 0
    
    if system_message:
        system_tokens = count_message_tokens(system_message, model)
        if system_tokens <= max_tokens:
            result.append(system_message)
            current_tokens = system_tokens
    
    # Add messages from newest to oldest, then restore chronological order
    kept_messages = []
    for message in reversed(other_messages):
        message_tokens = count_message_tokens(message, model)
        if current_tokens + message_tokens <= max_tokens:
            kept_messages.append(message)
            current_tokens += message_tokens
        else:
            # Can't fit any more messages
            break
    
    kept_messages.reverse()
    return result + kept_messages

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Quick token estimation for a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding selection
        
    Returns:
        Token count
    """
    encoding = get_encoding_for_model(model)
    return len(encoding.encode(text))
