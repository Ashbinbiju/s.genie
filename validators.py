"""
Validation utilities for input sanitization and security
"""
import re
from typing import Optional

class ValidationError(Exception):
    """Custom validation exception"""
    pass

def validate_symbol(symbol: str) -> str:
    """
    Validate and sanitize stock symbol
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Sanitized symbol
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    # Remove whitespace
    symbol = symbol.strip().upper()
    
    # Check format (alphanumeric with optional -EQ suffix)
    pattern = r'^[A-Z0-9]+([-][A-Z]{2})?$'
    if not re.match(pattern, symbol):
        raise ValidationError(f"Invalid symbol format: {symbol}")
    
    # Length check
    if len(symbol) > 20:
        raise ValidationError("Symbol too long")
    
    return symbol

def validate_timeframe(timeframe: str) -> str:
    """
    Validate timeframe parameter
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Validated timeframe
        
    Raises:
        ValidationError: If timeframe is invalid
    """
    valid_timeframes = ['1min', '5min', '15min', '30min', 'hour', 'day', 'week', 'month']
    
    timeframe = timeframe.lower().strip()
    
    if timeframe not in valid_timeframes:
        raise ValidationError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
    
    return timeframe

def validate_pagination(page: int, limit: int) -> tuple:
    """
    Validate pagination parameters
    
    Args:
        page: Page number
        limit: Items per page
        
    Returns:
        Tuple of (validated_page, validated_limit)
        
    Raises:
        ValidationError: If parameters are invalid
    """
    try:
        page = int(page)
        limit = int(limit)
    except (ValueError, TypeError):
        raise ValidationError("Page and limit must be integers")
    
    if page < 1:
        raise ValidationError("Page must be >= 1")
    
    if limit < 1 or limit > 100:
        raise ValidationError("Limit must be between 1 and 100")
    
    return page, limit

def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input to prevent injection attacks
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return ""
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # Trim whitespace
    value = value.strip()
    
    # Limit length
    if len(value) > max_length:
        value = value[:max_length]
    
    return value

def validate_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
    """
    Validate score is within range
    
    Args:
        score: Score to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated score
        
    Raises:
        ValidationError: If score is out of range
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        raise ValidationError("Score must be a number")
    
    if score < min_val or score > max_val:
        raise ValidationError(f"Score must be between {min_val} and {max_val}")
    
    return score
