"""
Structured Logging Module

Provides JSON-formatted structured logging with correlation IDs,
context propagation, and log levels for production observability.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional, List
from contextvars import ContextVar
import uuid
import traceback


# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredLogger(logging.Logger):
    """
    Custom logger that outputs JSON-formatted structured logs.
    
    Features:
    - Automatic correlation ID tracking
    - Context enrichment (symbol, strategy, etc.)
    - Structured field support
    - Multiple output handlers
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        output_file: Optional[str] = None,
        include_context: bool = True,
    ):
        super().__init__(name, level)
        
        self.include_context = include_context
        self.context_fields: Dict[str, Any] = {}
        
        # Remove default handlers
        self.handlers = []
        
        # Create formatter
        formatter = JsonFormatter(include_context=self.include_context)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        
        # File handler (optional)
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
    
    def set_context(self, **kwargs) -> None:
        """Add context fields to all subsequent log messages."""
        self.context_fields.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self.context_fields.clear()
    
    def _log_with_context(
        self,
        level: int,
        msg: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Internal method to log with enriched context."""
        extra = kwargs.get("extra", {})
        
        # Add correlation ID
        corr_id = correlation_id_var.get()
        if corr_id:
            extra["correlation_id"] = corr_id
        
        # Add logger context
        extra.update(self.context_fields)
        
        # Add extra fields from call
        if extra_fields:
            extra.update(extra_fields)
        
        kwargs["extra"] = extra
        super()._log(level, msg, (), **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Log exception with full stack trace."""
        exc_info = kwargs.pop("exc_info", True)
        extra_fields = kwargs.pop("extra_fields", {})
        
        if exc_info:
            stack_trace = traceback.format_exc()
            extra_fields["stack_trace"] = stack_trace
        
        self._log_with_context(logging.ERROR, msg, extra_fields=extra_fields, **kwargs)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_data["correlation_id"] = record.correlation_id
        
        # Add context fields
        if self.include_context:
            for key, value in record.__dict__.items():
                if key not in [
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "asctime",
                    "name",
                ] and not key.startswith("_"):
                    try:
                        # Ensure JSON serializable
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def get_logger(
    name: str,
    level: int = logging.INFO,
    output_file: Optional[str] = None,
    include_context: bool = True,
) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        output_file: Optional file path for log output
        include_context: Whether to include context fields
    
    Returns:
        StructuredLogger instance
    """
    logger = StructuredLogger(
        name=name,
        level=level,
        output_file=output_file,
        include_context=include_context,
    )
    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        correlation_id: Optional ID to use. If None, generates a new UUID.
    
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id_var.set(None)


def setup_structured_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level
        log_file: Optional file path to write logs
        json_format: Whether to use JSON format
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger("market_predictor_ml")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
