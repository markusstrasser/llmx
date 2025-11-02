"""Logger with agent-inspectable output"""

import sys
import json
from datetime import datetime
from typing import Optional, Any

# Global config
_config = {"debug": False, "json_mode": False}


def configure_logger(debug: bool = False, json_mode: bool = False):
    """Configure logger settings"""
    _config["debug"] = debug
    _config["json_mode"] = json_mode


def log(level: str, message: str, data: Optional[Any] = None):
    """Log to stderr (never interferes with stdout)"""
    if level == "debug" and not _config["debug"]:
        return

    if _config["json_mode"]:
        # JSON format for agent parsing
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if data:
            log_entry["data"] = data
        print(json.dumps(log_entry), file=sys.stderr)
        return

    # Human-readable format
    prefixes = {
        "info": "ℹ",
        "warn": "⚠",
        "error": "✗",
        "debug": "🔍",
    }
    prefix = prefixes.get(level, "•")

    formatted = f"{prefix} {message}"
    if data:
        formatted += f" {json.dumps(data)}"

    print(formatted, file=sys.stderr)


class Logger:
    """Logger instance"""

    @staticmethod
    def info(msg: str, data: Optional[Any] = None):
        log("info", msg, data)

    @staticmethod
    def warn(msg: str, data: Optional[Any] = None):
        log("warn", msg, data)

    @staticmethod
    def error(msg: str, data: Optional[Any] = None):
        log("error", msg, data)

    @staticmethod
    def debug(msg: str, data: Optional[Any] = None):
        log("debug", msg, data)


logger = Logger()
