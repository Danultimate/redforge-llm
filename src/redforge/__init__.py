"""RedForge — adversarial testing for LLM applications.

See DESIGN.md for the full design.
"""

from redforge.scanner import ScanConfig, Scanner
from redforge.types import (
    AttackPrompt,
    AttackResult,
    ErrorClass,
    ScanResult,
    Severity,
    TargetResponse,
    Verdict,
)

__version__ = "0.1.0"
__all__ = [
    "Scanner",
    "ScanConfig",
    "TargetResponse",
    "AttackPrompt",
    "Severity",
    "ErrorClass",
    "Verdict",
    "AttackResult",
    "ScanResult",
]
