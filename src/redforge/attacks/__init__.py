"""Attack modules — DESIGN.md §6.2."""

from redforge.attacks.base import AttackModule, AttackVariant
from redforge.attacks.jailbreak import Jailbreak
from redforge.attacks.prompt_injection import PromptInjection

__all__ = ["AttackModule", "AttackVariant", "PromptInjection", "Jailbreak"]
