"""Judge implementations — DESIGN.md §6.4."""

from redforge.scoring.judges.anthropic import AnthropicJudge
from redforge.scoring.judges.ollama import OllamaJudge
from redforge.scoring.judges.openai import OpenAIJudge

__all__ = ["AnthropicJudge", "OpenAIJudge", "OllamaJudge"]
