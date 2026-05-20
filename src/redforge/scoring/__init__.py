"""Scoring layer — DESIGN.md §6.4."""

from redforge.scoring.base import Judge, Scorer
from redforge.scoring.default import DefaultScorer
from redforge.scoring.heuristic import HeuristicScorer
from redforge.scoring.judges.anthropic import AnthropicJudge
from redforge.scoring.judges.ollama import OllamaJudge
from redforge.scoring.judges.openai import OpenAIJudge

__all__ = [
    "Scorer",
    "Judge",
    "DefaultScorer",
    "HeuristicScorer",
    "AnthropicJudge",
    "OpenAIJudge",
    "OllamaJudge",
]
