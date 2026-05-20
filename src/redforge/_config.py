"""YAML config + target module loader for `redforge scan`.

The YAML schema is a superset of `ScanConfig` plus two CLI-only sections:

  - `target_spec`: "module:symbol" (defaults to "target:target")
  - `judge`: {type: anthropic|openai|ollama|none, model: <str>}

This module is intentionally separate from `ScanConfig` itself so the library
core stays free of CLI/import concerns.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from redforge.executor import ExecutorConfig
from redforge.scanner import ScanConfig
from redforge.scoring import DefaultScorer
from redforge.scoring.base import Judge, Scorer


class JudgeFileConfig(BaseModel):
    type: Literal["anthropic", "openai", "ollama", "none"] = "anthropic"
    model: str | None = None
    host: str | None = None  # ollama-only


class ScanFileConfig(BaseModel):
    """User-facing YAML schema. Fields not in ScanConfig are CLI-only."""

    sample_size: int | None = 15
    seed: int = 42
    artifact_dir: Path = Path(".redforge/runs")
    strict: bool = False
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    target_spec: str = "target:target"
    judge: JudgeFileConfig = Field(default_factory=JudgeFileConfig)

    def to_scan_config(self) -> ScanConfig:
        return ScanConfig(
            sample_size=self.sample_size,
            seed=self.seed,
            artifact_dir=self.artifact_dir,
            executor=self.executor,
            strict=self.strict,
        )


@dataclass
class LoadedConfig:
    file_config: ScanFileConfig
    scan_config: ScanConfig
    config_path: Path | None  # None when no file was found


def load_config(path: Path | None) -> LoadedConfig:
    """Load a YAML config from `path`. If `path` is None or missing, returns
    defaults — letting users run `redforge scan` without writing a YAML file.
    """
    if path is None or not path.exists():
        file_config = ScanFileConfig()
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping.")
        file_config = ScanFileConfig.model_validate(raw)
    return LoadedConfig(
        file_config=file_config,
        scan_config=file_config.to_scan_config(),
        config_path=path if (path and path.exists()) else None,
    )


def load_target(target_spec: str, cwd: Path) -> Any:
    """Resolve `<module>:<symbol>` to a callable.

    For simple module names (no dot), this loads `<cwd>/<module>.py` directly
    via `importlib.util.spec_from_file_location` — this avoids polluting
    `sys.path` and isolates each scan from any other modules of the same name.

    For dotted module names (e.g. `my_app.evals:scan_target`), this falls back
    to regular `import_module` with the CWD on `sys.path`.
    """
    if ":" not in target_spec:
        raise ValueError(
            f"target_spec {target_spec!r} must be 'module:symbol' "
            "(e.g. 'target:target')."
        )
    module_name, _, symbol = target_spec.partition(":")
    cwd = cwd.resolve()

    if "." not in module_name:
        # Simple-module convention: load the exact file at <cwd>/<module>.py.
        candidate = cwd / f"{module_name}.py"
        if not candidate.exists():
            raise ImportError(
                f"No `{module_name}.py` found in {cwd}. "
                f"Run `redforge init` to scaffold a starter target.py, "
                f"or set target_spec in redforge.yaml to point at your module."
            )
        spec = importlib.util.spec_from_file_location(
            f"redforge_target_{module_name}", candidate
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # Dotted module — fall back to standard import with CWD on path.
        cwd_str = str(cwd)
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        if module_name in sys.modules:
            del sys.modules[module_name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"Could not import {module_name!r} from {cwd_str!r}. "
                f"Original error: {e}"
            ) from e

    if not hasattr(module, symbol):
        raise AttributeError(
            f"Module {module_name!r} has no attribute {symbol!r}. "
            f"Expected an async callable exporting `{symbol}`."
        )
    return getattr(module, symbol)


def build_scorer(judge_config: JudgeFileConfig) -> Scorer:
    """Construct a DefaultScorer with the requested judge implementation."""
    judge = _build_judge(judge_config)
    return DefaultScorer(judge=judge)


def _build_judge(judge_config: JudgeFileConfig) -> Judge | None:
    if judge_config.type == "none":
        return None
    if judge_config.type == "anthropic":
        from redforge.scoring.judges import AnthropicJudge

        if judge_config.model is not None:
            return AnthropicJudge(model=judge_config.model)
        return AnthropicJudge()
    if judge_config.type == "openai":
        from redforge.scoring.judges import OpenAIJudge

        if judge_config.model is not None:
            return OpenAIJudge(model=judge_config.model)
        return OpenAIJudge()
    if judge_config.type == "ollama":
        from redforge.scoring.judges import OllamaJudge

        kwargs: dict[str, Any] = {}
        if judge_config.model is not None:
            kwargs["model"] = judge_config.model
        if judge_config.host is not None:
            kwargs["host"] = judge_config.host
        return OllamaJudge(**kwargs)
    raise ValueError(f"Unknown judge type: {judge_config.type!r}")
