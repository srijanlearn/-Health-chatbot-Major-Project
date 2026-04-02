# app/llm_engine.py
"""
Unified Local LLM Engine for HealthyPartner v2.

Hardware-aware model selection, dual-model inference
(fast classifier + quality generator), Ollama integration
with health checks and auto-download.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import psutil

try:
    import ollama
    from ollama import Client as OllamaClient

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Model Tier Definitions ────────────────────────────────────────────────────

MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "ultra_light": {
        "main_model": "qwen2.5:1.5b",
        "fast_model": "qwen2.5:0.5b",
        "min_ram_gb": 4,
        "description": "Ultra-light: Raspberry Pi / IoT (4 GB RAM min)",
    },
    "balanced": {
        "main_model": "qwen3:4b",
        "fast_model": "qwen2.5:0.5b",
        "min_ram_gb": 8,
        "description": "Balanced: Laptops / PCs (8 GB RAM recommended)",
    },
    "quality": {
        "main_model": "qwen3:8b",
        "fast_model": "qwen2.5:1.5b",
        "min_ram_gb": 16,
        "description": "Quality: Best accuracy (16 GB RAM recommended)",
    },
}

DEFAULT_TIER = "balanced"


# ── System Info ───────────────────────────────────────────────────────────────


@dataclass
class SystemInfo:
    """Snapshot of the host machine's hardware capabilities."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    cpu_brand: str
    gpu_available: bool
    gpu_name: Optional[str]
    os_name: str
    arch: str
    recommended_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_gb": round(self.total_ram_gb, 1),
            "available_ram_gb": round(self.available_ram_gb, 1),
            "cpu_count": self.cpu_count,
            "cpu_brand": self.cpu_brand,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "os_name": self.os_name,
            "arch": self.arch,
            "recommended_tier": self.recommended_tier,
        }


def detect_system() -> SystemInfo:
    """Detect hardware capabilities and recommend a model tier."""
    import platform as _platform

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    avail_gb = mem.available / (1024**3)

    # CPU info
    cpu_count = psutil.cpu_count(logical=True) or 1
    try:
        cpu_brand = _platform.processor() or "Unknown"
    except Exception:
        cpu_brand = "Unknown"

    # GPU detection (best-effort)
    gpu_available = False
    gpu_name = None
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_available = True
            gpu_name = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    # Check for Apple Silicon GPU (Metal)
    if not gpu_available and _platform.system() == "Darwin":
        if "arm" in _platform.machine().lower():
            gpu_available = True
            gpu_name = "Apple Silicon (Metal)"

    # Recommend tier based on available RAM
    if total_gb >= 16:
        recommended = "quality"
    elif total_gb >= 8:
        recommended = "balanced"
    else:
        recommended = "ultra_light"

    return SystemInfo(
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        cpu_count=cpu_count,
        cpu_brand=cpu_brand,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        os_name=_platform.system(),
        arch=_platform.machine(),
        recommended_tier=recommended,
    )


# ── LLM Engine ────────────────────────────────────────────────────────────────


class LLMEngine:
    """
    Unified local LLM engine with dual-model architecture.

    - **Fast model** (e.g. Qwen 0.5B): intent classification, routing, scoring
    - **Main model** (e.g. Qwen3 4B): answer generation, reasoning, analysis

    All inference runs through Ollama on localhost.
    """

    def __init__(
        self,
        tier: Optional[str] = None,
        main_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "The 'ollama' package is required. Install it with: pip install ollama"
            )

        self._host = ollama_host
        self._client = OllamaClient(host=ollama_host)
        self._system_info = detect_system()

        # Resolve tier
        resolved_tier = tier or os.getenv("HP_MODEL_TIER") or self._system_info.recommended_tier
        if resolved_tier not in MODEL_TIERS:
            logger.warning("Unknown tier '%s', falling back to '%s'", resolved_tier, DEFAULT_TIER)
            resolved_tier = DEFAULT_TIER
        self._tier = resolved_tier
        tier_config = MODEL_TIERS[self._tier]

        # Model names (explicit overrides take precedence)
        self.main_model = main_model or os.getenv("HP_MAIN_MODEL") or tier_config["main_model"]
        self.fast_model = fast_model or os.getenv("HP_FAST_MODEL") or tier_config["fast_model"]

        # Generation defaults
        self._main_opts = {
            "temperature": 0.3,
            "num_ctx": 4096,
            "num_predict": 512,
        }
        self._fast_opts = {
            "temperature": 0.0,
            "num_ctx": 2048,
            "num_predict": 64,
        }

        logger.info(
            "LLMEngine initialised — tier=%s, main=%s, fast=%s",
            self._tier,
            self.main_model,
            self.fast_model,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def system_info(self) -> SystemInfo:
        return self._system_info

    # ── Core inference methods ─────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        system_prompt: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run a fast classification/routing task using the lightweight model.

        Designed for: intent detection, question routing, confidence scoring.
        Typical latency: <100ms.
        """
        opts = {**self._fast_opts, **(options or {})}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        try:
            response = self._client.chat(
                model=self.fast_model,
                messages=messages,
                options=opts,
            )
            return response["message"]["content"].strip()
        except Exception:
            logger.exception("classify() failed on fast model '%s'", self.fast_model)
            # Fallback: try with main model
            try:
                response = self._client.chat(
                    model=self.main_model,
                    messages=messages,
                    options=opts,
                )
                return response["message"]["content"].strip()
            except Exception:
                logger.exception("classify() fallback also failed")
                return ""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[Dict[str, Any]] = None,
        think: bool = False,
    ) -> str:
        """
        Generate a full response using the main (quality) model.

        Designed for: answer generation, document analysis, medical reasoning.

        Args:
            prompt: The user question / task.
            system_prompt: System-level instructions.
            context: Retrieved document context to ground the answer.
            history: Prior conversation turns [{"role": "user"|"assistant", "content": "..."}].
            options: Override generation parameters.
            think: If True, prepend a "think step-by-step" instruction (Qwen3 think mode).
        """
        opts = {**self._main_opts, **(options or {})}
        messages: List[Dict[str, str]] = []

        # System prompt
        sys_content = system_prompt
        if think:
            sys_content += "\n\nThink step-by-step before answering."
        if sys_content:
            messages.append({"role": "system", "content": sys_content})

        # Conversation history
        if history:
            messages.extend(history)

        # Build user message with context
        user_content = prompt
        if context:
            user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"
        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat(
                model=self.main_model,
                messages=messages,
                options=opts,
            )
            return response["message"]["content"].strip()
        except Exception:
            logger.exception("generate() failed on main model '%s'", self.main_model)
            return "I'm sorry, I encountered an error generating a response. Please try again."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Stream tokens from the main model. Yields one token/chunk at a time.
        """
        opts = {**self._main_opts, **(options or {})}
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)

        user_content = prompt
        if context:
            user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"
        messages.append({"role": "user", "content": user_content})

        try:
            stream = self._client.chat(
                model=self.main_model,
                messages=messages,
                options=opts,
                stream=True,
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception:
            logger.exception("generate_stream() failed")
            yield "I'm sorry, I encountered an error. Please try again."

    # ── Model management ───────────────────────────────────────────────────────

    def list_local_models(self) -> List[str]:
        """Return names of models already downloaded in Ollama."""
        try:
            response = self._client.list()
            return [m.get("name", m.get("model", "")) for m in response.get("models", [])]
        except Exception:
            logger.exception("Failed to list local models")
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is downloaded locally."""
        local = self.list_local_models()
        # Match with or without tag (e.g. "qwen3:4b" matches "qwen3:4b")
        return any(model_name in m or m.startswith(model_name.split(":")[0]) for m in local)

    def ensure_models_available(self) -> Dict[str, bool]:
        """
        Check that both fast and main models are available.
        Returns status dict. Does NOT auto-download (that requires user consent).
        """
        status = {
            "main_model": self.is_model_available(self.main_model),
            "fast_model": self.is_model_available(self.fast_model),
            "main_model_name": self.main_model,
            "fast_model_name": self.fast_model,
        }

        if not status["main_model"]:
            logger.warning(
                "Main model '%s' not found. Run: ollama pull %s",
                self.main_model,
                self.main_model,
            )
        if not status["fast_model"]:
            logger.warning(
                "Fast model '%s' not found. Run: ollama pull %s",
                self.fast_model,
                self.fast_model,
            )

        return status

    def pull_model(self, model_name: str) -> bool:
        """Download a model via Ollama. Blocks until complete."""
        try:
            logger.info("Pulling model '%s' — this may take a few minutes...", model_name)
            self._client.pull(model_name)
            logger.info("✅ Model '%s' downloaded successfully", model_name)
            return True
        except Exception:
            logger.exception("Failed to pull model '%s'", model_name)
            return False

    # ── Health checks ──────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """
        Full system health check: Ollama connectivity, model availability, hardware.
        """
        result: Dict[str, Any] = {
            "healthy": False,
            "ollama_connected": False,
            "models": {},
            "system": self._system_info.to_dict(),
            "tier": self._tier,
        }

        # Check Ollama connectivity
        try:
            self._client.list()
            result["ollama_connected"] = True
        except Exception as exc:
            result["ollama_error"] = str(exc)
            return result

        # Check models
        result["models"] = self.ensure_models_available()
        result["healthy"] = (
            result["ollama_connected"]
            and result["models"].get("main_model", False)
            and result["models"].get("fast_model", False)
        )

        # Quick inference test if models are available
        if result["healthy"]:
            try:
                start = time.time()
                test_response = self.classify("Say OK")
                latency_ms = (time.time() - start) * 1000
                result["inference_test"] = {
                    "passed": bool(test_response),
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception:
                result["inference_test"] = {"passed": False}

        return result

    # ── LangChain compatibility layer ──────────────────────────────────────────

    def as_langchain_llm(self, model: str = "main"):
        """
        Return a LangChain-compatible ChatOllama instance.

        Args:
            model: "main" for quality model, "fast" for classifier model.
        """
        from langchain_community.chat_models import ChatOllama

        if model == "fast":
            return ChatOllama(
                model=self.fast_model,
                base_url=self._host,
                temperature=self._fast_opts["temperature"],
                num_ctx=self._fast_opts["num_ctx"],
                num_predict=self._fast_opts["num_predict"],
            )
        else:
            return ChatOllama(
                model=self.main_model,
                base_url=self._host,
                temperature=self._main_opts["temperature"],
                num_ctx=self._main_opts["num_ctx"],
                num_predict=self._main_opts["num_predict"],
            )

    def __repr__(self) -> str:
        return (
            f"LLMEngine(tier={self._tier!r}, "
            f"main={self.main_model!r}, fast={self.fast_model!r})"
        )
