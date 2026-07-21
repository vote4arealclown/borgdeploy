"""Pollinations.ai image generation skill for Borg."""
from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

from borg.config import settings
from borg.events import EventLog, event_log
from borg.metrics import image_generation_latency_seconds, timed
from borg.safety import ActionKind, SafetyGate, safety
from borg.schemas import ImageGenerationInput, ImageGenerationResult


class PollinationsImageClient:
    """Generate images via the Pollinations.ai API.

    Supports the simple GET ``/image/{prompt}`` endpoint for shareable URLs
    and the OpenAI-compatible ``POST /v1/images/generations`` endpoint for
    base64 data or explicit sizing.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        default_model: Optional[str] = None,
        image_dir: Optional[Path] = None,
        events: EventLog = event_log,
        safety_gate: SafetyGate = safety,
    ) -> None:
        self.base_url = (base_url or settings.pollinations_base_url).rstrip("/")
        self.api_key = api_key or settings.pollinations_api_key or os.environ.get(settings.pollinations_api_key_env, "")
        self.timeout = timeout if timeout is not None else settings.pollinations_timeout_seconds
        self.default_model = default_model or settings.pollinations_default_model
        self.image_dir = image_dir or settings.pollinations_image_dir
        self.events = events
        self.safety = safety_gate
        self.enabled = settings.pollinations_enabled

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _decision(self, detail: dict[str, Any]) -> dict[str, Any]:
        return self.safety.check(ActionKind.IMAGE_GENERATION, detail)

    async def list_models(self) -> list[dict[str, Any]]:
        """Return available image/video models from Pollinations."""
        url = f"{self.base_url}/image/models"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return list(resp.json())
        except Exception as exc:
            self.events.emit(f"Pollinations models fetch failed: {exc}", category="image_gen", level="WARN")
            return []

    async def generate_url(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> ImageGenerationResult:
        """Generate a shareable image URL via GET /image/{prompt}."""
        detail = {"prompt": prompt, "model": model, "width": width, "height": height}
        decision = self._decision(detail)
        if not decision["approved"]:
            return ImageGenerationResult(
                status="needs_confirmation",
                prompt=prompt,
                model=model or self.default_model,
                error="image_generation action requires confirmation",
            )

        if not self.enabled:
            return ImageGenerationResult(
                status="disabled",
                prompt=prompt,
                model=model or self.default_model,
                error="Pollinations image generation is disabled in config",
            )

        chosen_model = model or self.default_model
        params: dict[str, Any] = {"model": chosen_model, "width": width, "height": height}
        if seed is not None:
            params["seed"] = seed
        if self.api_key:
            params["key"] = self.api_key
        query = "&".join(f"{k}={v}" for k, v in params.items())
        encoded_prompt = quote(prompt, safe="")
        url = f"{self.base_url}/image/{encoded_prompt}?{query}"

        with timed(image_generation_latency_seconds):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
            except Exception as exc:
                msg = f"Pollinations image generation failed: {type(exc).__name__}: {exc}"
                self.events.emit(msg, category="image_gen", level="ERROR")
                return ImageGenerationResult(
                    status="error",
                    prompt=prompt,
                    model=chosen_model,
                    error=msg,
                )

        self.events.emit(f"Generated image URL for prompt: {prompt[:80]}", category="image_gen")
        return ImageGenerationResult(
            status="ok",
            prompt=prompt,
            model=chosen_model,
            url=url,
        )

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        save: bool = True,
    ) -> ImageGenerationResult:
        """Generate an image via POST /v1/images/generations and optionally save it locally."""
        detail = {"prompt": prompt, "model": model, "width": width, "height": height}
        decision = self._decision(detail)
        if not decision["approved"]:
            return ImageGenerationResult(
                status="needs_confirmation",
                prompt=prompt,
                model=model or self.default_model,
                error="image_generation action requires confirmation",
            )

        if not self.enabled:
            return ImageGenerationResult(
                status="disabled",
                prompt=prompt,
                model=model or self.default_model,
                error="Pollinations image generation is disabled in config",
            )

        chosen_model = model or self.default_model
        payload = {
            "prompt": prompt,
            "model": chosen_model,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
            "n": 1,
        }
        if seed is not None:
            payload["seed"] = seed

        with timed(image_generation_latency_seconds):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/v1/images/generations",
                        json=payload,
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                msg = f"Pollinations image generation failed: {type(exc).__name__}: {exc}"
                self.events.emit(msg, category="image_gen", level="ERROR")
                return ImageGenerationResult(
                    status="error",
                    prompt=prompt,
                    model=chosen_model,
                    error=msg,
                )

        b64 = data.get("data", [{}])[0].get("b64_json", "")
        revised_prompt = data.get("data", [{}])[0].get("revised_prompt", "")
        local_path: Optional[str] = None
        if b64 and save:
            local_path = self._save_image(b64, chosen_model)

        self.events.emit(
            f"Generated image for prompt: {prompt[:80]}",
            category="image_gen",
            metadata={"model": chosen_model, "path": local_path, "revised_prompt": revised_prompt[:200]},
        )
        return ImageGenerationResult(
            status="ok",
            prompt=prompt,
            model=chosen_model,
            b64_json=b64 if not save else None,
            local_path=local_path,
            usage=data.get("usage"),
        )

    async def generate_from_schema(
        self,
        request: ImageGenerationInput,
        save: bool = True,
    ) -> ImageGenerationResult:
        """Convenience wrapper that accepts a validated schema."""
        if request.response_format == "url":
            return await self.generate_url(
                prompt=request.prompt,
                model=request.model,
                width=request.width,
                height=request.height,
                seed=request.seed,
            )
        return await self.generate(
            prompt=request.prompt,
            model=request.model,
            width=request.width,
            height=request.height,
            seed=request.seed,
            save=save,
        )

    def _save_image(self, b64_data: str, model: str) -> str:
        """Persist a base64 image to the configured image directory."""
        out_dir = Path(self.image_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"borg_{model}_{timestamp}.png"
        path = out_dir / filename
        path.write_bytes(base64.b64decode(b64_data))
        return str(path)


image_client = PollinationsImageClient()
