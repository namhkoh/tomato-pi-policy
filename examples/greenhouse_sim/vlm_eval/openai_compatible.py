"""Dependency-light client for OpenAI-compatible multimodal chat endpoints."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import mimetypes
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

from .prompt import SYSTEM_PROMPT
from .schema import response_json_schema


class EndpointError(RuntimeError):
    """An authenticated inference request failed."""

    def __init__(self, message: str, *, status: int | None = None, response_body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.response_body = response_body


@dataclasses.dataclass(frozen=True)
class EncodedImage:
    path: pathlib.Path
    width: int
    height: int
    sha256: str
    submitted_sha256: str
    submitted_mime_type: str
    transcoded: bool
    data_url: str


@dataclasses.dataclass(frozen=True)
class ChatResult:
    response: dict[str, Any]
    content: str
    latency_s: float
    structured_mode: str


def encode_image(
    path: str | pathlib.Path,
    *,
    submission_format: str = "jpeg",
    jpeg_quality: int = 92,
) -> EncodedImage:
    """Read a source image and prepare provider bytes without changing it."""

    from PIL import Image

    if submission_format not in {"jpeg", "original"}:
        raise ValueError("submission_format must be 'jpeg' or 'original'")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    resolved = pathlib.Path(path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    with Image.open(resolved) as image:
        width, height = image.size
        image.load()
        source_mime_type = Image.MIME.get(image.format or "") or mimetypes.guess_type(resolved.name)[0]
        if source_mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(f"unsupported image type: {source_mime_type}")
        if submission_format == "jpeg" and source_mime_type != "image/jpeg":
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality, subsampling=0)
            submitted = buffer.getvalue()
            submitted_mime_type = "image/jpeg"
            transcoded = True
        else:
            submitted = raw
            submitted_mime_type = source_mime_type
            transcoded = False
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return EncodedImage(
        path=resolved,
        width=width,
        height=height,
        sha256=hashlib.sha256(raw).hexdigest(),
        submitted_sha256=hashlib.sha256(submitted).hexdigest(),
        submitted_mime_type=submitted_mime_type,
        transcoded=transcoded,
        data_url=f"data:{submitted_mime_type};base64,{base64.b64encode(submitted).decode('ascii')}",
    )


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    image_data_url: str,
    max_tokens: int,
    temperature: float,
    structured_mode: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build one provider request; private simulator labels are not accepted."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if structured_mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "greenhouse_vlm_cutpoint_v1",
                "strict": True,
                "schema": response_json_schema(),
            },
        }
    elif structured_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif structured_mode != "none":
        raise ValueError(f"unsupported structured mode: {structured_mode}")
    return payload


class OpenAICompatibleVisionClient:
    """Call `/chat/completions` without adding an SDK to the project runtime."""

    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 90.0) -> None:
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key.strip()
        self._timeout_s = timeout_s

    def infer(
        self,
        *,
        model: str,
        prompt: str,
        image_data_url: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
        structured_mode: str = "auto",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> ChatResult:
        """Run inference, falling back only when structured-output syntax is unsupported."""

        modes = ("json_schema", "json_object", "none") if structured_mode == "auto" else (structured_mode,)
        last_error: EndpointError | None = None
        for index, mode in enumerate(modes):
            payload = build_chat_payload(
                model=model,
                prompt=prompt,
                image_data_url=image_data_url,
                max_tokens=max_tokens,
                temperature=temperature,
                structured_mode=mode,
                system_prompt=system_prompt,
            )
            try:
                response, latency_s = self._post(payload)
            except EndpointError as exc:
                last_error = exc
                structured_format_rejected = exc.status in {400, 404, 422} or (
                    exc.status == 500 and "InternalServerError" in exc.response_body
                )
                if (
                    structured_mode == "auto" and structured_format_rejected and index < len(modes) - 1
                ):
                    continue
                raise
            return ChatResult(
                response=response,
                content=_response_content(response),
                latency_s=latency_s,
                structured_mode=mode,
            )
        assert last_error is not None
        raise last_error

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "tomato-pi-policy-vlm-eval/1",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                response_bytes = response.read()
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")[:4000]
            raise EndpointError(
                f"inference endpoint returned HTTP {exc.code}",
                status=exc.code,
                response_body=response_body,
            ) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(f"could not reach inference endpoint: {exc.reason}") from exc
        latency_s = time.perf_counter() - started
        try:
            value = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise EndpointError("inference endpoint returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise EndpointError("inference endpoint returned a non-object JSON payload")
        return value, latency_s


def _response_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointError("response is missing choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    raise EndpointError("response message content is not text")
