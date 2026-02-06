import base64
import json
from pathlib import Path
from typing import Dict

import requests

from logging_config import get_logger


MODEL_URL = "http://0.0.0.0:8000/v1/chat/completions"


logger = get_logger(__name__)


def get_headers() -> Dict[str, str]:
    return {
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode it as base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_payload(model: str, image_path: str) -> dict:
    """Create payload for nemotron-parse model with local image file."""
    # Encode local image to base64
    base64_image = encode_image_to_base64(image_path)

    # Determine image format from file extension
    image_format = image_path.lower().split(".")[-1]
    if image_format == "jpg":
        image_format = "jpeg"

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}",
                        },
                    }
                ],
            },
        ],
        # set max output tokens to a high value to ensure we get the full response from the model
        "max_tokens": 4096,
    }


def _build_response_path(image_path: str) -> Path:
    """Return JSON output path next to the page image.

    For an image like /.../mydoc/mydoc_page-1.png, this yields
    /.../mydoc/mydoc_page-1.json.
    """
    img_path = Path(image_path)
    return img_path.with_suffix(".json")


def call_nemotron_parse(
    image_path: str,
    model: str = "nvidia/nemotron-parse",
    url: str = MODEL_URL,
    timeout: int = 60,
) -> dict:
    """Call the Nemotron-parse model for a single page image.

        The raw JSON response is also written to disk in the same
        directory as the page image, using the same basename with a
        .json extension.

    Returns the parsed JSON response from the model.
    """
    logger.info(
        f"Calling Nemotron-parse | image={image_path} | model={model} | url={url}"
    )

    payload = get_payload(model=model, image_path=image_path)
    headers = get_headers()

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    logger.info(f"Nemotron-parse HTTP response | status={response.status_code}")
    response.raise_for_status()
    response_json = response.json()

    # Persist the model response to disk next to the image
    out_path = _build_response_path(image_path=image_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(response_json, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved Nemotron-parse response JSON | path={out_path}")

    return response_json
