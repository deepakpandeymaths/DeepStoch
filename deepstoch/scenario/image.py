"""Dependency-free image metadata extraction for future multimodal evidence."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"


@dataclass(frozen=True)
class ImageEvidence:
    path: str
    image_type: str
    width: int
    height: int
    file_size_bytes: int
    features: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != PNG_SIGNATURE or content[12:16] != b"IHDR":
        raise ValueError("PNG file is not valid")
    width, height = struct.unpack(">II", content[16:24])
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return width, height


def _jpeg_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 4 or content[:2] != JPEG_SIGNATURE:
        raise ValueError("JPEG file is not valid")
    index = 2
    while index + 9 < len(content):
        if content[index] != 0xFF:
            index += 1
            continue
        marker = content[index + 1]
        index += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(content):
            break
        segment_length = struct.unpack(">H", content[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(content):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                break
            height, width = struct.unpack(">HH", content[index + 3:index + 7])
            if width < 1 or height < 1:
                break
            return width, height
        index += segment_length
    raise ValueError("JPEG dimensions could not be read")


def extract_image_evidence(path: str | Path) -> ImageEvidence:
    """Extract reproducible metadata without interpreting image content."""
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"image does not exist: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGES:
        supported = ", ".join(sorted(SUPPORTED_IMAGES))
        raise ValueError(f"unsupported image type: {suffix or '<none>'}; use {supported}")
    content = image_path.read_bytes()
    if suffix == ".png":
        width, height = _png_dimensions(content)
        image_type = "png"
    else:
        width, height = _jpeg_dimensions(content)
        image_type = "jpeg"
    source_ref = str(image_path)
    registry = FeatureRegistry()
    metadata = [
        ("width_pixels", float(width), "header_dimension"),
        ("height_pixels", float(height), "header_dimension"),
        ("aspect_ratio", width / height, "derived_dimension_ratio"),
        ("file_size_bytes", float(len(content)), "filesystem_metadata"),
    ]
    for name, value, transformation in metadata:
        registry.add(EvidenceFeature(name, value, "image", source_ref, None, 1.0, transformation))
    return ImageEvidence(source_ref, image_type, width, height, len(content), registry.as_dict())
