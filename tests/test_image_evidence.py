import struct

import pytest

from deepstoch.scenario import extract_image_evidence


def _minimal_png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"0000"


def test_png_metadata_becomes_image_features(tmp_path) -> None:
    path = tmp_path / "scene.png"
    path.write_bytes(_minimal_png(640, 480))

    result = extract_image_evidence(path)

    assert result.image_type == "png"
    assert (result.width, result.height) == (640, 480)
    assert result.features[0]["source_type"] == "image"
    assert result.features[0]["source_ref"] == str(path)
    assert any(feature["name"] == "aspect_ratio" and feature["value"] == 4 / 3 for feature in result.features)


def test_image_adapter_rejects_invalid_files(tmp_path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a png")

    with pytest.raises(ValueError, match="PNG file is not valid"):
        extract_image_evidence(path)
