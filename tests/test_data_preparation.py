import io
import json
import sys

import pytest

from scripts import prepare_data


@pytest.mark.parametrize("damage", ["size", "format"])
def test_download_manifest_and_verification(tmp_path, monkeypatch, damage):
    examples = [{"instruction": "Add one and two", "answer": "3"}]

    def download(url, timeout):
        content = (
            json.dumps(examples).encode() if url.endswith(".json") else b"License text"
        )
        return io.BytesIO(content)

    monkeypatch.setattr(prepare_data.urllib.request, "urlopen", download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--datasets", "math_10k", "--output-dir", str(tmp_path)],
    )
    prepare_data.main()
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["revision"] == "main"
    assert len(manifest["files"]) == 9
    for name, info in manifest["files"].items():
        assert set(info) == (
            {"url", "size_bytes", "examples"}
            if name.endswith(".json")
            else {"url", "size_bytes"}
        )

    monkeypatch.setattr(
        sys, "argv", ["prepare_data.py", "--verify-only", "--output-dir", str(tmp_path)]
    )
    prepare_data.main()
    path = tmp_path / "ft-training_set" / "math_10k.json"
    if damage == "size":
        path.write_text("[]")
        message = "File size mismatch"
    else:
        path.write_text("{}".ljust(path.stat().st_size))
        message = "Expected nonempty JSON examples"
    with pytest.raises(ValueError, match=message):
        prepare_data.main()
