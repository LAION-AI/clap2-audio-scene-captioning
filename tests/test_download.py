import hashlib
import pytest
from clap2_scenes.index import download


def test_download_verifies_before_installing_and_repairs_corrupt_cache(tmp_path):
    source = tmp_path / "source.npy"
    source.write_bytes(b"valid matrix bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "cache" / "index.npy"
    assert download(source.as_uri(), target, digest).read_bytes() == source.read_bytes()
    target.write_bytes(b"corrupt")
    assert download(source.as_uri(), target, digest).read_bytes() == source.read_bytes()
    source.write_bytes(b"wrong upstream data")
    target.unlink()
    with pytest.raises(ValueError, match="checksum"):
        download(source.as_uri(), target, digest)
    assert not target.exists() and not list(target.parent.glob("*.tmp"))
