# /// script
# dependencies = ["huggingface_hub>=0.30", "numpy>=1.26"]
# ///
"""CPU-only R6 private artifact audit. Pass HF_TOKEN via HF Jobs --secrets.
No training, model downloads, or secret output.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "codeflash85/ternary-pet-r5-checkpoints"
EXPECTED = [
    ("granite_seed170141/H_final_compact.npz", 50895933,
     "3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888", 170141, 168, 249561088),
    ("smol_seed190027/H_final_compact.npz", 64195086,
     "5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe", 190027, 224, 314572800),
]

def check_one(api, token, remote, expected_size, expected_sha, seed, layers, weights):
    local = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=remote, token=token))
    h = hashlib.sha256()
    with local.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    assert local.stat().st_size == expected_size, (remote, "size mismatch")
    assert h.hexdigest() == expected_sha, (remote, "SHA256 mismatch")
    with np.load(local, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest_utf8"].tobytes().decode("utf-8"))
        assert manifest["seed"] == seed
        assert len(manifest["layers"]) == layers
        assert manifest["quantized_weights"] == weights
        assert sum(item["count"] for item in manifest["layers"]) == weights
        for item in manifest["layers"]:
            key = item["key"]
            count = item["count"]
            assert int(np.prod(item["shape"])) == count
            packed = archive["codes_" + key]
            assert packed.size == (count + 3) // 4
            # Inspect each 2-bit code without instantiating a full decoded weight tensor.
            for shift in (0, 2, 4, 6):
                codes = (packed >> shift) & 3
                assert not np.any(codes == 3), (remote, key, "invalid ternary packing")
            alpha = archive["alpha_" + key]
            assert np.all(np.isfinite(alpha)) and np.all(alpha > 0), (remote, key, "invalid alpha")
    print("R6_SNAPSHOT_AUDIT_OK", json.dumps({
        "remote": remote, "bytes": expected_size, "sha256": expected_sha,
        "seed": seed, "layers": layers, "quantized_weights": weights,
    }), flush=True)

def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Missing HF_TOKEN secret; refusing unauthenticated private audit.")
    api = HfApi(token=token)
    info = api.repo_info(repo_id=REPO, repo_type="dataset")
    assert info.private is True, "Checkpoint dataset must remain private"
    for args in EXPECTED:
        check_one(api, token, *args)
    print("R6_BOTH_PRIVATE_SNAPSHOTS_VERIFIED", flush=True)

if __name__ == "__main__":
    main()
