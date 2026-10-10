# /// script
# dependencies = ["huggingface_hub>=0.30"]
# ///
"""CPU-only, secret-backed verification of private HF dataset checkpoint uploads.

Launch from authenticated local HF CLI with:
  hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN https://raw.githubusercontent.com/steveonw/hackathon/main/ternary_pet/r5_checkpoint_remote_upload_smoketest.py

No model downloads, GPUs, or checkpoint content. Never prints the token.
"""
import os
from huggingface_hub import HfApi

REPO = "codeflash85/ternary-pet-r5-checkpoints"
PATH = "verification/remote_cpu_secret_check.txt"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("FAIL: HF_TOKEN was not injected into this remote job")
    api = HfApi(token=token)
    who = api.whoami()
    username = who.get("name")
    if username != "codeflash85":
        raise SystemExit(f"FAIL: authenticated as unexpected account: {username!r}")
    info = api.repo_info(repo_id=REPO, repo_type="dataset")
    if info.private is not True:
        raise SystemExit("FAIL: destination dataset is not private")
    content = b"Ternary Pet R5 remote CPU secret and private dataset upload OK\n"
    api.upload_file(
        path_or_fileobj=__import__("io").BytesIO(content),
        path_in_repo=PATH,
        repo_id=REPO,
        repo_type="dataset",
        commit_message="Verify R5 job-secret private snapshot write",
    )
    remote = api.hf_hub_download(repo_id=REPO, filename=PATH, repo_type="dataset")
    with open(remote, "rb") as f:
        if f.read() != content:
            raise SystemExit("FAIL: uploaded file did not round-trip")
    print("R5_REMOTE_CHECKPOINT_UPLOAD_OK")
    print(f"repo={REPO} private=True file={PATH} account={username}")

if __name__ == "__main__":
    main()
