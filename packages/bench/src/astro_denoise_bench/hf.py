"""Hugging Face distribution of the curated DC2 block.

The curated block is published once by the dataset owner
(``push_curated``, needs a write token in ``HF_TOKEN``) and downloaded
by every benchmark user (``fetch_curated``, public read). The block is
self-contained — regions, trimmed truth catalogs and manifest with
relative paths — so it works unmodified on any machine.

The trained method weights follow the same channel as model repos.
"""

from __future__ import annotations

from pathlib import Path

MANIFEST_NAME = "manifest.json"


def fetch_curated(repo_id, out_dir, revision=None, token=None):
    """Download the curated block from a HF dataset repo.

    Returns the local manifest path. Files already present and current
    are reused (snapshot download is incremental).
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(out_dir),
    )
    manifest = Path(local) / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(f"{repo_id} does not contain {MANIFEST_NAME}; not a curated block")
    return manifest


def push_curated(repo_id, curated_dir, token=None, private=False):
    """Upload a curated block directory to a HF dataset repo (owner only).

    Creates the repo when missing. ``token`` falls back to the
    ``HF_TOKEN`` env var / cached login of huggingface_hub.
    """
    curated_dir = Path(curated_dir)
    if not (curated_dir / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"{curated_dir} has no {MANIFEST_NAME}; curate first")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(curated_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="curated DC2 block",
        ignore_patterns=["denoised_*", "metrics_*", "previews_*", "report*.json"],
    )
    return f"https://huggingface.co/datasets/{repo_id}"
