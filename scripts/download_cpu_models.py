"""Cache the small Qwen pair in WSL's standard Hugging Face cache."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    cache = Path.home() / ".cache/huggingface/hub"
    manifest = {}
    for repo in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        revision = HfApi().model_info(repo).sha
        print(f"Downloading {repo} at {revision} into {cache}", flush=True)
        path = snapshot_download(
            repo, revision=revision, cache_dir=str(cache), max_workers=2,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
        )
        manifest[repo] = {"revision": revision, "path": path}
        print(f"Ready: {repo}: {path}", flush=True)
    dest = Path(__file__).resolve().parents[1] / ".results/cpu_models.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Model revisions: {dest}", flush=True)


if __name__ == "__main__":
    main()
