"""Download the pre-trained UNet model from GitHub Releases.

Usage:
    python download_model.py

The model is required to run the binarization pipeline.
It will be saved to pretrained_models/unet_220805.pth (~118 MB).
"""
import urllib.request
import sys
from pathlib import Path

owner, repo = "NMar33", "layerseg"
REPO = f"{owner}/{repo}"
VERSION = "v1.0"
MODEL_URL = f"https://github.com/{REPO}/releases/download/{VERSION}/unet_220805.pth"

MODEL_DIR = Path(__file__).parent / "pretrained_models"
MODEL_PATH = MODEL_DIR / "unet_220805.pth"
EXPECTED_SIZE_MB = 118  # approximate expected file size


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({percent}%)")
    else:
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
    sys.stdout.flush()


def download_model():
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from GitHub Releases...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Destination: {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress_hook)
        print()  # newline after progress
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"\nYou can download the model manually from:")
        print(f"  {MODEL_URL}")
        print(f"and place it at: {MODEL_PATH}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        sys.exit(1)

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    if size_mb < EXPECTED_SIZE_MB * 0.9:
        print(f"WARNING: Downloaded file is {size_mb:.1f} MB, expected ~{EXPECTED_SIZE_MB} MB.")
        print("The file may be corrupted. Try downloading again.")
    else:
        print(f"Download complete: {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_model()
