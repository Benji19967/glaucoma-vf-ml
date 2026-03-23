import shutil
import subprocess
from pathlib import Path

import git
import requests


def get_git_root(path) -> Path:

    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return Path(git_root)


ROOT = get_git_root(__file__)
DATA_DIR = ROOT / "data"
GRAPE_DIR = DATA_DIR / "GRAPE"
UWHVF_DIR = DATA_DIR / "UWHVF"

RAR_FILES = [
    "CFPs.rar",
    "annotated_images.rar",
    "annotations_json.rar",
    "ROI_images.rar",
]

# Mapping: Figshare file ID -> local filename
FIGSHARE_FILES = {
    "41670009": "VFs_and_clinical_info.xlsx",
    "41358156": "CFPs.rar",
    "41358159": "annotated_images.rar",
    "41358162": "annotations_json.rar",
    "41358150": "ROI_images.rar",
}


def download_figshare():
    """
    Downloads missing files from Figshare API.
    """
    GRAPE_DIR.mkdir(parents=True, exist_ok=True)
    print("Checking GRAPE Figshare files...")

    for file_id, name in FIGSHARE_FILES.items():
        dest = GRAPE_DIR / name
        if not dest.exists():
            url = f"https://api.figshare.com/v2/file/download/{file_id}"
            print(f"Downloading {name} (ID: {file_id})...")

            # Use stream=True for large .rar files to save memory
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            print(f"{name} already exists, skipping.")


def extract_rar_files():
    """Extracts RAR files using unar."""
    print("\nExtracting RAR files...")

    # Locate unar
    extractor = shutil.which("unar")
    if not extractor:
        print("Error: 'unar' not found in PATH.")
        print(
            "   Install via: brew install unar (Mac) or sudo apt install unar (Linux/WSL)"
        )
        return

    for rar in RAR_FILES:
        path = GRAPE_DIR / rar
        if path.exists():
            print(f"  -> Extracting {rar}...")
            # -f forces overwrite, -o specifies output directory
            subprocess.run(
                [extractor, "-f", "-o", str(GRAPE_DIR), str(path)], check=True
            )
        else:
            print(f"Skipping {rar} (not found)")


def sync_uwhvf():
    """Clones or updates the UWHVF git repository."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not UWHVF_DIR.exists():
        print("Cloning UWHVF dataset...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/uw-biomedical-ml/uwhvf.git",
                str(UWHVF_DIR),
            ],
            check=True,
        )
    else:
        print("UWHVF already exists, pulling latest updates...")
        subprocess.run(["git", "-C", str(UWHVF_DIR), "pull"], check=True)


if __name__ == "__main__":
    try:
        download_figshare()
        print("-" * 30)
        extract_rar_files()
        print("-" * 30)
        sync_uwhvf()
        print("\nData setup complete!")
    except Exception as e:
        print(f"\nError during setup: {e}")
