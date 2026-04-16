import os
import shutil
import socket
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from re import L

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


def setup_and_extract_rar(rar_file_path, extraction_path="extracted_data"):
    """
    For Ubelix HPC cluster
    """
    # 1. Configuration
    url = "https://7-zip.org/a/7zz-linux-x64.tar.xz"
    binary_tar = "7zz-linux-x64.tar.xz"
    binary_name = "./7zz"

    print(f"--- Starting extraction process for: {rar_file_path} ---")

    # 2. Download 7-Zip binary if it doesn't exist
    if not os.path.exists(binary_name):
        print(f"Downloading 7-Zip from {url}...")
        urllib.request.urlretrieve(url, binary_tar)

        print("Extracting 7-Zip binary...")
        # Note: standard tarfile module handles .xz if lzma is available
        with tarfile.open(binary_tar, "r:xz") as tar:
            tar.extract("7zz")

        # Ensure it is executable
        os.chmod(binary_name, 0o755)
        os.remove(binary_tar)
        print("7-Zip setup complete.")

    for rar in RAR_FILES:
        path = GRAPE_DIR / rar
        print(f"Extracting {path}")
        try:
            # 4. Run the extraction command
            # 'x' = extract with full paths
            # '-o' = output directory (Note: no space after -o)
            # '-y' = assume Yes on all queries (non-interactive)
            cmd = [binary_name, "x", path, f"-o{str(GRAPE_DIR)}", "-y"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Success!")
        except subprocess.CalledProcessError as e:
            print(f"Error during extraction: {e}")
            print(f"Stderr: {e.stderr}")


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


def is_on_hpc():
    hostname = socket.gethostname()
    if "gnode" in hostname or "submit" in hostname:
        return True
    return False


if __name__ == "__main__":
    try:
        download_figshare()
        print("-" * 30)
        if is_on_hpc():
            setup_and_extract_rar()
        else:
            extract_rar_files()
        print("-" * 30)
        sync_uwhvf()
        print("\nData setup complete!")
    except Exception as e:
        print(f"\nError during setup: {e}")
