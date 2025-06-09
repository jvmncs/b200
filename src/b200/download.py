import pathlib

import modal

from b200 import common

app = modal.App("b200")
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "pip install -U uv",
        "uv pip install --system huggingface_hub[hf_xet,hf_transfer]",
    )
)
HUGGINGFACE_CACHE_PATH = pathlib.Path("/root") / ".cache" / "huggingface"


@app.function(
    image=download_image,
    volumes={
        HUGGINGFACE_CACHE_PATH.as_posix(): common.hf_cache_vol,
    },
)
def clear_volume():
    import shutil

    print(f"Clearing all contents from {str(HUGGINGFACE_CACHE_PATH)}...")
    common.hf_cache_vol.reload()
    if HUGGINGFACE_CACHE_PATH.exists():
        for item in HUGGINGFACE_CACHE_PATH.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"Successfully cleared {HUGGINGFACE_CACHE_PATH}.")
    else:
        print(f"{HUGGINGFACE_CACHE_PATH} does not exist, nothing to clear.")
    common.hf_cache_vol.commit()


@app.function(
    image=download_image,
    volumes={
        "/root/.cache/huggingface": common.hf_cache_vol,
    },
    timeout=120 * 60,
)
def download():
    from huggingface_hub import snapshot_download

    common.hf_cache_vol.reload()
    snapshot_download(repo_id=common.MODEL_NAME, revision=common.MODEL_REVISION)
    common.hf_cache_vol.commit()


@app.local_entrypoint()
def main():
    print("Clearing volume...")
    clear_volume.remote()

    print("Downloading weights to volume...", end="")
    download.remote()
    print("downloaded!")
