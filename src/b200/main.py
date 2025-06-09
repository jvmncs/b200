import modal

from b200 import common

app = modal.App("b200")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "pip install -U uv",
        "uv pip install --system huggingface_hub[hf_xet]",
        "uv pip install --system --torch-backend=cu128 vllm==0.9.0.1",
        "uv pip install --system https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl",
    )
    .env(
        {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_USE_V1_ENGINE": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)
VLLM_PORT = 8000
API_KEY = "wow-cool-huh"
vllm_cache_vol = modal.Volume.from_name("b200-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200:8",
    scaledown_window=15 * 60,
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": common.hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=60 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level",
        "info",
        common.MODEL_NAME,
        "--revision",
        common.MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
        "--tensor-parallel-size",
        "8",
    ]

    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def test(test_timeout=60 * 60):
    import json
    import time
    import urllib

    print(f"Running health check for server at {serve.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            print("trying request")
            with urllib.request.urlopen(serve.get_web_url() + "/health") as response:
                print("request finished")
                if response.getcode() == 200:
                    up = True
        except Exception as e:
            if time.time() - start > test_timeout:
                break
            print(e)
            time.sleep(delay)

    assert up, f"Failed health check for server at {serve.get_web_url()}"

    print(f"Successful health check for server at {serve.get_web_url()}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {serve.get_web_url()}", *messages, sep="\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"messages": messages, "model": common.MODEL_NAME})
    req = urllib.request.Request(
        serve.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))
