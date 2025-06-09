import modal

hf_cache_vol = modal.Volume.from_name("b200-huggingface-cache", create_if_missing=True)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
MODEL_REVISION = "4236a6af538feda4548eca9ab308586007567f52"
