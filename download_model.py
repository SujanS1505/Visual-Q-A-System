from huggingface_hub import snapshot_download

model_name = "Salesforce/blip2-opt-2.7b"

snapshot_download(
    repo_id=model_name,
    resume_download=True,       # Resume partial downloads
)
