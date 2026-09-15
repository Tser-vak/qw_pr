FROM vllm/vllm-openai:nightly-dev-x86_64-cu13.0.1-591bb95
# Install the required plugin to load GGUF models in vLLM
RUN pip install --no-cache-dir --upgrade vllm-gguf-plugin

# Patch vllm-gguf-plugin to support qwen3_5 architecture
COPY patch.py .
RUN python3 patch.py
