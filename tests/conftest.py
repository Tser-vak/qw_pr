import pytest
import os
from openai import OpenAI

@pytest.fixture
def vllm_base_url():
    # When running tests locally, use localhost:8000. 
    # When in docker, use the env var if provided.
    return os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

@pytest.fixture
def model_name():
    return os.getenv("MODEL_NAME", "qwen3-27b")

@pytest.fixture
def openai_client(vllm_base_url):
    return OpenAI(api_key="EMPTY", base_url=vllm_base_url)
