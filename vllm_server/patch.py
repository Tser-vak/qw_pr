import sys
import re

path = "/usr/local/lib/python3.12/dist-packages/vllm_gguf_plugin/weights_adapter/default.py"

with open(path, "r") as f:
    code = f.read()

new_code = """if model_type == "qwen3_5":
            model_type = "qwen2"
        else:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")"""

# Try exact double quotes
old_code_1 = 'raise RuntimeError(f"Unknown gguf model_type: {model_type}")'
# Try exact single quotes
old_code_2 = "raise RuntimeError(f'Unknown gguf model_type: {model_type}')"

if old_code_1 in code:
    code = code.replace(old_code_1, new_code)
elif old_code_2 in code:
    code = code.replace(old_code_2, new_code)
else:
    # Fallback to Regex just in case
    code, count = re.subn(r"raise RuntimeError\(f.Unknown gguf model_type: \{model_type\}.*\)", new_code, code)
    if count == 0:
        print("FATAL: Could not find the exception line to patch in default.py")
        sys.exit(1)

with open(path, "w") as f:
    f.write(code)
print("Successfully patched vllm-gguf-plugin for qwen3_5 support!")
