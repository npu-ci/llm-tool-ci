"""
Launch an OpenAI API server with multiple model workers.
"""
import sys

sys.path.append("")
import os


def launch_process(cmd):
    os.popen(cmd)


if __name__ == "__main__":
    launch_process("python3 -m fastchat.serve.controller")
    launch_process("python3 -m fastchat.serve.openai_api_server")

    models = [
        ("/opt/big_models/bert-base-cased", "model_worker"),
        ("/opt/big_models/bloom-560m", "model_worker"),
        # npu not support vllm
        # ("/opt/big_models/gt2", "vllm_worker"),
    ]

    for i, (model_path, worker_name) in enumerate(models):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={i} python3 -m fastchat.serve.{worker_name} "
            f"--model-path {model_path} --device npu --port {30000 + i} "
            f"--worker-address http://localhost:{30000 + i} "
        )
        if worker_name == "vllm_worker":
            cmd += "--tokenizer hf-internal-testing/llama-tokenizer"

        launch_process(cmd)

    while True:
        pass
