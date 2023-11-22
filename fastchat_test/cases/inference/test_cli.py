"""
Test command line interface for model inference.
"""
import argparse
import os

from fastchat.utils import run_cmd


def test_single_npu():
    models = [
        "/opt/big_models/bloom-560m",
        "/opt/big_models/gpt2",
    ]

    for model_path in models:
        if "model_weights" in model_path and not os.path.exists(
                os.path.expanduser(model_path)
        ):
            continue
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} --device npu "
            f"--style programmatic < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            raise RuntimeError("test %s in single npu error." % model_path)

        print("")


def test_multi_npu():
    models = [
        "/opt/big_models/bloom-560m",
        "/opt/big_models/gpt2",
    ]

    for model_path in models:
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} --device npu "
            f"--style programmatic --num-gpus 2 --max-gpu-memory 14Gib < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            raise RuntimeError("test %s in multi npu error." % model_path)
        print("")


def test_8bit():
    models = [
        "/opt/big_models/bloom-560m"
    ]

    for model_path in models:
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} --device npu "
            f"--style programmatic --load-8bit < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            raise RuntimeError("test %s in load 8bit error." % model_path)
        print("")


def test_hf_api():
    models = [
        "/opt/big_models/bloom-560m",
        "/opt/big_models/gpt2",
    ]

    for model_path in models:
        cmd = f"python3 -m fastchat.serve.huggingface_api --model-path {model_path} --device npu"
        ret = run_cmd(cmd)
        if ret != 0:
            raise RuntimeError("test %s for hf api error." % model_path)
        print("")


if __name__ == "__main__":
    rst = 0
    try:
        test_single_npu()
    except RuntimeError as e:
        print(e)
        rst += 1
    try:
        test_multi_npu()
    except RuntimeError as e:
        print(e)
        rst += 1
    # not support now 20231109
    # try:
    #    test_8bit()
    # except RuntimeError as e:
    #    print(e)
    #    rst += 1
    try:
        test_hf_api()
    except RuntimeError as e:
        print(e)
        rst += 1
    if rst != 0:
        raise AssertionError("%d case error" % rst)
