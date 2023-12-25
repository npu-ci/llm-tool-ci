"""
Test command line for model finetune.
"""

from fastchat.utils import run_cmd


def test_single_npu_fine_tune(cmd):
    model_path = "/opt/big_models/bloom-560m"
    print("test_single_npu_fine_tune: model_path:%s" % model_path)
    print("------" * 10)
    ret = run_cmd(cmd % (1, model_path))
    if ret != 0:
        raise RuntimeError("test %s in single npu error." % model_path)

    print("")


def test_multi_npu_fine_tune(cmd):
    models = [
        "/opt/big_models/bloom-560m",
        "/opt/big_models/gpt2",
    ]
    for model_path in models:
        print("test_multi_npu_fine_tune: model_path:%s" % model_path)
        print("------" * 10)
        if "llama" in model_path:
            _cmd = cmd + '  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"'
        else:
            _cmd = cmd
        ret = run_cmd(cmd % (2, model_path))
        if ret != 0:
            raise RuntimeError("test %s in multi npu error." % model_path)
        print("")


if __name__ == "__main__":
    rst = 0
    _cmd = (
        'torchrun --nproc_per_node=%d --master_port=20001 fastchat/train/train.py  '
        '--model_name_or_path  %s  '
        '--data_path /opt/nlp_data/evol-instruct-chinese--1-subset.json  '
        '--fp16 True  --output_dir output_model  --num_train_epochs 2  '
        '--per_device_train_batch_size 8  --per_device_eval_batch_size 1  '
        '--gradient_accumulation_steps 1  --evaluation_strategy "no"  '
        '--save_strategy "steps"  --save_steps 2000  '
        '--save_total_limit 200  --learning_rate 5e-5  --weight_decay 0.  '
        '--lr_scheduler_type "cosine"  --logging_steps 1  '
        '--fsdp "full_shard auto_wrap"  --model_max_length 1024  '
        '--gradient_checkpointing True  --lazy_preprocess True'
    )
    try:
        test_single_npu_fine_tune(_cmd)
    except RuntimeError as e:
        print(e)
        rst += 1
    try:
        test_multi_npu_fine_tune(_cmd)
    except RuntimeError as e:
        print(e)
        rst += 1
    if rst != 0:
        raise AssertionError("Total %d cases error" % rst)
