"""
Test command line interface for model inference.
"""
import sys
import json
from fastchat.utils import run_cmd

infer_cmd = (
    f"python3 -m fastchat.serve.cli --model-path output_model --device npu "
    f"--style programmatic --num-gpus 2 --max-gpu-memory 14Gib < test_cli_inputs.txt"
)
finetune_cmd = (
    'torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train.py  '
    '--model_name_or_path  %s  '
    '--data_path /opt/nlp_data/evol-instruct-chinese--1-subset.json  '
    '--fp16 True  --output_dir output_model  --num_train_epochs 1  '
    '--per_device_train_batch_size 8  --per_device_eval_batch_size 1  '
    '--gradient_accumulation_steps 1  --evaluation_strategy "no"  '
    '--save_strategy "steps"  --save_steps 2000  '
    '--save_total_limit 200  --learning_rate 5e-5  --weight_decay 0.  '
    '--lr_scheduler_type "cosine"  --logging_steps 1  '
    '--fsdp "full_shard auto_wrap"  --model_max_length 1024  '
    '--gradient_checkpointing True  --lazy_preprocess True'
)


def test_multi_npu(f_c, i_c):
    # 训练
    ret = run_cmd(f_c)
    if ret != 0:
        raise RuntimeError("finetune %s error." % model_path)
    # 推理
    ret = run_cmd(i_c)
    if ret != 0:
        raise RuntimeError("inference %s's finetune model error." % model_path)
    print("---" * 20)


if __name__ == "__main__":
    rst = 0
    args = sys.argv[1:]
    json_file = args[0]
    result_dict = {}
    with open(json_file, 'r') as fp:
        models = json.load(fp)
        for model_name, model_path in models.items():
            print(model_path)
            _fine_cmd = finetune_cmd % model_path
            if "llama" in model_name or "Llama" in model_name:
                _fine_cmd = _fine_cmd + "  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'"
            try:
                test_multi_npu(_fine_cmd, infer_cmd)
                result_dict[model_name] = True
            except Exception as e:
                print(e)
                rst += 1
                result_dict[model_name] = False
    result_path = "fastchat-models.json"
    with open(result_path, "w") as rfp:
        json.dump(result_dict, rfp)

    if rst != 0:
        print("total %d model error" % rst)
