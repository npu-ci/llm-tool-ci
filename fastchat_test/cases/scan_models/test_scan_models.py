"""
Test command line interface for model inference.
"""
import json
import sys
import traceback

from fastchat.utils import run_cmd

infer_cmd = (
    f"python3 -m fastchat.serve.cli --model-path output_model --device npu "
    f"--style programmatic --num-gpus 2 --max-gpu-memory 14Gib < tests/test_cli_inputs.txt"
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
    '--lazy_preprocess True'
)

use_gradient_checkpointing = "  --gradient_checkpointing True"
save_logfile = "> %s_finetune.log"


def test_multi_npu(m_name, f_c, i_c):
    # 训练
    ret = run_cmd(f_c)
    if ret != 0:
        raise RuntimeError("finetune %s error." % model_path)
    with open("%s_finetune.log" % m_name, mode='r') as f:
        line = f.readline()
        while line:
            if "'loss': 0.0," in line or "'loss': -":
                raise ValueError("Got loss <=0.0, some errors caused by precision have occurred.")
            line = f.readline()
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
            if "cpm-ant" in model_name:
                _fine_cmd += "  --fsdp_transformer_layer_cls_to_wrap 'CpmAntSegmentPositionEmbedding'"
                continue
            if "llama" in model_name or "Llama" in model_name:
                _fine_cmd += "  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'"
            if "bert" in model_name:
                _fine_cmd += "  --fsdp_transformer_layer_cls_to_wrap 'BertLayer'"
            if "fuyu" in model_name:
                _fine_cmd += "  --fsdp_transformer_layer_cls_to_wrap 'PersimmonDecoderLayer'"
            _fine_cmd += use_gradient_checkpointing
            _fine_cmd += save_logfile % model_name
            try:
                test_multi_npu(_fine_cmd, infer_cmd, model_name)
                result_dict[model_name] = True
            except Exception as e:
                traceback.print_exc()
                rst += 1
                result_dict[model_name] = False
    result_path = "fastchat-models.json"
    with open(result_path, "w") as rfp:
        json.dump(result_dict, rfp)

    if rst != 0:
        print("total %d model error" % rst)
