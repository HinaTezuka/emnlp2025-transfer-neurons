cd analysis/reasoning
source venv_for_eval_harness/bin/activate
module load cuda/12.1

export CUDA_VISIBLE_DEVICES=0
export model_id="model_id/at/huggingface_hub"
export model_type="model_nickname"
export lang=ja

# baseline (random n. deactivate.)
export steer_config_path=steer_configs/${lang}_baseline.pt # make this .pt file with "analysis/reasoning/make_steer_config.py".

python -m lm_eval \
    --model steered \
    --model_args pretrained=${model_id},steer_path=${steer_config_path} \
    --tasks mmlu_prox_${lang} \
    --device cuda:0 \
    --batch_size auto \

deactivate