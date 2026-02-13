cd analysis/reasoning

export CUDA_VISIBLE_DEVICES=0 # single GPU.
export model_id="model_id/at/huggingface_hub"
export model_type="model_nickname"
export lang=ja

# top-n Type-1 neurons deactivate.
export steer_config_path=steer_configs/${lang}.pt # make this .pt file with "analysis/reasoning/make_steer_config.py".

python -m lm_eval \
    --model steered \
    --model_args pretrained=${model_id},steer_path=${steer_config_path} \
    --tasks mmlu_prox_${lang} \
    --device cuda:0 \
    --batch_size auto \

deactivate