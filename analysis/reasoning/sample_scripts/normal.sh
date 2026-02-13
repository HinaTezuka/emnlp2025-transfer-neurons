cd analysis/reasoning

export CUDA_VISIBLE_DEVICES=0
export model_id="model_id/at/huggingface_hub"
export model_type="model_nickname"
export lang=ja

python -m lm_eval \
    --model vllm \
    --model_args pretrained=${model_id} \
    --batch_size auto \
    --tasks mmlu_prox_${lang} \
    --device cuda:0 \
    --log_samples \
    --output_path results/outputs/${model_type}_${lang}_without_any_intervention.jsonl