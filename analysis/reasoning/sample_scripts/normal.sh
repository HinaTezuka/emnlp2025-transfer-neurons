cd analysis/reasoning
source venv_for_eval_harness/bin/activate
module load cuda/12.1

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

deactivate