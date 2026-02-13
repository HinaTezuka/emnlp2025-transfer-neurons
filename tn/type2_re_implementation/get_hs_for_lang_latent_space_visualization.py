import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import (
    # get_hidden_states_including_emb_layer,
    # get_hidden_states_including_emb_layer_with_edit_activation,
    unfreeze_pickle,
    save_np_arrays,
)

from .utils import (
    cache_last_token_hs,
    cache_last_token_hs_with_edit_activation,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    parser.add_argument('--intervention_type', type=str, required=True, choices=['w/o', 'type1', 'type2'], help='Without any intervention (normal forward path), top-n Type1 neurons intervention or top-n Type2 neurons intervention.')
    parser.add_argument('--languages', type=str, nargs='+', required=True, default='ja nl ko it en', help='Language variation of latent space (this must be compatible with sentence data).')
    parser.add_argument('--sentence_data_path', type=str, default='data/example_sentences/mkqa_example_sentences.pkl', help='multilingual sentence path (In the paper, we used MKQA parallel question sentences for ja/nl/ko/it/en (1k sentences per language).)')
    args = parser.parse_args()

    """
    In the paper, we used parallel question sentences in MKQA dataset for language latent spaces visualization (1k sentence per language).

    example usage:
    python -m type2_re_implementation.get_hs_for_lang_latent_space_visualization \
        --model_id meta-llama/Meta-Llama-3-8B \
        --intervention_type type2 \
        --languages ja nl ko it en \

    python -m type2_re_implementation.get_hs_for_lang_latent_space_visualization \
        --model_id mistralai/Mistral-7B-v0.3 \
        --intervention_type type2 \
        --languages ja nl ko it en \
    """

    model_id = args.model_id
    intervention_type = args.intervention_type
    sentence_path = args.sentence_data_path
    langs = args.languages

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # unfreeze sentences.
    sentences = unfreeze_pickle(sentence_path)

    # run.
    for lang in tqdm(langs, total=len(langs), desc=f'{str(len(langs)).upper()} languages, intervention type: {intervention_type}'):
        if lang == 'en' or intervention_type == 'w/o': # w/o intervention for English sentences.
            hs_array = cache_last_token_hs(model, tokenizer, device, sentences[lang], is_emb_layer_included=True)
        elif intervention_type in ['type1', 'type2']:
            tn_path = f'data/tn/type2_re_impl/{lang}_{intervention_type}_top100000.pkl' # example path to the list of top-n transfer neurons.
            neurons = unfreeze_pickle(tn_path)[:1000] # ex: top-1k neurons.
            hs_array = cache_last_token_hs_with_edit_activation(model, tokenizer, device, sentences[lang], is_emb_layer_included=True, neurons=neurons)
        else:
            raise ValueError(f'Unknown intervention_type: {intervention_type}')
        
        # save.
        path = f'data/inner_reps/hs_for_lang_latent/type2_re_impl/{"normal" if intervention_type == "w/o" else intervention_type}_{lang}.npz'
        save_np_arrays(path, hs_array)

        print(f'{lang} completed.')