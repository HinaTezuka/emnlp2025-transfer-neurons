import argparse
import os
import pickle
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.general_utils import (
    seed_everything,
    track_activations_with_text_data,
    unfreeze_pickle,
    save_np_arrays,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    # In the paper, as a sentence dataset, we sampled 1k sentences from tatoeba corpus (https://huggingface.co/datasets/Helsinki-NLP/tatoeba) for each language.
    parser.add_argument('--ja_sentence_path', type=str, default='data/example_sentences/ja_mono_test.pkl', help='sentence data for Japanese.')
    parser.add_argument('--nl_sentence_path', type=str, default='data/example_sentences/nl_mono_test.pkl', help='sentence data for Dutch.')
    parser.add_argument('--ko_sentence_path', type=str, default='data/example_sentences/ko_mono_test.pkl', help='sentence data for Korean.')
    parser.add_argument('--it_sentence_path', type=str, default='data/example_sentences/it_mono_test.pkl', help='sentence data for Italian.')
    parser.add_argument('--en_sentence_path', type=str, default='data/example_sentences/en_mono_test.pkl', help='sentence data for English.')
    args = parser.parse_args()

    """
    example usage:
    python -m analysis.get_activations_for_corr_analysis \
        --model_id meta-llama/Meta-Llama-3-8B
    """

    model_id = args.model_id
    ja = unfreeze_pickle(args.ja_sentence_path)
    nl = unfreeze_pickle(args.nl_sentence_path)
    ko = unfreeze_pickle(args.ko_sentence_path)
    it = unfreeze_pickle(args.it_sentence_path)
    en = unfreeze_pickle(args.en_sentence_path)
    sentences = ja + nl + ko + it + en
    print(f'Got {len(sentences)} sentences in total.') # 5000.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # run.
    seed_everything()
    activations = track_activations_with_text_data(model, tokenizer, device, sentences)

    # save.
    save_path = f'data/inner_reps/activations_for_corr_analysis.npz'
    save_np_arrays(save_path, activations)
    print(f'Completed.')