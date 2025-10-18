import argparse
from collections import defaultdict
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from ..utils.general_utils import (
    seed_everything,
    save_as_pickle,
    unfreeze_pickle,
)

def aggregate_hs_for_type1(model, tokenizer, device, num_layers, data, L2):
    c_hidden_states = defaultdict(list)
    for text1, text2 in tqdm(data, total=len(data), desc=f'aggregating hs for centroids computation (Type-1 neurons, eng-{L2} sentence pairs) ...'):
        inputs1 = tokenizer(text1, return_tensors='pt').to(device) # english text.
        inputs2 = tokenizer(text2, return_tensors='pt').to(device) # L2 text.

        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)
            output2 = model(**inputs2, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states[1:] # exclude embedding layer.
        all_hidden_states2 = output2.hidden_states[1:]
        last_token_index1 = inputs1['input_ids'].shape[1] - 1
        last_token_index2 = inputs2['input_ids'].shape[1] - 1

        for layer_idx in range(num_layers):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            hs2 = all_hidden_states2[layer_idx][:, last_token_index2, :].squeeze().detach().cpu().numpy()
            c = np.stack([hs1, hs2])
            c = np.mean(c, axis=0)
            c_hidden_states[layer_idx].append(c)

    return dict(c_hidden_states)

def aggregate_hs_for_type2(model, tokenizer, device, num_layers, data, L2):
    hidden_states = defaultdict(list)
    for text in tqdm(data, total=len(data), desc=f'aggregating hs for centroids computation (Type-2 neurons, {L2} sentences) ...'):
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)

        all_hidden_states = output.hidden_states[1:] # exclude embedding layer
        last_token_index = inputs['input_ids'].shape[1] - 1
        for layer_idx in range(num_layers):
            hs = all_hidden_states[layer_idx][:, last_token_index, :].squeeze().detach().cpu().numpy()
            hidden_states[layer_idx].append(hs)

    return dict(hidden_states)

def get_centroid(hidden_states: dict):
    centroids = []
    for layer_idx, c in hidden_states.items():
        final_c = np.mean(c, axis=0)
        centroids.append(final_c)
    return centroids

"""
SENTENCE DATASET REQUIREMENTS.
L1: fixed to English (We assume English latent space serves as a shared semantic latent space in middle layers).
L2: language you want to detect as Transfer Neurons (in the paper, we detected TNs for ja/nl/ko/it).
For Type-1 neurons detection: sentences must be a list of parallels pairs(tuple) of english-L2: [(L1_sentence1, parallel_sentence_in_L2), (L1_sentence2, parallel_sentence_in_L2), ...]
For Type-2 neurons detection: sentence must be a list of L2 sentences: [L2_sentence1, L2_sentence2, L2_sentence3, ...]

You can use any sentence datasets as long as the datasets meet the conditions above.

example usage:
python calc_centroid.py \
    meta-llama/Meta-Llama-3-8B \
    type1 \
    ja \
    data/sentences/parallel_en-ja.pkl
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help='model_path at HuggingFace Hub.')
    parser.add_argument('TN_Type', type=str, help='Type of TN you want to detect.')
    parser.add_argument('lang_for_TN', type=str, help='language you wan to detect as Transfer Neurons.')
    parser.add_argument('sentence_path', type=str, halp='sentences dataset path.')
    args = parser.parse_args()

    model_id = args.model_id
    tn_type = args.TN_Type
    L2 = args.lang_for_TN
    sentence_path = args.sentence_path

    # model params.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_layers = model.config.num_hidden_layers    

    seed_everything()

    # compute centroids.
    sentences = unfreeze_pickle(sentence_path)
    hidden_states = aggregate_hs_for_type1(model, tokenizer, device, num_layers, sentences, L2) if tn_type == 'type1' else aggregate_hs_for_type2(model, tokenizer, device, num_layers, sentences, L2)
    centroids = get_centroid(hidden_states) # list: [centroid_for_layer1(np.ndarray), centroid_for_layer2, ...]

    # save centroids as pkl.
    save_path = f'data/centroids/c_for_{tn_type}_{L2}.pkl'
    save_as_pickle(save_path, centroids)