import argparse
from collections import defaultdict
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.general_utils import (
    seed_everything,
    save_as_pickle,
    unfreeze_pickle,
)

def aggregate_hs_for_type1(model, tokenizer, device, num_layers, data):
    c_hidden_states = defaultdict(list)
    for text1, text2 in data:
        inputs1 = tokenizer(text1, return_tensors="pt").to(device) # english text
        inputs2 = tokenizer(text2, return_tensors="pt").to(device) # L2 text

        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)
            output2 = model(**inputs2, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states[1:] # remove embedding layer
        all_hidden_states2 = output2.hidden_states[1:]
        last_token_index1 = inputs1["input_ids"].shape[1] - 1
        last_token_index2 = inputs2["input_ids"].shape[1] - 1

        for layer_idx in range(num_layers):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            hs2 = all_hidden_states2[layer_idx][:, last_token_index2, :].squeeze().detach().cpu().numpy()
            # save mean of (en_ht, L2_ht). <- estimated shared point in shared semantic space.
            c = np.stack([hs1, hs2])
            c = np.mean(c, axis=0)
            c_hidden_states[layer_idx].append(c)

    return dict(c_hidden_states)

def aggregate_hs_for_type2(model, tokenizer, device, num_layers, data):
    hidden_states = defaultdict(list)

    torch.manual_seed(42)
    for text1 in data:
        inputs = tokenizer(text1, return_tensors="pt").to(device) # english text

        # get hidden_states
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)

        all_hidden_states = output.hidden_states[1:] # exclude embedding layer
        last_token_index = inputs["input_ids"].shape[1] - 1
        for layer_idx in range(num_layers):
            hs = all_hidden_states[layer_idx][:, last_token_index, :].squeeze().detach().cpu().numpy()
            hidden_states[layer_idx].append(hs)

    return dict(hidden_states)

def get_centroid(hidden_states: dict):
    centroids = [] # [c1, c2, ] len = layer_num
    for layer_idx, c in hidden_states.items():
        final_c = np.mean(c, axis=0) # calc mean of c(shared point per text) of all text.
        centroids.append(final_c)
    return centroids

"""
SENTENCE DATASET REQUIREMENTS.
L1: fixed to English (We assume English latent space serves as a shared semantic latent space in middle layers).
L2: language you want to detect as Transfer Neurons (in the paper, we detected TNs for ja/nl/ko/it).
for Type-1 neurons detection: sentences must be a list of parallels pairs(tuple) of english-L2: [(L1_sentence1, parallel_sentence_in_L2), (L1_sentence2, parallel_sentence_in_L2), ...]
for Type-2 neurons detection: sentence must be a list of L2 sentences: [L2_sentence1, L2_sentence2, L2_sentence3, ...]

You can use any sentence datasets as long as the datasets meet the conditions above.
"""

"""
hwo to use (e.g.,):
python calc_centroid.py \
    meta-llama/Meta-Llama-3-8B \
    type1 \
    ja \
    data/sentences/parallel_en-ja.pkl
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='model_path at HuggingFace Hub.')
    parser.add_argument('TN_Type', help='Type of TN you want to detect.')
    parser.add_argument('lang_for_TN', help='language you wan to detect as Transfer Neurons.')
    parser.add_argument('sentence_path', halp='sentences dataset path,')
    args = parser.parse_args()

    model_id = args.model_id
    tn_type = args.TN_Type
    L2 = args.lang_for_TN
    sentence_path = args.sentence_path

    # model params.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_layers = model.config.num_hidden_layers    

    seed_everything()

    # compute centroids.
    sentences = unfreeze_pickle(sentence_path)
    hidden_states = aggregate_hs_for_type1(model, tokenizer, device, num_layers, sentences) if tn_type == 'type1' else aggregate_hs_for_type2(model, tokenizer, device, num_layers, sentences)
    centroids = get_centroid(hidden_states)

    # save centroids as pkl.
    save_path = f"data/centroids/c_for_type1_{L2}.pkl"
    save_as_pickle(save_path, centroids)