import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.general_utils import (
    seed_everything,
    save_as_pickle,
    unfreeze_pickle,
)

def aggregate_hs_for_type2(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, num_layers: int, data: List[str], L2: str) -> np.ndarray:
    hidden_states = np.zeros((num_layers, len(data), model.config.hidden_size))
    current_text_idx = None

    # hook fn for saving last token hidden states.
    def cache_layer_output(layer_idx: int, hs: np.ndarray):
        def hook(m, inp, out):
            if isinstance(out, tuple): # Tuple[torch.Tensor]
                hs[layer_idx, current_text_idx, :] = out[0][0, -1, :].detach().cpu().numpy()
            else: # torch.Tensor
                hs[layer_idx, current_text_idx, :] = out[0, -1, :].detach().cpu().numpy()

            return out
        
        return hook

    # set hooks.
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(cache_layer_output(layer_idx=layer_idx, hs=hidden_states))
        handles.append(handle)

    for text_idx, text in tqdm(enumerate(data), total=len(data), desc=f'aggregating hs for centroids computation (Type-2 neurons, {L2} sentences) ...'):
        current_text_idx = text_idx
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            _ = model(**inputs)
    
    # remove hooks.
    for handle in handles:
        handle.remove()

    return hidden_states

def get_centroid(hidden_states: np.ndarray) -> List[np.ndarray]:
    centroids = []
    for layer_idx in range(32): # 32 decoder layers.
        final_c = np.mean(hidden_states[layer_idx, :, :], axis=0)
        centroids.append(final_c)
    return centroids

"""
Revised version of centroids estimation code for detecting Type-2 neurons.

example usage:
python -m tn.calc_type2_centroids_for_tn_detection \
    --model_id meta-llama/Meta-Llama-3-8B \
    --lang_for_TN ja \
    --sentence_path path/to/your/monolingual_sentences_ja.pkl
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    parser.add_argument('--lang_for_TN', type=str, required=True, help='language you wan to detect as Transfer Neurons.')
    parser.add_argument('--sentence_path', type=str, required=True, help='sentences dataset path.')
    args = parser.parse_args()

    model_id = args.model_id
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
    hidden_states = aggregate_hs_for_type2(model, tokenizer, device, num_layers, sentences, L2)
    centroids = get_centroid(hidden_states) # list: [centroid_for_layer1(np.ndarray), centroid_for_layer2, ..., centroid_for_layer32]

    # save centroids as pkl.
    save_path = f'data/centroids/type2_re_impl/c_for_type2_{L2}.pkl'
    save_as_pickle(save_path, centroids)