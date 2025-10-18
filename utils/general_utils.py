from collections import defaultdict
import os
import pickle
import random
import sys

from baukit import TraceDict # baukit: https://github.com/davidbau/baukit
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
from tqdm import tqdm


def seed_everything(seed: int=42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_inner_reps(model, prompt):
    num_layers = model.config.num_hidden_layers
    MLP_act = [f'model.layers.{i}.mlp.act_fn' for i in range(num_layers)]
    MLP_up_proj = [f'model.layers.{i}.mlp.up_proj' for i in range(num_layers)]
    ATT_act = [f'model.layers.{i}.self_attn.o_proj' for i in range(num_layers)]

    with TraceDict(model, MLP_act + MLP_up_proj + ATT_act) as ret:
        with torch.no_grad():
            outputs = model(prompt, output_hidden_states=True)
    
    MLP_act_values = [ret[act].output for act in MLP_act]
    up_proj_values = [ret[proj].output for proj in MLP_up_proj]
    ATT_values = [ret[att].output for att in ATT_act]
    
    return MLP_act_values, up_proj_values, ATT_values, outputs

def compute_scores_for_tn_detection(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    num_layers = model.config.num_hidden_layers
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = model.config.intermediate_size
    scores_all_txt = np.zeros((num_candidate_layers, num_neurons, len(data))) # temp save array across all the input samples.
    final_scores = np.zeros((num_candidate_layers, num_neurons)) # save array for final score of each candidate neuron.

    for text_idx, text in tqdm(enumerate(data), total=len(data), desc=f'Computing scores for all the candidate neurons for {len(data)} samples ...'):
        inputs = tokenizer(text, return_tensors='pt').to(device)

        # extract hidden activations.
        MLP_act_values, up_proj_values, post_attention_values, outputs = get_inner_reps(model, inputs.input_ids)
        hs_all_layer = outputs.hidden_states # including 0th layer(emb layer): emb layer is needed to calc scores for neurons in 1st-decoder layer.
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        for layer_idx, neurons in candidate_neurons.items():
            c = centroids[layer_idx].reshape(1, -1) # centroid of the i-th layer.
            # (i-1)-th layer hidden_state.
            hs_pre = hs_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer attention output.
            atts = post_attention_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer activation values (act_fn(x) * up_proj(x))
            act_fn = MLP_act_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            acts = act_fn * up_proj

            # layer score at i-th layer.
            layer_score = (hs_pre + atts).reshape(1, -1) # H^l-1 + Att^l.
            if score_type == 'L2_dis':
                layer_score = euclidean_distances(layer_score, c)[0, 0]
            elif score_type == 'cos_sim':
                layer_score = cosine_similarity(layer_score, c)[0, 0]

            neuron_indices = np.array(candidate_neurons[layer_idx])
            value_vectors = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_indices].detach().cpu().numpy()
            # neuron scores at i-th layer.
            hs_with_neurons = (hs_pre + atts + (acts.reshape(-1, 1) * value_vectors)) # H^l-1 + Att^l + av (a: activation value, v: correnponding value vector). 
            if score_type == 'L2_dis':
                neuron_scores = euclidean_distances(hs_with_neurons, c).reshape(-1)
                neuron_scores = np.where(neuron_scores <= layer_score, abs(layer_score - neuron_scores), -abs(layer_score - neuron_scores))
            elif score_type == 'cos_sim':
                neuron_scores = cosine_similarity(hs_with_neurons, c).reshape(-1)
                neuron_scores = np.where(neuron_scores >= layer_score, abs(layer_score - neuron_scores), -abs(layer_score - neuron_scores))
            scores_all_txt[layer_idx, :, text_idx] = neuron_scores

    # final scores (mean scores for all the input samples).
    final_scores[:, :] = np.mean(scores_all_txt, axis=2)

    return final_scores

def sort_neurons_by_score(final_scores):
    # {(layer_idx, neuron_idx): score, (layer_idx, neuron_idx): score, ...}
    score_dict = {
        (layer_idx, neuron_idx): final_scores[layer_idx, neuron_idx]
        for layer_idx in range(final_scores.shape[0])
        for neuron_idx in range(final_scores.shape[1])
    }
    # sort in descending order: [(layer_idx, neuron_idx), (layer_idx, neuron_idx), ...]
    sorted_neurons = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)

    return sorted_neurons, score_dict

def save_as_pickle(file_path: str, target_dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + '.tmp'

    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(target_dict, f)
        os.replace(temp_path, file_path)
        print('Pickle file successfully saved.')
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def unfreeze_pickle(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Pickle file not found: {file_path}')

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f'Error unpickling file {file_path}: {e}')

def save_np_arrays(save_path, np_array):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        np.savez(save_path, data=np_array)
        print(f'Array successfully saved to {save_path}')
    except Exception as e:
        print(f'Failed to save array: {e}')

def unfreeze_np_arrays(save_path):
    try:
        with np.load(save_path) as data:
            return data['data']
    except Exception as e:
        print(f'Failed to load array: {e}')
        return None