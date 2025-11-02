from collections import defaultdict
import os
import pickle
import random
from typing import Dict, List, Tuple

from baukit import TraceDict # baukit: https://github.com/davidbau/baukit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    
    act_fn_values = [ret[act].output for act in MLP_act]
    up_proj_values = [ret[proj].output for proj in MLP_up_proj]
    ATT_values = [ret[att].output for att in ATT_act]
    
    return act_fn_values, up_proj_values, ATT_values, outputs

def compute_scores_for_tn_detection(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = model.config.intermediate_size
    scores_all_txt = np.zeros((num_candidate_layers, num_neurons, len(data))) # temp save array across all the input samples.
    final_scores = np.zeros((num_candidate_layers, num_neurons)) # save array for final score of each candidate neuron.

    for text_idx, text in tqdm(enumerate(data), total=len(data), desc=f'Computing scores of all the candidate neurons for {len(data)} samples ...'):
        inputs = tokenizer(text, return_tensors='pt').to(device)

        # extract hidden activations.
        act_fn_values, up_proj_values, post_attention_values, outputs = get_inner_reps(model, inputs.input_ids)
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
            act_fn = act_fn_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # MLP Computation: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            acts = act_fn * up_proj # you can get the same values using "register_forward_pre_hook" to the mlp.down_proj module.

            # layer score at i-th layer.
            hs_before_mlp = (hs_pre + atts).reshape(1, -1) # H^l-1 + Att^l.
            if score_type == 'L2_dis':
                layer_score = euclidean_distances(hs_before_mlp, c)[0, 0]
            elif score_type == 'cos_sim':
                layer_score = cosine_similarity(hs_before_mlp, c)[0, 0]

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

def track_activations_with_text_data(model, tokenizer, device, data):
    activations_array = np.zeros((model.config.num_hidden_layers, model.config.intermediate_size, len(data)), dtype=np.float16)
    current_text_idx = None

    def activation_extractor(model, input, layer_idx: int):
        # extract last-token activations.
        activation_values = input[0][0, -1, :].detach().cpu().numpy()
        activations_array[layer_idx, :, current_text_idx] = activation_values

    # set hooks.
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handle = layer.mlp.down_proj.register_forward_pre_hook(
            lambda model, input, layer_idx=layer_idx: activation_extractor(model, input, layer_idx)
        )
        handles.append(handle)

    for text_idx, text in tqdm(enumerate(data), total=len(data), desc='Collecting activations...'):
        current_text_idx = text_idx
        inputs = tokenizer(text, return_tensors='pt').to(device)
        # run inference.
        with torch.no_grad():
            _ = model(**inputs)

    # remove handles.
    for handle in handles:
        handle.remove()

    return activations_array

def compute_corr_ratio(categories: np.ndarray, values: np.ndarray):
    # this function were originally cited from: https://www.sambaiz.net/en/article/441/.
    interclass_variation  = sum([
        (len(values[categories == i]) * ((values[categories == i].mean() - values.mean()) ** 2)).sum() for i in np.unique(categories)
    ]) 
    total_variation = sum((values - values.mean()) ** 2)

    return interclass_variation / total_variation

def edit_activation(output, layer, layer_idx_and_neuron_idx):
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if f"model.layers.{layer_idx}." in layer:
            output[:, -1, neuron_idx] *= 0

    return output

def get_hidden_states_including_emb_layer_with_edit_activation(model, tokenizer, device, data, neurons: List[Tuple[int, int]], lang: str) -> np.ndarray:
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in neurons]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, neurons)) as tr:

        return get_hidden_states_including_emb_layer(model, tokenizer, device, data, lang)

def get_hidden_states_including_emb_layer(model, tokenizer, device, data, lang) -> np.ndarray:
    hidden_states = np.zeros((model.config.num_hidden_layers+1, len(data), model.config.hidden_size)) # model.config.num_hidden_layers+1: include emb layer.

    for text_idx, text in tqdm(enumerate(data), total=len(data), desc=f'Collecting hs for {lang}...'):
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states # include embedding layer.
        for layer_idx in range(len(all_hidden_states)):
            hidden_states[layer_idx, text_idx, :] = all_hidden_states[layer_idx][:, -1, :].squeeze().detach().cpu().numpy() # last token hs.

    return hidden_states

def take_hs_similarities_with_edit_activation(model, tokenizer, device: str, neurons: List[Tuple[int, int]], data: List[str]) -> Dict[int, List[float]]:
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in neurons]))
    similarities = defaultdict(list)

    for L1_txt, L2_txt in tqdm(data, total=len(data), desc=f'Taking hs similarities for sentence pairs ...'):
        # get L2 hs (with intervention).
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, neurons)):
            inputs_L2 = tokenizer(L2_txt, return_tensors='pt').to(device)
            with torch.no_grad():
                output_L2 = model(**inputs_L2, output_hidden_states=True)
            last_token_index_L2 = inputs_L2['input_ids'].shape[1] - 1
            last_token_hidden_states_L2 = [
                layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
                for layer_hidden_state in output_L2.hidden_states[1:]
            ]

        # get L1 hs (without intervention).
        inputs_L1 = tokenizer(L1_txt, return_tensors='pt').to(device)
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
        last_token_index_L1 = inputs_L1['input_ids'].shape[1] - 1
        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
            for layer_hidden_state in output_L1.hidden_states[1:]
        ]

        # cos_sim for the current sentence pair.
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: List[np.ndarray], last_token_hidden_states_L2: List[np.ndarray], similarities: Dict[int, List[float]]) -> Dict[int, List[float]]:
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0]
        similarities[layer_idx].append(sim)

    return similarities

def plot_hs_sim_hist(dict_same_semantics: Dict[int, float], dict_diff_semantics: Dict[int, float], lang: str, save_path: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict_same_semantics.keys()))
    values1 = list(dict_same_semantics.values())
    values2 = list(dict_diff_semantics.values())

    offset = 0.1

    # plot hist
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(8, 7))
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.ylim(-0.5, 1)
    plt.title(f'en-{lang}', fontsize=35)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(fontsize=25)
    plt.grid(True)
    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

def generate_baseline_neurons(transfer_neurons: List[Tuple[int, int]], layer_range: Tuple[int, int], neuron_range: Tuple[int, int]) -> List[Tuple[int, int]]:
    a, b = layer_range
    c, d = neuron_range
    n = len(transfer_neurons)
    
    all_candidates = [(l_idx, n_idx) for l_idx in range(a, b) for n_idx in range(c, d)]

    available = list(set(all_candidates) - set(transfer_neurons))
    
    # select random n.
    baseline = random.sample(available, n)
    return baseline
    
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