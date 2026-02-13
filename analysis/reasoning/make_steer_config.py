import os
import pickle
import random
from typing import List, Tuple

import torch

def unfreeze_pickle(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Pickle file not found: {file_path}')

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f'Error unpickling file {file_path}: {e}')

def generate_baseline_neurons(transfer_neurons: List[Tuple[int, int]], layer_range: Tuple[int, int], neuron_range: Tuple[int, int]) -> List[Tuple[int, int]]:
    a, b = layer_range
    c, d = neuron_range
    n = len(transfer_neurons)
    
    all_candidates = [(l_idx, n_idx) for l_idx in range(a, b) for n_idx in range(c, d)]

    available = list(set(all_candidates) - set(transfer_neurons))
    
    # select random n.
    baseline = random.sample(available, n)
    return baseline

def make_steer_config(neuron_list, hidden_size, action="multiply", clamp_value=0.0): # we do not use clamp_value.
    config = {}
    for layer, neuron in neuron_list:
        hook_name = f"layers.{layer}.mlp.act_fn"
        
        # If this layer is not yet in config, initialize a new entry
        if hook_name not in config:
            steering_vector = torch.ones(1, hidden_size)
            config[hook_name] = {
                "steering_vector": steering_vector,
                "steering_coefficient": clamp_value,
                "action": action,
                "bias": None
            }
        
        # Set the corresponding neuron position to 0.0
        config[hook_name]["steering_vector"][0, neuron] = 0.0

    return config

if __name__ == "__main__":
    """
    usage: python analysis/reasoning/make_steer_config.py
    """

    langs = ['ja', 'ko', 'it']
    # langs = ['ja', 'ko', 'fr']
    intervention_num = 1000
    action_type = 'multiply' # we only consider "multiply" steering.
    
    # あとで消す.
    model_type_for_test = 'aya'
    
    neuron_num = 14336
    for L2 in langs:
        # get type-1 neurons.
        type1_path = f'' # your top-n Type-1 neurons path
        type1_neurons = unfreeze_pickle(type1_path)

        # baseline.
        random.seed(42)
        random_neurons = generate_baseline_neurons(transfer_neurons=type1_neurons, layer_range=(0, 20), neuron_range=(0, neuron_num))

        # steer_config for top-n Type-1 neurons.
        config = make_steer_config(
            type1_neurons,
            hidden_size=neuron_num,
            action=action_type,
            clamp_value=0.0
        )
        # steer_config for random n. baseline neurons.
        config_baseline = make_steer_config(
            random_neurons, # baseline
            hidden_size=neuron_num,
            action=action_type,
            clamp_value=0.0
        )

        # save (make analysis/reasoning/steer_configs directory beforehand).
        output_path = f'analysis/reasoning/steer_configs/{L2}.pt'
        output_path_baseline = f'analysis/reasoning/steer_configs/{L2}_baseline.pt'
        torch.save(config, output_path)
        print(f'Saved steer_config to: {output_path}')
        torch.save(config_baseline, output_path_baseline)
        print(f'Saved steer_config to: {output_path}')