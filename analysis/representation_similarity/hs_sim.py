"""
hidden states similarity under the following settings:
    (a): top-n Type-1 neuorns intervention.
    (b): top-n Type-2 neurons intervention.
    (c): baseline (random n. neurons from the same layers as transfer neurons) intervention.
"""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.general_utils import (
    seed_everything,
    unfreeze_pickle, 
    generate_baseline_neurons,
    take_hs_similarities_with_edit_activation,
    plot_hs_sim_hist,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    parser.add_argument('--L2_language', type=str, required=True, help='L2 Language (English-L2 sentence pair). This must be compatible with the language of transfer neruons you want to deactivate.')
    parser.add_argument('--parallel_sentences_path', type=str, required=True, default='data/example_sentences/ja_parallel_test.pkl', help='Path to parallel sentence pairs(in the paper, we sampled 1k english-L2 parallel sentence pairs from tatoeba curpus).')
    parser.add_argument('--non_parallel_sentences_path', type=str, required=True, default='data/example_sentences/ja_non_parallel.pkl', help='Path to non-parallel sentence pairs(in the paper, we sampled 1k english-L2 non-paralles sentence pairs from tatoeba curpus).')
    parser.add_argument('--intervention_type', type=str, required=True, choices=['type1', 'type2'], help='Without any intervention (normal forward path), top-n Type1 neurons intervention or top-n Type2 neurons intervention.')
    parser.add_argument('--tn_path', type=str, required=True, help='Path to a list of top-n transfer neurons ([(layer_idx, neuron_idx), (layer_idx, neuron_idx), ...]) that you want to deactivate.')
    parser.add_argument('--save_path_intervention', type=str, required=True, help='Path to save transfer neurons intervention result.')
    parser.add_argument('--save_path_baseline', type=str, required=True, help='Path to save baseline neurons intervention result.')
    args = parser.parse_args()

    """
    In the paper, for both parallel and non-parallel sentence pairs (English-L2), we sampled sentences from tatoeba curpus.

    example usage:
    python -m analysis.representation_similarity.hs_sim \
        --model_id meta-llama/Meta-Llama-3-8B \
        --L2_language ja \
        --parallel_sentences_path path/to/your/parallel_sentence_pairs_ja \
        --non_parallel_sentences_path path/to/your/random_sentence_pairs_ja \
        --intervention_type 'type1' \
        --tn_path path/to/your/top_n_type1_neurons_ja \
        --save_path_intervention figs/hs_sim/type1_deactivated_ja \
        --save_path_baseline figs/hs_sim/baseline_deactivated_ja
    """

    model_id = args.model_id
    lang = args.L2_language
    parallel_sentences_path = args.parallel_sentences_path
    non_parallel_sentences_path = args.non_parallel_sentences_path
    intervention_type = args.intervention_type
    tn_path = args.tn_path
    save_path_intervention = args.save_path_intervention
    save_path_baseline = args.save_path_baseline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    seed_everything()

    # unfreeze sentences.
    parallel_sentence_pairs = unfreeze_pickle(parallel_sentences_path)
    non_parallel_sentence_pairs = unfreeze_pickle(non_parallel_sentences_path)

    # unfreeze neurons.
    neurons = unfreeze_pickle(tn_path)
    # make baseline neurons.
    layer_range = (0, 20) if intervention_type == 'type1' else (20, 32)
    neuron_range = (0, model.config.intermediate_size)
    random_neurons = generate_baseline_neurons(neurons, layer_range, neuron_range)

    # hs similarity with top-n transfer neurons deactivated.
    print(f'\nCollecting hs with top-n transfer neurons deactivated.')
    similarities_same_semantics = take_hs_similarities_with_edit_activation(model, tokenizer, device, neurons, parallel_sentence_pairs)
    similarities_non_same_semantics = take_hs_similarities_with_edit_activation(model, tokenizer, device, neurons, non_parallel_sentence_pairs) 
    final_results_same_semantics = {}
    final_results_non_same_semantics = {}
    for layer_idx in range(model.config.num_hidden_layers):
        final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
        final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
    
    plot_hs_sim_hist(final_results_same_semantics, final_results_non_same_semantics, lang, save_path_intervention) # visualization.
    
    # hs similarity with baseline neurons deactivated.
    print(f'\nCollecting hs with randon n. neurons deactivated.')
    similarities_same_semantics = take_hs_similarities_with_edit_activation(model, tokenizer, device, random_neurons, parallel_sentence_pairs)
    similarities_non_same_semantics = take_hs_similarities_with_edit_activation(model, tokenizer, device, random_neurons, non_parallel_sentence_pairs) 
    final_results_same_semantics = {}
    final_results_non_same_semantics = {}
    for layer_idx in range(model.config.num_hidden_layers):
        final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
        final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()

    plot_hs_sim_hist(final_results_same_semantics, final_results_non_same_semantics, lang, save_path_baseline) 

    print(f'Completed. language: {lang}')