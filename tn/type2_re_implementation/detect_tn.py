import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.general_utils import (
    seed_everything,
    compute_scores_for_tn_detection,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

"""
re-impl version of Type-2 neurons detection code.
Estimate centroids by executing 'type2_re_implementation/calc_type2_centroids_for_tn_detection.py' and use 'data/centroids/type2_re_impl/c_for_{tn_type}_{L2}.pkl' as centroids for the detection code in this file.

example usage:
python -m tn.type2_re_implementation.detect_tn \
    --model_id mistralai/Mistral-7B-v0.3 \
    --top_n 1000 \
    --lang_for_TN ja \
    --centroids_path data/centroids/type2_re_impl/c_for_type2_ja.pkl \
    --sentence_path path/to/your/monolingual_sentences_ja.pkl
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    parser.add_argument('--top_n', type=int, required=True, default=1000, help='top-n neurons from the score ranking.')
    parser.add_argument('--lang_for_TN', type=str, required=True, help='language you wan to detect as Transfer Neurons.')
    parser.add_argument('--centroids_path', type=str, required=True, help='path for the list of centroids (.pkl).')
    parser.add_argument('--sentence_path', type=str, required=True, help='sentences dataset path.')
    args = parser.parse_args()

    model_id = args.model_id
    top_n = args.top_n
    L2 = args.lang_for_TN
    centroids_path = args.centroids_path
    sentence_path = args.sentence_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model params
    num_layers = model.config.num_hidden_layers # 32.
    num_neurons = model.config.intermediate_size

    candidate_layers_range = num_layers # 32 decoder layers.
    candidates = {}
    for layer_idx in range(candidate_layers_range):
        for neuron_idx in range(num_neurons):
            candidates.setdefault(layer_idx, []).append(neuron_idx)
    
    """ Type-2 transfer neurons detection. """
    seed_everything()

    # unfreeze centroids.
    centroids = unfreeze_pickle(centroids_path)
    # monolingual(L2) sentences for detection.
    sentences = unfreeze_pickle(sentence_path) # must be: [sentence1, sentence2, ...].

    # run detection.
    scores = compute_scores_for_tn_detection(model, tokenizer, device, sentences, candidates, centroids)
    neuron_ranking, score_dict = sort_neurons_by_score(scores)
    neuron_ranking = [neuron for neuron in neuron_ranking if neuron[0] in [ _ for _ in range(20, 32)]] # layer filtering for Type-2 neurons (layer range: 21-32 decoder layers).
    top_transfer_neurons = neuron_ranking[:top_n]

    # save as pkl.
    sorted_neurons_path = f'data/tn/type2_re_impl/{L2}_type2_top{top_n}.pkl'
    save_as_pickle(sorted_neurons_path, top_transfer_neurons)
    score_dict_path = f'data/tn/type2_re_impl/{L2}_type2_top{top_n}_score_dict.pkl'
    save_as_pickle(score_dict_path, score_dict)