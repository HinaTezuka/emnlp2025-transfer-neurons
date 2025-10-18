import argparse
import pickle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.general_utils import (
    seed_everything,
    compute_scores_for_tn_detection,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help='model_path at HuggingFace Hub.')
    parser.add_argument('TN_Type', type=str, help='Type of TN you want to detect.')
    parser.add_argument('top_n', type=int, default=1000, help='top-n neurons from the score ranking.')
    parser.add_argument('lang_for_TN', type=str, help='language you wan to detect as Transfer Neurons.')
    parser.add_argument('scoring_type', type=str, help='scoring metric: either "cos_sim"(cosine similarity) or "L2_dis"(euclidean distance) can be accepted.')
    parser.add_argument('centroids_path', type=str, help='path for the list of centroids (.pkl).')
    parser.add_argument('sentence_path', type=str, halp='sentences dataset path.')
    args = parser.parse_args()

    model_id = args.model_id
    tn_type = args.TN_Type
    top_n = args.top_n
    L2 = args.lang_for_TN
    score_type = args.scoring_type
    centroids_path = args.centroids_path
    sentence_path = args.sentence_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model params
    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size

    candidate_layers_range = 20 if tn_type == 'type1' else 32
    candidates = {}
    for layer_idx in range(candidate_layers_range):
        for neuron_idx in range(num_neurons):
            candidates.setdefault(layer_idx, []).append(neuron_idx)
    
    """ transfer neurons detection. """
    seed_everything()

    # unfreeze centroids.
    centroids = unfreeze_pickle(centroids_path)
    # monolingual(L2) sentences for detection.
    sentences = unfreeze_pickle(sentence_path)

    # run detection.
    scores = compute_scores_for_tn_detection(model, tokenizer, device, sentences, candidates, centroids, score_type)
    sorted_neurons, score_dict = sort_neurons_by_score(scores)
    if tn_type == 'type2':
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
    sorted_neurons = sorted_neurons[:top_n]

    # save as pkl.
    sorted_neurons_path = f"data/tn/{L2}_{tn_type}.pkl"
    score_dict_path = f"data/tn/{L2}_{tn_type}_score_dict.pkl"
    save_as_pickle(sorted_neurons_path, sorted_neurons)
    save_as_pickle(score_dict_path, score_dict)