import argparse
from collections import defaultdict

import numpy as np

from utils.general_utils import (
    compute_corr_ratio,
    unfreeze_pickle,
    unfreeze_np_arrays,
)

# language labels for Japanese/Dutch/Korean/Italian/English (This label must be compatible with sentence data at analysis/get_activations_for_corr_analysis.py).
l1 = [ 1 for _ in range(1000)]
l2 = [ 0 for _ in range(1000)]
labels_dict = {
    'ja': l1 + l2 + l2 + l2 + l2,
    'nl': l2 + l1 + l2 + l2 + l2,
    'ko': l2 + l2 + l1 + l2 + l2,
    'it': l2 + l2 + l2 + l1 + l2,
    'en': l2 + l2 + l2 + l2 + l1,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_arr_path', type=str, required=True, help='Path to a .npz file containing a NumPy array of shape (num_layers, num_activations, num_sentence_samples) that stores MLP activations for the desired model.')
    parser.add_argument('--langs', type=str, nargs='+', required=True, default='ja nl ko it', help='List of language codes of transfer neurons.') # in the paper, 'ja nl ko it'.
    parser.add_argument('--top_n', type=int, default=1000, choices=[10, 100, 1000], help='Top-n transfer neurons from the rankings.')
    args = parser.parse_args()
    """
    example usage:
    python -m analysis.corr_analysis \
        --activations_arr_path data/inner_reps/activations_for_corr_analysis.npz \
        --langs ja nl ko it \
        --top-n 1000
    """

    activations_arr_path = args.activations_arr_path
    langs = args.langs
    top_n = args.top_n

    # unfreeze activations.
    activations_arr = unfreeze_np_arrays(activations_arr_path)
    print(f'Unfreezed activations array of shape: {activations_arr.shape}')

    for lang in langs:
        labels_list = np.asarray(labels_dict[lang])

        for tn_type in ['type1', 'type2']:
            neurons_path = f'data/tn/{lang}_{tn_type}_top1000.pkl'
            neurons = unfreeze_pickle(neurons_path)

            corr_ratios = []
            for layer_idx, neuron_idx in neurons[:top_n]:
                corr_ratio_score = compute_corr_ratio(labels_list, activations_arr[layer_idx, neuron_idx, :].astype(np.float64))
                corr_ratios.append(corr_ratio_score)
            mean_corr_ratio_top_n = np.mean(np.asarray(corr_ratios))

            # print results.
            print(f'Correlation Ratio: {tn_type}, {lang}')
            print(mean_corr_ratio_top_n)