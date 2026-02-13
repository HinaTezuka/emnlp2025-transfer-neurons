import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, matrix_rank
from matplotlib.backends.backend_pdf import PdfPages

from utils.general_utils import (
    unfreeze_np_arrays
)

langs = ['ja', 'nl', 'ko', 'it', 'en']
threshold_log = defaultdict(lambda: defaultdict(list))
layer_num = 33 # emb_layer included.

for layer_i in range(layer_num):
    all_lang_cumexp = {}
    all_lang_thresh = {}
    for lang in langs:
        hs = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/normal_{lang}.npz')
        hs_layer = hs[layer_i, :, :] # shape: (sample_num, hs_dim)

        # X = hs_layer - hs_layer.mean(axis=0, keepdims=True)  # mean-centering.
        # u, s, vh = svd(X, full_matrices=False)
        
        u, s, vh = svd(hs_layer, full_matrices=False)

        explained_variance_ratio = (s ** 2) / np.sum(s ** 2) # explained variance ratio.
        cumulative_explained_variance = np.cumsum(explained_variance_ratio) # CEVR.
        thresholds = [0.9, 0.95, 0.99]

        threshold_points = {}
        for t in thresholds:
            k = np.searchsorted(cumulative_explained_variance, t) + 1
            threshold_points[t] = k
            print(f'{lang} - Layer {layer_i}: {int(t*100)}% variance explained by top {k} components')
            threshold_log[t][lang].append(k)

        all_lang_cumexp[lang] = cumulative_explained_variance
        all_lang_thresh[lang] = threshold_points

# Summary plot
output_dir = f'' # set a base path (saving dir) properly.
os.makedirs(output_dir, exist_ok=True)

colors_by_lang = {
    'en': '#1f77b4',
    'ja': '#ff7f0e',
    'ko': '#2ca02c',
    'it': '#d62728',
    'nl': '#9467bd',
}

threshold_colors = {
    0.9: '#08c93b',
    0.95: '#54AFE4',
    0.99: '#24158A',
}

for threshold in [0.9, 0.95, 0.99]:
    plt.figure(figsize=(10, 6))

    for lang in langs:
        y = threshold_log[threshold][lang]
        plt.plot(range(len(y)), y, label=lang, color=colors_by_lang[lang], linewidth=2)

    plt.title(f'{int(threshold * 100)}% Variance', fontsize=30)
    plt.xlabel('Layers', fontsize=30)
    plt.ylabel('# Components', fontsize=30)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title='Language', fontsize=25, title_fontsize=25)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{int(threshold * 100)}")
    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()