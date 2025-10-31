import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import (
    get_hidden_states_including_emb_layer,
    get_hidden_states_including_emb_layer_with_edit_activation,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays
)


def plot_pca(model_type: str, features_L1: dict, features_L2: dict, features_L3: dict, features_L4: dict, features_L5: dict, is_reverse: bool):
    languages = ["Japanese", "Dutch", "Korean", "Italian", "English"]
    colors = ["red", "blue", "yellow", "orange", "green"]

    if is_reverse:
        start, end = 21, 33
    else:
        start, end = 0, 21
    
    for layer_idx in range(start, end):  # Embedding layer + 32 hidden layers

        f1 = features_L1[layer_idx, :, :]
        f2 = features_L2[layer_idx, :, :]
        f3 = features_L3[layer_idx, :, :]
        f4 = features_L4[layer_idx, :, :]
        f5 = features_L5[layer_idx, :, :]

        all_features = np.concatenate([f1, f2, f3, f4, f5], axis=0)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_features)
        f1_2d = pca.transform(f1)
        f2_2d = pca.transform(f2)
        f3_2d = pca.transform(f3)
        f4_2d = pca.transform(f4)
        f5_2d = pca.transform(f5)

        # plot.
        plt.figure(figsize=(12, 12))
        for feats, color, label in zip([f1_2d, f2_2d, f3_2d, f4_2d, f5_2d], colors, languages):
            plt.scatter(feats[:, 0], feats[:, 1], color=color, label=label, alpha=0.7)
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=lang,
                markerfacecolor=col, markersize=30, alpha=0.7)
            for lang, col in zip(languages, colors)
        ]

        plt.xlabel('Principal Component I', fontsize=40)
        plt.ylabel('Principal Component II', fontsize=40)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.axis("equal")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        title = 'Emb Layer' if layer_idx == 0 else f'Layer {layer_idx}'
        file_name = 'emb_layer' if layer_idx == 0 else f'{layer_idx}'
        plt.title(title, fontsize=50)
        plt.legend(handles=legend_handles, fontsize=35)
        plt.grid(True)

        output_dir = f'figs/pca/{file_name}'

        with PdfPages(output_dir + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()

if __name__ == '__main__':
    ja = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/type2_ja.npz')
    nl = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/type2_nl.npz')
    ko = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/type2_ko.npz')
    it = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/type2_it.npz')
    en = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/type2_en.npz')

    plot_pca('llama3', ja, nl, ko, it, en, is_reverse=True)