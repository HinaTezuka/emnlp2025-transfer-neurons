import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.linalg import svd, matrix_rank
from matplotlib.backends.backend_pdf import PdfPages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler

from funcs import (
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
langs = ["ja", "nl", "ko", "it", "en"]
# models
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b', 'bigscience/bloom-3b']
threshold_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
is_scaled = False
intervention_type = 'normal' # normal, type-1(top-1k), type-2(top-1k). 

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'bloom'
    layer_num = 33 if model_type in ['llama3', 'mistral', 'aya'] else 31 # emb_layer included.

    for layer_i in range(layer_num):
        all_lang_cumexp = {}
        all_lang_thresh = {}

        for L2 in langs:
            if intervention_type == 'normal':
                hs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}.pkl")
            elif intervention_type == 'type-1':
                hs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}_type1.pkl")
            elif intervention_type == 'type-2':
                hs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/{L2}.pkl")
            hs_layer = np.array(hs[layer_i]) # shape: (sample_num, hs_dim)
            if is_scaled or model_type == 'phi4':
                scaler = StandardScaler()
                hs_layer = scaler.fit_transform(hs_layer)
                if model_type == 'phi4':
                    hs_layer = hs_layer.astype(np.float32)
            u, s, vh = svd(hs_layer, full_matrices=False)

            explained_variance_ratio = (s ** 2) / np.sum(s ** 2) # 寄与率.
            cumulative_explained_variance = np.cumsum(explained_variance_ratio) # 累積寄与率.
            thresholds = [0.9, 0.95, 0.99]

            threshold_points = {}
            for t in thresholds:
                k = np.searchsorted(cumulative_explained_variance, t) + 1
                threshold_points[t] = k
                print(f"{L2} - Layer {layer_i}: {int(t*100)}% variance explained by top {k} components")
                threshold_log[model_type][t][L2].append(k)

            all_lang_cumexp[L2] = cumulative_explained_variance
            all_lang_thresh[L2] = threshold_points

    # Summary plot
    if is_scaled:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary/scale"
    else:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary"
        # output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary/all_langs"
    os.makedirs(output_dir, exist_ok=True)

    colors_by_lang = {
        "en": "#1f77b4",
        "ja": "#ff7f0e",
        "ko": "#2ca02c",
        "it": "#d62728",
        "nl": "#9467bd",
        "vi": "#8c564b",
        "ru": "#e377c2",
        "fr": "#7f7f7f",
    }

    threshold_colors = {
        0.9: "#08c93b",
        0.95: "#54AFE4",
        0.99: "#24158A",
    }

    model_name_map = {
        "llama3": "LLaMA3-8B",
        "mistral": "Mistral-7B",
        "aya": "Aya expanse-8B",
        "bloom": "BLOOM-3B",
    }

    for model_type in threshold_log:
        for threshold in [0.9, 0.95, 0.99]:
            plt.figure(figsize=(10, 6))

            for lang in langs:
                y = threshold_log[model_type][threshold][lang]
                plt.plot(range(len(y)), y, label=lang, color=colors_by_lang[lang], linewidth=2)

            plt.title(f"{model_name_map[model_type]} - {int(threshold * 100)}% Variance", fontsize=30)
            plt.xlabel("Layers", fontsize=30)
            plt.ylabel("# Components", fontsize=30)
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(title="Language", fontsize=25, title_fontsize=25)
            plt.tight_layout()

            if intervention_type == 'normal':
                save_path = os.path.join(output_dir, f"{model_type}_{int(threshold * 100)}")
            elif intervention_type == 'type-1':
                save_path = os.path.join(output_dir, f"{model_type}_{int(threshold * 100)}_type1")
            elif intervention_type == 'type-2':
                save_path = os.path.join(output_dir, f"{model_type}_{int(threshold * 100)}_type2")
            with PdfPages(save_path + '.pdf') as pdf:
                pdf.savefig(bbox_inches='tight', pad_inches=0.01)
                plt.close()