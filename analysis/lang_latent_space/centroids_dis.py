from collections import defaultdict
from itertools import permutations
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from utils.general_utils import (
    unfreeze_np_arrays
)

langs = ['ja', 'nl', 'ko', 'it', 'en']
intervention_type = 'type2' # choice = ['normal', 'type1', 'type2'] normal: w/o intervention.

hs_ja = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/{intervention_type}_ja.npz')
hs_nl = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/{intervention_type}_nl.npz')
hs_ko = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/{intervention_type}_ko.npz')
hs_it = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/{intervention_type}_it.npz')
hs_en = unfreeze_np_arrays(f'data/inner_reps/hs_for_lang_latent/{intervention_type}_en.npz')

sim_dict = defaultdict(lambda: defaultdict(lambda: np.float64(0)))

for layer_i in range(33): # 33: emb layer + 32 decoder layers.
    if (intervention_type == 'type1' and layer_i in [ _ for _ in range(21, 33)]) or (intervention_type == 'type2' and layer_i in [ _ for _ in range(21)]):
        continue

    hs_ja_layer = hs_ja[layer_i, :, :] # shape: (n * d) n: sample_num, d: dimention of hs.
    hs_nl_layer = hs_nl[layer_i, :, :]
    hs_ko_layer = hs_ko[layer_i, :, :]
    hs_it_layer = hs_it[layer_i, :, :]
    hs_en_layer = hs_en[layer_i, :, :]

    lang2hs_layer = {
        "ja": hs_ja_layer,
        "nl": hs_nl_layer,
        "ko": hs_ko_layer,
        "it": hs_it_layer,
        "en": hs_en_layer,
    }

    for lang1, lang2 in permutations(langs, 2):
        c1 = np.mean(lang2hs_layer[lang1], axis=0).reshape(1, -1)
        c2 = np.mean(lang2hs_layer[lang2], axis=0).reshape(1, -1)

        sim_score = cosine_similarity(c1, c2).item()

        # record all avg_sims into sim_dict
        sim_dict[f'{lang1}-{lang2}'][layer_i] = sim_score

    """ plot """
    lang_sim_matrix = np.zeros((len(langs), len(langs)))
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if lang1 == lang2:
                lang_sim_matrix[i, j] = 1.0
            else:
                key = f"{lang1}-{lang2}"
                key_rev = f"{lang2}-{lang1}"

                if key in sim_dict and layer_i in sim_dict[key]:
                    sim = sim_dict[key][layer_i]
                    lang_sim_matrix[i, j] = sim
                elif key_rev in sim_dict and layer_i in sim_dict[key_rev]:
                    sim = sim_dict[key_rev][layer_i]
                    lang_sim_matrix[i, j] = sim

    # plot heatmap
    plt.rc('font',family='Cambria Math')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
    plt.figure(figsize=(8.5, 8))
    ax = sns.heatmap(
        lang_sim_matrix,
        xticklabels=langs,
        yticklabels=langs,
        annot=True,
        cmap='Blues',
        fmt='.2f',
        vmin=0,
        vmax=1,
        square=True,
        annot_kws={"size": 20}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.title(f"Layer {layer_i}", fontsize=30)
    plt.tick_params(labelsize=30)
    if intervention_type == 'type1':
        for label in ax.get_xticklabels():
            if label.get_text() == 'en':
                label.set_color('red')
                label.set_fontweight('bold')
                label.set_fontsize(40)
        for label in ax.get_yticklabels():
            if label.get_text() == 'en':
                label.set_color('red')
                label.set_fontweight('bold')
                label.set_fontsize(40)

    # set this path (save_dir) properly.
    if intervention_type == 'normal':
        save_dir = f''
    elif intervention_type == 'type1':
        save_dir = f''
    elif intervention_type == 'type2':
        save_dir = f''

    os.makedirs(save_dir, exist_ok=True)
    if layer_i == 0:
        save_path = f'{save_dir}/emb_layer'
    else:
        save_path = f'{save_dir}/layer_{layer_i}'

    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()