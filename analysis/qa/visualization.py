import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

from utils.general_utils import (
    unfreeze_pickle,
)

def get_q_above_th(THRESHOLD: int, normal: list, intervention: list, intervention_baseline: list):
    normal_r = []
    intervention_r = []
    intervention_baseline_r = []

    def get_q_idx_and_f1(score_list, target_idx):
        for qu_idx, f1_score in score_list:
            if qu_idx == target_idx:
                return qu_idx, f1_score
    
    normal_scores = []
    intervention_scores = []
    intervention_baseline_scores = []
    for q_idx, f1_score in normal:
        if f1_score >= THRESHOLD:
            n_score = f1_score
            normal_r.append((q_idx, f1_score))
            normal_scores.append(n_score)
            i_idx, i_score = get_q_idx_and_f1(intervention, q_idx) # i_idx: q_idx(intervention), i_score: f1_score(intervention)
            intervention_r.append((i_idx, i_score))
            intervention_scores.append(i_score)
            b_idx, b_score = get_q_idx_and_f1(intervention_baseline, q_idx) # b_idx: q_idx(intervention_baseline), b_score: f1_score(intervention_baseline)
            intervention_baseline_r.append((b_idx, b_score))
            intervention_baseline_scores.append(b_score)
    
    return normal_r, intervention_r, intervention_baseline_r, np.mean(np.array(normal_scores)), np.mean(np.array(intervention_scores)), np.mean(np.array(intervention_baseline_scores))

normal_dict = {}
intervention_dict = {}
intervention_baseline_dict = {}
    
langs = ['ja', 'nl', 'ko', 'it']
THRESHOLDS = [0.8, 0.5] # token-based f1 th.

for THRESHOLD in THRESHOLDS:
    for lang in langs:
        normal_list_path = f'data/qa/normal_{lang}.pkl' # .pkl file path that stores f1 scores under w/o intervention setting.
        normal = unfreeze_pickle(normal_list_path)
        intervention_path = f'data/qa/type1_{lang}.pkl' # .pkl file path that stores f1 scores under top-n Type-1 neurons intervention setting.
        intervention_baseline_path = f'data/qa/baseline_{lang}.pkl' # .pkl file path that stores f1 scores under random n. neurons intervention setting.
        intervention = unfreeze_pickle(intervention_path)
        intervention_baseline = unfreeze_pickle(intervention_baseline_path)

        # get q_idx and f1 above THRESHOLD.
        normal_l, intervention_l, intervention_baseline_l, normal_mean_score, intervention_mean_score, intervention_baseline_mean_score = get_q_above_th(THRESHOLD, normal, intervention, intervention_baseline)
        normal_dict[lang] = normal_l
        intervention_dict[lang] = intervention_l
        intervention_baseline_dict[lang] = intervention_baseline_l
        print(f'{lang}: w/o intervention:{normal_mean_score}, top-n type-1 intervention:{intervention_mean_score}, random n. intervention:{intervention_baseline_mean_score}')

    def plot_intervention_scatter(dict_normal, dict_intervene1, dict_intervene2):
        dicts = [dict_intervene1, dict_intervene2]
        dict_labels = ['Type-1 Neurons', 'Baseline']
        markers = ['s', '^']
        colors = ['green', 'red']

        # Union of all languages
        all_languages = sorted(set(dict_normal.keys()) | set(dict_intervene1.keys()) | set(dict_intervene2.keys()))

        plt.rc('font', family='Cambria Math')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']

        for lang in all_languages:
            base_data = dict_normal.get(lang, [])
            base_f1s = [f1 for _, f1 in base_data]

            if not base_data:
                continue

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('lightgray')

            for d_idx, d in enumerate(dicts):
                intervened_data = d.get(lang, [])
                if not intervened_data or len(intervened_data) != len(base_data):
                    continue
                intervened_f1s = [f1 for _, f1 in intervened_data]
                ax.scatter(
                    base_f1s,
                    intervened_f1s,
                    label=dict_labels[d_idx],
                    marker=markers[d_idx],
                    color=colors[d_idx],
                    alpha=0.8,
                )

            # Plot y=x line for reference
            min_f1 = min(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            max_f1 = max(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            ax.plot([min_f1, max_f1], [min_f1, max_f1], linestyle='--', color='blue', linewidth=2)

            ax.set_title(f'{lang}', fontsize=50)
            ax.set_xlabel('w/o Intervention', fontsize=40)
            ax.set_ylabel('Intervention', fontsize=40)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.tick_params(axis='both', labelsize=25)
            ax.legend(fontsize=35, markerscale=3)
            ax.grid(True)

            path = f'your/path/{lang}_{THRESHOLD}'
            with PdfPages(path + '.pdf') as pdf:
                pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
        
    # visualization
    plot_intervention_scatter(normal_dict, intervention_dict, intervention_baseline_dict)