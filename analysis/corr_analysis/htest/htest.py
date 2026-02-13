import argparse

import numpy as np
from scipy.stats import f
from scipy.stats import ttest_ind

from utils.general_utils import (
    compute_corr_ratio,
    unfreeze_np_arrays,
    unfreeze_pickle,
)

def welch_t_test(labels, values):
    group0 = values[labels == 0]
    group1 = values[labels == 1]

    # Welch t test（equal_var=False: assuming unequal variances）.
    t_stat, p_value = ttest_ind(group0, group1, equal_var=False)
    
    eta_squared = compute_corr_ratio(labels, values)
    
    return eta_squared, t_stat, p_value

def compute_eta_squared_and_f(categories, values):
    cats = np.unique(categories)
    n_total = len(values)
    n_groups = len(cats)

    group_means = np.array([values[categories == c].mean() for c in cats])
    group_counts = np.array([np.sum(categories == c) for c in cats])
    overall_mean = values.mean()

    ss_between = np.sum(group_counts * (group_means - overall_mean)**2)
    ss_within = 0
    for c in cats:
        group_vals = values[categories == c]
        ss_within += np.sum((group_vals - group_vals.mean())**2)

    df_between = n_groups - 1
    df_within = n_total - n_groups

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F = ms_between / ms_within
    eta_squared = ss_between / (ss_between + ss_within)
    p_value = f.sf(F, df_between, df_within)

    return eta_squared, F, p_value

# label settings
l1 = [1] * 1000
l2 = [0] * 1000
# lang.
labels_dict_lang_specific = {
    'ja': l1 + l2 + l2 + l2 + l2,
    'nl': l2 + l1 + l2 + l2 + l2,
    'ko': l2 + l2 + l1 + l2 + l2,
    'it': l2 + l2 + l2 + l1 + l2,
    'en': l2 + l2 + l2 + l2 + l1,
}
# lang_family.
labels_dict_lang_family_specific = {
    'ja': l1 + l2 + l1 + l2 + l2,
    'nl': l2 + l1 + l2 + l1 + l1,
    'ko': l1 + l2 + l1 + l2 + l2,
    'it': l2 + l1 + l2 + l1 + l1,
    'en': l2 + l1 + l2 + l1 + l1,
}

def main(args):
    path = f'' # set this path to the array storing activations of each neurons (Get this array by executing 'analysis/corr_analysis/get_activations_for_corr_analysis.py' file).
    activations_arr = unfreeze_np_arrays(path)
    neurons = unfreeze_pickle(args.TN_path)
    labels_list = np.array(labels_dict_lang_specific[args.lang]) if args.lang_or_lang_family == 'lang_specific' else np.array(labels_dict_lang_family_specific[args.lang])
    significance_level = 0.05 # 
    corr_ratio_threshold = args.corr_ratio_th # corr_ratio = eta_squared.

    print(f"{args.TN_type}, lang={args.lang}, {'language specificity label' if args.lang_or_lang_family == 'lang' else 'language-family specificity label'}")

    eta_list = []
    p_list = []
    top_n = 1000 # top-1k neurons.
    for layer_i, neuron_i in neurons[:top_n]:
        vals = activations_arr[layer_i, neuron_i, :].astype(np.float64)
        eta, _, p_val = compute_eta_squared_and_f(labels_list, vals) # ANOVA.
        # eta, _, p_cal = welch_t_test(labels_list, vals) # Welch t test.
        eta_list.append(eta)
        p_list.append(p_val)

    eta_arr = np.array(eta_list)
    p_arr = np.array(p_list)
    significant_count = np.sum((p_arr < significance_level) & (eta_arr >= corr_ratio_threshold))

    # show results.
    print(f'{significant_count}/{top_n} neurons are SIGNIFICANT (p < {significance_level}) and η² >= {corr_ratio_threshold}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TN_type', type=str, required=True, choices=['type1', 'type2'], help='Type of transfer neurons.')
    parser.add_argument('--lang', type=str, required=True, default='ja', help='Language code of transfer neurons.') 
    parser.add_argument('--TN_path', type=str, required=True, help='Path to .pkl file that stores top-n transfer neurons.')
    parser.add_argument('--corr_ratio_th', type=float, required=True, choices=[0.1, 0.25], help='Correlation ratio threshold (this value is equal to the "eta_squared" in "compute_eta_squared_and_f" function).')
    parser.add_argument('--lang_or_lang_family', type=str, required=True, choices=['lang_specific', 'lang_family_specific'], help='Language specificity or language-family specificity.')
    args = parser.parse_args()

    main(args)

    """
    example usage:
    python -m analysis.corr_analysis.htest.htest \
    --TN_type type2 \
    --lang ja \
    --TN_path path/to/your/type2_neurons_ja \
    --corr_ratio_th 0.1 \
    --lang_or_lang_family lang_specific
    """