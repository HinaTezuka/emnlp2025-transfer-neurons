from baukit import TraceDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.general_utils import (
    edit_activation,
    unfreeze_pickle,
)

"""
Funcs for computing mutual knn below (mutual_knn and compute_nearest_neighbors) are copied from: https://github.com/minyoungg/platonic-rep.
"""

def mutual_knn(feats_A, feats_B, topk=5):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0

    acc = (lvm_mask * llm_mask).sum(dim=1) / topk

    return acc.mean().item()

def compute_nearest_neighbors(feats, topk=5):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

def compute_mutual_knn(model, tokenizer, device, sentences: list, topk:int=5) -> list:
    layer_num = model.config.num_hidden_layers
    feats_L1 = torch.zeros((layer_num, len(sentences), model.config.hidden_size), device=device)
    feats_L2 = torch.zeros((layer_num, len(sentences), model.config.hidden_size), device=device)
    for txt_idx, (L1_txt, L2_txt) in tqdm(enumerate(sentences), total=len(sentences), desc=f'Computing mutual-knn for {len(sentences)} sentence pairs...'):
        inputs_L1 = tokenizer(L1_txt, return_tensors='pt').to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors='pt').to(device)
        # run inference.
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)
        hs_L1_all_layers = output_L1.hidden_states[1:]
        hs_L2_all_layers = output_L2.hidden_states[1:]
        for layer_idx in range(layer_num):
            hs_L1 = hs_L1_all_layers[layer_idx][:, -1, :].squeeze()
            hs_L2 = hs_L2_all_layers[layer_idx][:, -1, :].squeeze()
            feats_L1[layer_idx, txt_idx, :] = hs_L1
            feats_L2[layer_idx, txt_idx, :] = hs_L2
        
    # compute Mutual KNN.
    knn_scores = [] # [knn_score_layer1, knn_score_layer2, ...]
    feats_L1 = F.normalize(feats_L1, dim=-1)
    feats_L2 = F.normalize(feats_L2, dim=-1)
    for layer_idx in range(layer_num):
        knn_score = mutual_knn(feats_L1[layer_idx, :, :], feats_L2[layer_idx, :, :], topk=topk)
        knn_scores.append(knn_score)

    return knn_scores

def compute_mutual_knn_with_edit_activation(model, tokenizer, device, sentences: list, topk:int, layer_neuron_list:list):
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    # L1 hs.
    layer_num = model.config.num_hidden_layers
    feats_L1 = torch.zeros((layer_num, len(sentences), model.config.hidden_size), device=device)
    feats_L2 = torch.zeros((layer_num, len(sentences), model.config.hidden_size), device=device)
    for txt_idx, (L1_txt, L2_txt) in tqdm(enumerate(sentences), total=len(sentences), desc=f'Computing mutual-knn for {len(sentences)} sentence pairs...'):
        inputs_L1 = tokenizer(L1_txt, return_tensors='pt').to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors='pt').to(device)
        # L1 hs.
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
        hs_L1_all_layers = output_L1.hidden_states[1:]
        # L2 hs.
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
            with torch.no_grad():
                output_L2 = model(**inputs_L2, output_hidden_states=True)
        hs_L2_all_layers = output_L2.hidden_states[1:]    

        for layer_idx in range(layer_num):
            feats_L1[layer_idx, txt_idx, :] = hs_L1_all_layers[layer_idx][:, -1, :].squeeze()
            feats_L2[layer_idx, txt_idx, :] = hs_L2_all_layers[layer_idx][:, -1, :].squeeze()

    # calc Mutual KNN.
    knn_scores = [] # [knn_score_layer1, knn_score_layer2, ...]
    feats_L1 = F.normalize(feats_L1, dim=-1)
    feats_L2 = F.normalize(feats_L2, dim=-1)
    for layer_idx in range(layer_num):
        knn_score = mutual_knn(feats_L1[layer_idx, :, :], feats_L2[layer_idx, :, :], topk=topk)
        knn_scores.append(knn_score)
    
    return knn_scores


""" params. """
model_id = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
L1 = 'en'
topk = 10 # number of nearest neighbors (in the paper, 5 or 10).
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
intervention_type = 'w/o' # 'w/o' or 'type1' (w/o: without any intervention, type1: top-n type1 neurons intervention).

knn_scores = {}
for L2 in langs:
    path = f'path/to/sentences/parallel_sentence_pairs_between_en_{L2}' # In the paper, for sentences data, we used tatoeba corpus (1k translation sentence pairs between english and L2).
    sentences = unfreeze_pickle(path)

    if intervention_type == 'w/o':
        res = compute_mutual_knn(model, tokenizer, device, sentences, topk=topk) # res: [mknn_score_layer1, mknn_score_layer2, ...]
    elif intervention_type == 'type1':
        tn = unfreeze_pickle(f'path/to/top-n/type1/neurons')
        tn = [ n for n in tn if n[0] in [_ for _ in range(25)]][:1000]
        res = compute_mutual_knn_with_edit_activation(model, tokenizer, device, sentences, topk, tn)
    knn_scores[L2] = res


# visualization.
all_data = []
for lang in langs:
    values = knn_scores[lang]  # A list of length equal to the number of layers
    for layer, value in enumerate(values):
        all_data.append({
            'Layer': layer,
            'Mutual KNN': value,
            'L2': lang
        })

df = pd.DataFrame(all_data)

plt.rc('font',family='Cambria Math')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Layer', y='Mutual KNN', hue='L2', palette='tab10', linewidth=3)
model_name = 'your/model/name'
plt.title(f'{model_name}', fontsize=50)
plt.xlabel('Layer Index', fontsize=45)
plt.ylabel('Mutual KNN', fontsize=45)
plt.ylim(0, 0.6)
plt.tick_params(axis='both', labelsize=30)
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=25)
save_path = f'your/path/top{topk}'
with PdfPages(save_path + '.pdf') as pdf:
    pdf.savefig(bbox_inches='tight', pad_inches=0.01)
    plt.close()