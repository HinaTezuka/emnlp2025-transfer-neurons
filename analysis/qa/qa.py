import argparse
from collections import Counter
import os
import random
import re
import string
from typing import List, Tuple

from baukit import TraceDict
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.general_utils import (
    generate_baseline_neurons,
    save_as_pickle,
    unfreeze_pickle,
)

""" Evaluation funcs have been cited from: https://github.com/apple/ml-mkqa. """
MIXED_SEGMENTATION_LANGS = ["zh_cn", "zh_hk", "zh_tw", "ja", "th", "km"]
ARTICLE_REGEX_BY_LANG = {
    "en": r"\b(a|an|the)\b",
    "es": r"\b(un|una|unos|unas|el|la|los|las)\b",
    "vi": r"\b(của|là|cái|chiếc|những)\b",
    "de": r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
    "ar": "\sال^|ال",
    "nl": r"\b(de|het|een|des|der|den)\b",
    "sv": r"\b(en|ett)\b",
    "da": r"\b(en|et)\b",
    "no": r"\b(en|et|ei)\b",
    "fr": r"\b(le|la|l'|les|du|de|d'|des|un|une|des)",
    "pt": r"\b(o|a|os|as|um|uma|uns|umas)\b",
    "it": r"\b(il|lo|la|l'|i|gli|le|del|dello|della|dell'|dei|degli|degl'|delle|un'|uno|una|un)",
    "fi": r"\b(se|yks|yksi)\b",
    "hu": r"\b(a|az|egy)\b",
}

def whitespace_tokenize(text):
    return text.split()

def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
            temp_str = ""
        segs_out.append(char)

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

def normalize_answer_by_language(s, lang):
    """Lower text, remove punctuation, articles and extra whitespace.
    This function is customized by language.
    """

    def remove_articles(text, lang):
        article_regex = ARTICLE_REGEX_BY_LANG.get(lang)
        if article_regex:
            return re.sub(article_regex, " ", text)
        else:
            return text

    def white_space_fix(text, lang):

        if lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def calculate_f1(prediction, gold_answer, language):
    gold_toks = normalize_answer_by_language(gold_answer, language).split() if gold_answer else []
    pred_toks = normalize_answer_by_language(prediction, language).split() if prediction else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_common = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If the prediction or gold_answer is No Answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_common == 0:
        return 0.0

    recall = 1.0 * num_common / len(gold_toks)
    precision = 1.0 * num_common / len(pred_toks)
    return (2.0 * precision * recall) / (precision + recall)

def edit_activation_prompt_last(output, layer, layer_idx_and_neuron_idx):
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if f"model.layers.{layer_idx}." in layer and output.shape[1] != 1:
            output[:, -1, neuron_idx] *= 0

    return output

def edit_activation_prompt(output, layer, layer_idx_and_neuron_idx):
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if f"model.layers.{layer_idx}." in layer and output.shape[1] != 1:
            output[:, :, neuron_idx] *= 0

    return output

def mkqa_all(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, qa, L2: str, max_new_tokens: int):
    f1_for_vis = []

    for q_idx, sample in tqdm(enumerate(qa), total=len(qa), desc=f'{L2}...'):
        # answer type: long_answer, unanswerableは飛ばす.
        ans_type = sample['answers']['en'][0]['type']
        if ans_type in [1, 2]: continue

        q = sample['queries'][L2] # question
        a = []
        for ans in sample['answers'][L2]:
            if ans.get('text'):
                a.append(ans['text'])
            if ans.get('aliases'):
                a.extend(ans['aliases'])
        seen = set()
        a = [ans for ans in a if ans and not (ans in seen or seen.add(ans))]
            
        ans_type = sample['answers']['en'][0]['type']

        # make prompt.
        if L2 == 'ja': prompt = f'{q}? 答え: '
        elif L2 == 'nl': prompt = f'{q}? Antwoord: '
        elif L2 == 'ko': prompt = f'{q}? 답변: '
        elif L2 == 'it': prompt = f'{q}? Risposta: '
        elif L2 == 'en': prompt = f'{q}? Answer: '

        # run inference.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_len = inputs['input_ids'].size(1)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        # extract generated sequence.
        pre = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
        
        # f1
        f1_l = []
        for ans in a:
            f1_l.append(calculate_f1(pre, ans, L2))
        f1 = max(f1_l)

        f1_for_vis.append((q_idx, f1))
    
    return f1_for_vis

def mkqa_all_with_edit_activation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, qa, L2: str, layer_neuron_list: List[Tuple[int, int]], max_new_tokens: int, deact_type: str):
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    if deact_type == 'prompt':
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_prompt(output, layer, layer_neuron_list)) as tr:
            return mkqa_all(model, tokenizer, device, qa, L2, max_new_tokens)
    elif deact_type == 'prompt_last':
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_prompt_last(output, layer, layer_neuron_list)) as tr:
            return mkqa_all(model, tokenizer, device, qa, L2, max_new_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model_path at HuggingFace Hub.')
    parser.add_argument('--language', type=str, required=True, default='ja', help='Language for QA.')
    parser.add_argument('--max_new_tokens', type=int, required=True, choices=[5, 10])
    parser.add_argument('--deactivation_type', type=str, required=True, choices=['prompt', 'prompt_last'])
    parser.add_argument('--type1_neurons_path', type=str, required=True, help='.pkl file path that stores type-1 neurons.')
    args = parser.parse_args()

    model_id = args.model_id
    lang = args.language
    max_new_tokens = args.max_new_tokens
    deact_type = args.deactivation_type
    type1_neurons_path = args.type1_neurons_path

    # models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    qa = load_dataset('apple/mkqa')['train']

    # w/o any intervention (normal).
    f1_for_vis = mkqa_all(model, tokenizer, device, qa, lang, max_new_tokens)
    path_f1 = f'data/qa/normal_{lang}.pkl'
    os.makedirs(os.path.dirname(path_f1), exist_ok=True)
    save_as_pickle(path_f1, f1_for_vis)
    print(f'Completed: {lang}, w/o intervention.')
    
    # top-n Type-1 neurons intervention.
    type1_neurons = unfreeze_pickle(type1_neurons_path)
    f1_for_vis = mkqa_all_with_edit_activation(model, tokenizer, device, qa, lang, type1_neurons, max_new_tokens, deact_type)
    # save f1 scores for visualization.
    path_f1 = f'data/qa/type1_{lang}.pkl'
    os.makedirs(os.path.dirname(path_f1), exist_ok=True)
    save_as_pickle(path_f1, f1_for_vis)
    print(f'Completed: {lang}, type1 intervention.')

    # random n. neurons intervention (baseline).
    random.seed(42)
    baseline_neurons = generate_baseline_neurons(type1_neurons, (0, 20), (0, model.config.intermediate_size))
    f1_for_vis = mkqa_all_with_edit_activation(model, tokenizer, device, qa, lang, baseline_neurons, max_new_tokens, deact_type)
    path_f1 = f'data/qa/baseline_{lang}.pkl'
    os.makedirs(os.path.dirname(path_f1), exist_ok=True)
    save_as_pickle(path_f1, f1_for_vis)
    
    print(f'Completed: {lang}, random n. intervention.')