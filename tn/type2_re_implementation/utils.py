from baukit import TraceDict
import numpy as np
import torch
from tqdm import tqdm

def cache_last_token_hs(model, tokenizer, device, data, is_emb_layer_included=False) -> np.ndarray:
    hidden_states = np.zeros((model.config.num_hidden_layers, len(data), model.config.hidden_size)) if not is_emb_layer_included else np.zeros((model.config.num_hidden_layers+1, len(data), model.config.hidden_size))
    current_text_idx = None
    
    # set hooks.
    def cache_layer_output(layer_idx: int, hs: np.ndarray):
        def hook(m, inp, out):
            if isinstance(out, tuple):
                if not is_emb_layer_included:
                    hs[layer_idx, current_text_idx, :] = out[0][0, -1, :].detach().cpu().numpy()
                else: # emb layerを後で保存するので layer_idx=0 の次元はとっておく.
                    hs[layer_idx+1, current_text_idx, :] = out[0][0, -1, :].detach().cpu().numpy()

            return out
        
        return hook

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(cache_layer_output(layer_idx=layer_idx, hs=hidden_states))
        handles.append(handle)

    # run inference.
    for text_idx, text in tqdm(enumerate(data), total=len(data)):
        current_text_idx = text_idx
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=is_emb_layer_included)
        
        if is_emb_layer_included: # save emb layer hs to 0-th layer dims of hidden_states.
            hidden_states[0, current_text_idx, :] = outputs.hidden_states[0][0, -1, :].detach().cpu().numpy()
    
    # remove hooks.
    for handle in handles:
        handle.remove()

    return hidden_states

def edit_activation(output, layer, layer_idx_and_neuron_idx):
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if f"model.layers.{layer_idx}." in layer:
            output[:, -1, neuron_idx] *= 0

    return output

def cache_last_token_hs_with_edit_activation(model, tokenizer, device, data, is_emb_layer_included=False, neurons=None) -> np.ndarray:
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in neurons]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, neurons)) as tr:
        
        return cache_last_token_hs(model, tokenizer, device, data, is_emb_layer_included)