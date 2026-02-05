## Ablation study on Type-2 neurons.

In the paper, we used the codes like below to aggregate hidden states for centroids estimation of neurons detection.  
```python
with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)
hidden_states_all_decoder_layers = output.hidden_states[1:] # exclude embedding layer.
```

Since `` output_hidden_states=True `` of transformers library usually caches normalized hidden states only for final decoder layer of LLMs ( i.e, ``model.norm(hidden_states)`` for ``output.hidden_states[-1]`` ), the resulting estimated centroid for last decoder layer is layer normalized.  
In our scoring computation for transfer neurons detection (Eqs. 7, 8, and 9 in the paper), we compute distance (i.e, cosine similarity) between ``hidden_states_of_last_layer + attention_output_of_current_layer (+ only_one_neuron's_effect_in_mlp)`` and ``estimated_centroid_of_current_layer``.  

Hence, in this repo, we provide ablation codes for estimating the centroid of last decoder layer with non-normalized (raw) hidden_states and re-compute neuron scores for Type-2 neurons detection. (Note that this only matters for last decoder layer and Type-2 neurons detection.)
(As a results, re-identified Type-2 neurons yielded similar distributions and results as those provided in the paper.)