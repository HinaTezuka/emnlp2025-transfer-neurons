# The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs.

update comming soon...

paper: 

## Set up
```bash
python -m venv tn_detection
pip install -r requirements.txt
source bin/tn_detection/activate
```

## Transfer Neurons Detection
### Sentence Dataset Requirements
- L1: fixed to English (We assume English latent space serves as a shared semantic latent space in middle layers).
- L2: language you want to detect as transfer neurons (in the paper, we detected TNs for ja/nl/ko/it).
- For Type-1 neurons detection: sentences must be a list of parallels pairs(tuple) of english-L2: ```[(L1_sentence1, parallel_sentence_in_L2), (L1_sentence2, parallel_sentence_in_L2), ...]```
- For Type-2 neurons detection: sentence must be a list of L2 sentences: ```[L2_sentence1, L2_sentence2, L2_sentence3, ...]```

You can use any sentence datasets as long as the datasets meet the conditions above.  
(In the paper, we used tatoeba corpus: https://huggingface.co/datasets/Helsinki-NLP/tatoeba)

### Centroids Estimation
**For the centroids of shared semantic latent space (Type-1 neurons):**  
example usage:
```bash
python -m tn.calc_centroids_for_tn_detection \
    mistralai/Mistral-7B-v0.3 \
    type1 \
    ja \
    data/example_sentences/ja_parallel_train.pkl
```