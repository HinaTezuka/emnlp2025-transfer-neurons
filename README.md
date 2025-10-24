# The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs.

update comming soon...

paper: 

## Set Up
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
**example usage (for Type-1 neurons):**
```bash
python -m tn.calc_centroids_for_tn_detection \
    --model_id mistralai/Mistral-7B-v0.3 \
    --TN_Type type1 \
    --lang_for_TN ja \
    --sentence_path path/to/your/parallel_sentences_ja.pkl
```
**example usage (for Type-2 neurons):**
```bash
python -m tn.calc_centroids_for_tn_detection \
    --model_id mistralai/Mistral-7B-v0.3 \
    --TN_Type type2 \
    --lang_for_TN ja \
    --sentence_path path/to/your/monolingual_sentences_ja.pkl
```

### Transfer Neurons Detection
**example usage (for Type-1 neurons):**  
```bash
python -m tn.detect_tn \
    --model_id mistralai/Mistral-7B-v0.3 \
    --TN_Type type1 \
    --top_n 1000 \
    --lang_for_TN ja \
    --scoring_type cos_sim \
    --centroids_path path/to/your/centroids_for_type1_detection_ja.pkl \
    --sentence_path path/to/your/monolingual_sentences_ja.pkl
```
**example usage (for Type-2 neurons):**
```bash
python -m tn.detect_tn \
    --model_id mistralai/Mistral-7B-v0.3 \
    --TN_Type type1 \
    --top_n 1000 \
    --lang_for_TN ja \
    --scoring_type cos_sim \
    --centroids_path path/to/your/centroids_for_type2_detection_ja.pkl \
    --sentence_path path/to/your/monolingual_sentences_ja.pkl
```
As a distance function, you may choose either "cos_sim" (Cosine similarity) or "L2_dis" (Euclidean distance).

**Notes:** In our paper, we used LLMs consisting of 32 decoder layers. Accordingly, we set the candidate layers to 1–20 for identifying Type-1 neurons and 21–32 for identifying Type-2 neurons.
If you use an LLM with a different number of layers, please adjust the candidate layer range as appropriate by modifying ```candidate_layers_range = 20 if tn_type == 'type1' else 32``` in the ```tn/detect_tn.py``` file.

## Citation
```
@inproceedings{tezuka-etal-2025-tn,
    title={The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs.},
    author={Tezuka, Hinata and Inoue, Naoya},
    booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2025}
}
```

## Acknowledgemants
We would like to thank [baukit](https://github.com/davidbau/baukit) , for providing an excellent package that greatly facilitated our experiments.