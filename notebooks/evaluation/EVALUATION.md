# Evaluation Results

Evaluation performed on the Flickr30k validation set (1,014 images, 5 reference captions per image).

---

## 1. Summary Comparison

| Optimization Round | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy Search** | 0.6163 | 0.4118 | 0.2771 | 0.1903 | 0.1931 | 0.4775 | 0.3701 |
| **Beam Search** (`beam_size=3`) | **0.6563** | **0.4499** | **0.3139** | **0.2210** | **0.1969** | **0.4913** | **0.4263** |
| **Attention + Word2Vec (Beam Search)** | 0.6511 | 0.4489 | 0.3112 | 0.2176 | **0.2045** | **0.4946** | **0.4591** |
| **Attention (Beam Search)** | **0.6838** | **0.4805** | **0.3423** | **0.2451** | **0.2067** | **0.5023** | **0.5003** |

> *Note on Beam Search*: With GNMT length penalty (`alpha=1.0`), METEOR increased to 0.1989. (Without length penalty: BLEU-1: 0.6608, BLEU-4: 0.2204, METEOR: 0.1959, ROUGE-L: 0.4917, CIDEr: 0.4201).

---

## 2. Evaluation Details

### Round 1: Baseline + Greedy Search
- **Decoding Method**: `greedy`
- **Evaluated Images**: 1014
- **Parameters**:
  - `max_len`: 30
  - `batch_size`: 16
- **Metrics**:
  - `bleu1`: 0.6163
  - `bleu2`: 0.4118
  - `bleu3`: 0.2771
  - `bleu4`: 0.1903
  - `meteor`: 0.1931
  - `rouge_l`: 0.4775
  - `cider`: 0.3701

### Round 2: Baseline + Beam Search
- **Decoding Method**: `beam`
- **Evaluated Images**: 1014
- **Parameters**:
  - `beam_size`: 3
  - `length_penalty_alpha`: 0.6
  - `max_len`: 30
  - `batch_size`: 16
- **Metrics**:
  - `bleu1`: 0.6563
  - `bleu2`: 0.4499
  - `bleu3`: 0.3139
  - `bleu4`: 0.2210
  - `meteor`: 0.1969
  - `rouge_l`: 0.4913
  - `cider`: 0.4263

### Round 3: Attention + Word2Vec (Beam Search)
- **Model**: Attention decoder with pretrained Word2Vec embeddings
- **Decoding Method**: `greedy`
- **Evaluated Images**: 1014
- **Parameters**:
  - `max_len`: 30
  - `batch_size`: 16
- **Metrics**:
  - `bleu1`: 0.6511
  - `bleu2`: 0.4489
  - `bleu3`: 0.3112
  - `bleu4`: 0.2176
  - `meteor`: 0.2045
  - `rouge_l`: 0.4946
  - `cider`: 0.4591

### Round 4: Attention (Beam Search)
- **Model**: Attention decoder
- **Decoding Method**: `beam`
- **Evaluated Images**: 1014
- **Metrics**:
  - `bleu1`: 0.6838
  - `bleu2`: 0.4805
  - `bleu3`: 0.3423
  - `bleu4`: 0.2451
  - `meteor`: 0.2067
  - `rouge_l`: 0.5023
  - `cider`: 0.5003

