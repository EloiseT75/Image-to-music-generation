# pIAno - Musical transformer BPE + GPT-like


This folder contains an implementation of a **Transformer-based music generation model trained on ABC notation**.

The goal of this part of the project is to explore how **symbolic music representations can be tokenized and modeled with autoregressive Transformers**, similarly to language modeling.

Music is treated as a **sequence generation problem**, where musical scores written in **ABC notation** are processed as text and converted into tokens before being used to train a Transformer model.

---

# Overview

The pipeline implemented in this folder explores how a Transformer can learn to generate music when trained on **symbolic music sequences**.

ABC notation is treated as a textual representation of music, allowing us to apply **language modeling techniques** directly to musical data.

The model learns to generate new melodies by predicting the **next token in an ABC sequence**, similarly to how GPT-like models generate text.

---

# Pipeline Summary

The full training pipeline follows these steps:

```
ABC dataset
      │
      ▼
Dataset loading and cleaning
(data_utils.py)
      │
      ▼
BPE tokenizer training
(tokenizer_utils.py)
      │
      ▼
Tokenization of ABC sequences
      │
      ▼
Transformer training
(model.py)
      │
      ▼
Autoregressive music generation
```

The complete pipeline is orchestrated inside the main notebook.

---

# Dataset

We use the **Irish Massive ABC Notation Dataset**, available on HuggingFace.

Dataset name:

```
sander-wood/irishman
```

Properties:

* ~216,000 traditional Irish tunes
* ABC notation format
* mostly monophonic melodies

Example tune:

```
X:10160
L:1/8
M:12/8
K:G
B | dcA BAG F2 G ABc | B2 c e2 d B3 BAB |
```

Dataset utilities are implemented in:

```
data_utils.py
```

This module:

* loads ABC sequences from the dataset
* filters empty entries
* splits the data into training and validation sets.

---

# Tokenization

Instead of designing handcrafted musical tokens, we **learn the token vocabulary directly from the dataset** using **Byte-Pair Encoding (BPE)**.

Tokenizer implementation:

```
tokenizer_utils.py
```

Tokenizer properties:

* vocabulary size: **1024 tokens**
* learned directly from the dataset
* byte-level pre-tokenization

The tokenizer automatically discovers recurring musical fragments such as:

```
BAG
dcA
|:
:|
```

These tokens often correspond to **frequent melodic patterns or structural markers in ABC notation**.

---

# Model

The model is a **causal Transformer language model** inspired by GPT architectures.

Implementation:

```
model.py
```

Architecture:

* Transformer layers: **4**
* Attention heads: **4**
* Embedding dimension: **256**
* Context length: **256 tokens**
* Vocabulary size: **1024**

Main components include:

* token embeddings
* positional embeddings
* multi-head self-attention
* feed-forward layers
* autoregressive decoding

The model is trained using a **next-token prediction objective** with **cross-entropy loss**.

---

# Training

The model is trained on tokenized ABC sequences to predict the **next token in a sequence**.

Example training dynamics:

```
step 0      train loss ≈ 7.1
step 1000   train loss ≈ 2.6
step 2800   train loss ≈ 2.0
```

Final validation metrics:

* validation loss ≈ **2.0**
* perplexity ≈ **7.4**

These results indicate that the model successfully learns **statistical musical patterns present in the dataset**.

---

# Generation

After training, the model can generate new musical sequences in ABC notation.

Example output:

```
X:1
T:Generated Tune
M:4/4
L:1/8
K:G
G2 | G2 G>B B>G | G>AB>G G>B | d>ef>g |
```

Generated melodies typically show:

* plausible ABC structure
* recurring melodic patterns
* stylistic similarity with the training dataset.

However, longer generations may still show:

* repetition
* minor syntax errors
* concatenation of multiple tunes.

---

# Memorization Analysis

We verified that the model does not simply copy training tunes.

Two tests were performed.

Maximum similarity with the dataset:

```
≈ 0.46
```

Longest exact substring overlap:

```
14 characters
```

Example fragment:

```
K:G
G2 | G2
```

This fragment is extremely common in the dataset and does not indicate memorization.

Conclusion:

The model reproduces **stylistic motifs** but does not copy complete tunes from the dataset.

---

# Project Structure

```
Musical_transformer_BPE+GPT-like/

data_utils.py
tokenizer_utils.py
model.py
abc_transformer.ipynb
README.md
```

File descriptions:

* **data_utils.py**
  Loads the ABC dataset and performs train/validation splitting.

* **tokenizer_utils.py**
  Trains and loads the BPE tokenizer and provides encoding/decoding utilities.

* **model.py**
  Contains the GPT-like Transformer architecture and generation function.

* **abc_transformer.ipynb**
  Main notebook that runs the full training and generation pipeline.

---

# Running the Project

Install dependencies:

```
pip install torch datasets tokenizers
```

## Training and Generation

The **entire pipeline is executed inside the main notebook**:

```
abc_transformer.ipynb
```

The notebook performs the full workflow:

1. Load and preprocess the dataset
2. Train the BPE tokenizer
3. Encode ABC sequences into tokens
4. Train the Transformer model
5. Generate new musical sequences

The Python modules in this folder are imported by the notebook to implement each component of the pipeline.

---

# Limitations

This model is intentionally **small and simple** to keep the pipeline clear and easy to analyze.

Known limitations include:

* repetitive melodic patterns
* limited long-range musical structure
* occasional ABC syntax errors
* small context window

Despite these limitations, the model demonstrates how **symbolic music can be modeled using Transformer language models**.
