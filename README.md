# Transformer from Scratch

### Building attention mechanisms manually, then applying Transformer representations to CIFAR-10

A PyTorch project exploring the core mechanics of Transformer architectures through direct tensor-level implementations of self-attention, multi-head attention and causal masking, followed by a Transformer-based image-classification experiment on CIFAR-10.

## Result

The CIFAR-10 classifier reached a best recorded validation accuracy of **82.8%** during 50 epochs of training.

| Metric | Result |
| --- | ---: |
| Best validation accuracy | **82.80%** |
| Best epoch | **46** |
| Epoch-50 validation accuracy | **82.72%** |
| Epoch-50 validation loss | **0.5578** |

## What I implemented

- scaled dot-product self-attention from tensor operations;
- manual Query, Key and Value projections;
- multi-head attention reshaping and parallel attention computation;
- validation against PyTorch attention implementations;
- causal attention masking;
- Transformer blocks with LayerNorm, residual connections and feed-forward networks;
- image patch embeddings for CIFAR-10;
- classification-token based image classification;
- training and validation loops with AdamW and data augmentation.

## Attention from scratch

The first part of the notebook constructs self-attention directly from matrix operations.

Given Query, Key and Value representations, attention scores are computed using scaled query-key similarity, normalised with softmax and applied to the value vectors.

The implementation is numerically checked against `torch.nn.functional.scaled_dot_product_attention`.

## Multi-head attention

The single-head implementation is extended to multiple heads by splitting the embedding dimension, performing attention independently in each head and concatenating the outputs.

The resulting tensors are compared against `torch.nn.MultiheadAttention` to validate the implementation.

## Causal masking

A triangular attention mask prevents each sequence position from attending to future positions, reproducing the autoregressive constraint used in decoder-style Transformer models.

Attention maps are visualised per head to inspect the effect of masking.

## Transformer block

A simplified Transformer block combines:

- Layer Normalisation;
- multi-head self-attention;
- residual connections;
- feed-forward layers;
- GELU activation;
- dropout.

## CIFAR-10 image classification

The final experiment applies Transformer representations to images.

CIFAR-10 images are divided into non-overlapping patches using a convolutional patch-embedding layer. The patch vectors are combined with a learnable classification token and passed through a Transformer encoder.

The classification-token representation is then passed to an MLP head predicting one of the ten CIFAR-10 classes.

The encoder in this experiment uses a configured Hugging Face `BertModel`; the lower-level attention mechanisms earlier in the notebook are implemented manually for comparison and understanding.

## Training setup

- dataset: CIFAR-10;
- patch size: 4 × 4;
- hidden dimension: 256;
- Transformer layers: 12;
- attention heads: 8;
- batch size: 192;
- optimiser: AdamW;
- learning rate: 5e-4;
- epochs: 50;
- augmentation: random crop and horizontal flip.

Validation performance peaked at **82.80% accuracy at epoch 46** and finished at **82.72% at epoch 50**.

## Repository structure

```text
transformer-from-scratch/
├── transformer_from_scratch.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tech

**Python · PyTorch · Torchvision · Transformers · self-attention · multi-head attention · causal masking · CIFAR-10 · vision transformers**

## Reference

The notebook was developed from a Transformer-from-scratch teaching tutorial and extended through implementation exercises, validation checks and the CIFAR-10 training experiment.
