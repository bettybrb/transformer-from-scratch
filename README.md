# Transformer from Scratch

A deep learning project exploring the internal mechanics of Transformer architectures, from manually implementing scaled dot-product attention and multi-head attention to training a Transformer-based image classifier on CIFAR-10.

The project builds the core Transformer operations step by step in PyTorch before applying the same principles to computer vision.

## Project Overview

The notebook progresses through several stages of Transformer implementation:

1. scaled dot-product self-attention
2. multi-head self-attention
3. causal attention masking
4. Transformer blocks with residual connections and feed-forward layers
5. image patch embeddings
6. Transformer-based CIFAR-10 classification

This provides both a low-level implementation of the attention mechanism and a practical application of Transformers to image classification.

## Self-Attention from Scratch

The first section manually constructs Query, Key and Value projections and computes attention using matrix operations.

Attention scores are calculated from query-key similarities, scaled by the head dimension and normalised with softmax before being used to produce weighted value representations.

The implementation is checked against PyTorch's built-in scaled dot-product attention to verify the result.

## Multi-Head Attention

The single-head implementation is extended to multi-head attention using a 768-dimensional embedding split across 12 attention heads.

Query, Key and Value representations are reshaped into independent heads, attention is calculated in parallel, and the resulting representations are concatenated back into the original embedding dimension.

The implementation is compared against `torch.nn.MultiheadAttention` for validation.

## Causal Attention

A causal attention mask is constructed to prevent tokens from attending to future positions. Attention matrices for individual heads are visualised to inspect the resulting attention patterns.

## Transformer Block

A simplified Transformer block is implemented using:

- Layer Normalisation
- Multi-Head Attention
- Residual Connections
- Feed-Forward Networks
- GELU activation
- Dropout

This demonstrates how the attention mechanism fits into the larger Transformer architecture.

## Transformer-Based Image Classification

The final section applies Transformer concepts to computer vision using the CIFAR-10 dataset.

Images are divided into patches using a convolutional patch embedding layer. The resulting patch representations are treated as a sequence and combined with a learnable classification token.

A Transformer encoder processes the patch sequence and the final classification-token representation is passed through a neural-network readout head to predict one of the ten CIFAR-10 classes.

## Training

The image classifier uses:

- CIFAR-10 training and validation data
- random cropping and horizontal flipping for augmentation
- image normalisation
- AdamW optimisation
- cross-entropy classification loss
- training loss and accuracy tracking

Training behaviour was explored across 10, 30 and 50 epochs to examine how optimisation duration affected convergence.

## Repository Structure

- `transformer_from_scratch.ipynb` - complete attention, Transformer and CIFAR-10 classification implementation
- `requirements.txt` - Python dependencies
- `.gitignore` - excludes generated datasets, checkpoints and local environment files

## Technologies

- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- NumPy
- Matplotlib
- CIFAR-10

## Installation

```bash
pip install -r requirements.txt
```

## Concepts Demonstrated

- Scaled dot-product attention
- Query, Key and Value projections
- Multi-head self-attention
- Causal attention masking
- Transformer architecture
- Residual connections
- Feed-forward networks
- Image patch embeddings
- Classification tokens
- Vision Transformers
- Data augmentation
- Deep learning optimisation
- Image classification

## Motivation

Rather than treating Transformers as a black-box library component, this project builds their central operations directly with tensor mathematics before applying them to a practical computer-vision task. The progression from individual attention calculations to an end-to-end image classifier provides a practical understanding of how Transformer architectures operate internally.
