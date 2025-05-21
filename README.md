# GPT-2 Model Overview

A concise, human-friendly guide to the GPT-2 “decoder-only” Transformer architecture and how its pieces fit together.

---

## 📦 What Is GPT-2?

GPT-2 is a large, autoregressive language model trained only to predict the next token in a text. Despite no task-specific supervision, its scale and training data give it surprisingly strong zero-shot performance on translation, Q&A, summarization, and more—all via clever prompt design.

---

## 🏗️ Core Components

1. **Decoder-Only Transformer**  
   - Stacks **48** identical layers (no separate encoder).  
   - Each layer applies **masked self-attention**, so position 𝑡 can only “see” positions ≤𝑡.  
   - Auto-regressive: generates text one token at a time.

2. **Hidden Representations**  
   - **Dimension (dₘₒdₑₗ): 1600**  
     Each token is mapped to a 1600-dimensional vector—enough capacity to capture syntax, semantics, and world knowledge.

3. **Multi-Head Self-Attention**  
   - **20 heads per layer**  
   - Each head learns a different “view” of the context (e.g., syntax vs. semantics).  
   - Heads’ outputs are concatenated and re-projected to 1600 dims.

4. **Feed-Forward Network**  
   - **Inner size: 4 × dₘₒdₑₗ = 6400**  
   - Two linear transforms with a nonlinear activation (GELU/ReLU):  
     1. 1600 → 6400  
     2. Activation  
     3. 6400 → 1600  

5. **Positional Encodings**  
   - Fixed sinusoidal vectors added to token embeddings to convey word order.  
   - Enables the model to reason about relative and absolute positions.

6. **Layer Normalization (Pre-Norm)**  
   - Normalizes token vectors before every attention and feed-forward block.  
   - Stabilizes training across 48 layers.

7. **Unified Next-Token Objective**  
   - No separate heads for classification or regression.  
   - All tasks (translation, summarization, Q&A, etc.) are framed as “complete the text.”

---

## 🚀 Why It Works

- **Scale**: 1.5 billion parameters + 40 GB of diverse WebText  
- **Emergent Abilities**: Learning to predict the next word implicitly teaches grammar, facts, reasoning patterns, and more.  
- **Zero-Shot Flexibility**: Any task you can describe in natural language can be “prompted” without extra fine-tuning.

---

## 📖 Example Prompt Usage

```text
Summarize the following article:
“Large language models have revolutionized NLP by…”
Summary:
