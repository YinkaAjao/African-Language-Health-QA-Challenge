# African-Language-Health-QA-Challenge
Multilingual Health Question Answering in Low-Resource African Languages: A Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9A826)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
Developing healthcare Artificial Intelligence (AI) for low-resource African languages presents a critical dual challenge: the catastrophic risk of medical hallucinations and the severe scarcity of high-quality training data. 

This repository contains the complete end-to-end Machine Learning pipeline I engineered to answer health queries in dialects such as Swahili, Akan, Luganda, and Amharic. Through a rigorous progression of 27 experiments, I mapped the theoretical limits of statistical lexical algorithms (TF-IDF, BM25), parameter-efficient generative fine-tuning (mT5, ByT5, Gemma-2B), and dense semantic retrieval (E5, BGE-M3).

**Peak Performance:** Achieved a **Zindi Score of 0.5718** using a State-of-the-Art Dense Bi-Encoder (`BAAI/bge-m3`) augmented with strict algorithmic language-constraint masks.

---

## Key Findings & Architectural Breakthroughs

Throughout this research, I discovered that industry-standard NLP techniques built for high-resource languages actively degrade performance when applied to morphologically complex, data-starved African languages. 

1. **The Generative Bottleneck:** Small encoder-decoder models (~300M parameters) lack the capacity to learn agglutinative African grammar from subsampled data. Attempting Retrieval-Augmented Generation (RAG) on zero-shot LLMs caused catastrophic syntax failure.
2. **The Agglutinative Trap:** Upgrading traditional lexical models to BM25 crashed accuracy scores because whitespace tokenization destroys shared morphological prefixes/suffixes. Character-level N-Grams (TF-IDF) are far superior for African textual roots.
3. **The Cross-Lingual Semantic Flaw:** Dense embedding models (like `E5-Small`) perfectly map universal medical concepts but completely ignore dialect barriers, confidently returning perfect medical advice in the wrong language (scoring 0.0000 on the LLM Judge).
4. **The Winning Architecture (Experiment 22):** The optimal solution is to decouple semantic search from language filtering. By combining a deep dense retriever (`BGE-M3`) with a hard-coded Boolean language tensor mask, I mathematically eradicated cross-lingual hallucination.

---

## Leaderboard Progression (Top 10 Pivot Points)

| Exp | Architecture / Methodology | Zindi Score | ROUGE-L F1 | LLM Judge | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | Global TF-IDF (Char N-Grams) | 0.4956 | 0.4051 | 0.6437 | Strong Lexical Baseline |
| **09** | Custom Siamese GRU | 0.3336 | 0.2305 | 0.5070 | Data Starvation / Overfit |
| **11** | RAG (mT5 Zero-Shot) | 0.0515 | 0.0384 | 0.0840 | Generative Hallucination |
| **17** | ByT5-Small QLoRA | 0.0395 | 0.0462 | 0.0000 | Tokenizer-Free Sequence Failure |
| **18** | Gemma-2B QLoRA (Causal LLM) | 0.2801 | 0.1825 | 0.4345 | Generative Leap |
| **19** | Hybrid Pipeline (E5 + mT5) | 0.5305 | 0.4243 | 0.7299 | Breakthrough Hybrid |
| **20** | BGE-M3 Dense Retrieval | 0.5608 | 0.4617 | 0.7407 | SOTA Semantic Baseline |
| **22** | **Masked BGE-M3 (Constrained)** | **0.5718** | **0.5559** | **0.7216** | **Project Peak** |
| **24** | Okapi BM25 Lexical | 0.4552 | 0.4448 | 0.5958 | Tokenizer Whitespace Failure |
| **27** | The Platinum Pipeline (Reranker) | 0.5151 | 0.4878 | 0.7162 | Cross-Attention Failure |

---

This project was developed across Google Colab (for rapid statistical prototyping) and Kaggle (for dual-GPU LLM training).

## Repository Structure

```text
├── data/               # Target directory for Zindi Train/Test/Val CSVs
├── notebook/           # Notebook containing all experiments
├── models/             # Final Zindi-formatted submission CSVs
└── README.md

1. Clone the repository:
git clone [https://github.com/YinkaAjao/African-Language-Health-QA-Challenge.git]

2. Configure your environment:
Ensure your data files (Train.csv, Test.csv, Val.csv) are located in the /data directory. For Kaggle reproduction of the Gemma-2B architecture, ensure you have your Hugging Face API key mapped to HF_TOKEN in Kaggle Secrets.
3. Run the notebook
