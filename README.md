# Albat

This is the repository for code developed for the dissertation "Domain Adaptation through Adversarial Training of
Albert for Aspect-based Sentiment Analysis".

We have used code derived from "[Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)" and 
"[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)", performing aspect-based sentiment analysis (ABSA) by using adversarial training with [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942).

## ABSA Tasks
We focus on three tasks in ABSA:

- **Aspect Extraction (AE)**: given a review sentence, find the aspect phrases;

- **Aspect Sentiment Classification (ASC)**: given an aspect and a review sentence, detect the polarity of that aspect;

- **End-to-end ABSA (E2E-ABSA)**: given a review sentence, find aspects and classify their sentiment polarities.

## Execution

The Jupyter notebook `ALBAT_Loader.ipynb` contains code for fine-tuning `albert-base-v2` on the three ABSA tasks with various datasets. We also provide instructions for further pre-training Albert in the markdown document `pt_model/albat_pt_1/readme.md`.