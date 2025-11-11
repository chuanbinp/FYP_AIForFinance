# FYP_AIForFinance
## Financial sentiment analysis of cryptocurrency microblog content
Phoe, C. B. (2022). AI for Finance. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/158241  

## About

This project tackles the limitations of existing sentiment lexicons and models in the context of financial and cryptocurrency social media. A hybrid approach: combining custom-built, domain-specific lexicons with deep learning models delivers state-of-the-art prediction accuracy, achieving a **82.6% F1-score** for actionable sentiment extraction from crypto microblog data.
  
Report under: FYPReport_PhoeChuanBin_U1821679J.pdf  
Poster under: Poster.pdf  
Presentation under: Presentation.pptx  
Grade: A+  
  
Code is in python and ipynb files, but _excluded data and BERT models due to size limitations_.

## Video Summary  
https://github.com/user-attachments/assets/1e210de1-3b41-48a7-b84b-3aee49acae08

## Poster
![image](https://user-images.githubusercontent.com/35805397/160373887-ed11936d-0ad2-4b5b-93c0-7db0bee7c92d.png)

## Motivation

Traditional sentiment tools fail in financial contexts due to:
- **Lack of temporal awareness** (e.g., “transitory” sentiment shift post-Fed policy changes)
- **Inability to capture crypto social media lingo** (terms like HODL, LFG, absent in general lexicons)
- **Limited domain training** (generic lexicons lack financial annotation/refresh, leading to low crypto accuracy)

This project bridges these gaps by creating finetuned lexicons and integrating them with neural models for robust, domain-aware sentiment analysis.

## Datasets

All datasets curated and annotated to maximize generalization and platform coverage:
- **SemEval 2017 Task 5**: 2,023 financial tweets
- **IEEE Data Port**: 1,300 S&P 500 annotated tweets
- **Scraped StockTwits small**: 1,161 self-annotated posts
- **Scraped StockTwits large**: 680,933 posts (with crowd-sourced sentiment tags)

**Manual and automated annotation procedures (Cohen’s Kappa: 0.855) ensure high-quality ground truth**.

## Architectural Overview

### Symbolic (Lexicon-based)  
- Custom lexicons:  
  - **StockTwitLexi**—Domain-specific, optimized via Bayesian/TF-IDF scoring  
  - **Senti-DD**—Context-aware, captures polysemy/directionality  
- Lexicon ensemble with voting (soft, hard, leave-2-out) across 7 sources and temporal/semantic trackers

### Sub-Symbolic (Deep Learning)  
- RNN variants: Uni-LSTM, Bi-LSTM, Uni-GRU, Bi-GRU  
- **BERT (domain-finetuned)**—best-in-class F1 for semantic contextualization

### Hybrid Approach  
- **Optimal weighted fusion**, symbolic + BERT (80/20), yielded **F1 = 0.826** (ensemble best, outperforming individual models)

## Key Results

- Domain-specific lexicons systematically outperform generic (StockTwitLexi: F1=0.756 vs. SenticNet: F1=0.715)
- Context-aware mechanisms (Senti-DD) improve ensemble performance despite weak solo performance
- **BERT outperforms RNNs** for crypto sentiment due to attention-based disambiguation of jargon
- Hybrid fusion architecture gives **synergistic gains**—balancing model interpretability and accuracy
- Multi-dataset training **ensures cross-platform generalization**

## For Investors

- End-to-end pipeline for extracting actionable insights from social media
- Research-backed, statistically-rigorous approach with explainability and robust empirical validation
- Direct relevance for alternative data-driven finance and real-time investment analytics



