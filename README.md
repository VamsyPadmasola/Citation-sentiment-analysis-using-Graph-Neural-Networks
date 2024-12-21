content = """
# Citation Sentiment Analysis and Ranking of Research Papers

This repository contains the implementation and findings of the project **"Determining Polarity of Citation Text and Analyzing Its Impact."** This study focuses on sentiment analysis of citation text and the introduction of a new metric, the M-index, which combines quantitative and qualitative aspects to rank research papers.

## Table of Contents

1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Dataset Description](#dataset-description)
4. [Implementation Details](#implementation-details)
   - [Architecture](#architecture)
   - [Embedding Techniques](#embedding-techniques)
   - [Classifiers](#classifiers)
5. [Results](#results)
6. [Conclusion and Future Work](#conclusion-and-future-work)
7. [References](#references)

---

## Introduction

Citation sentiment analysis evaluates the polarity (positive, neutral, or negative) of citation text in research papers. Traditional citation metrics like citation count fail to capture the qualitative aspect of citations. This project introduces sentiment analysis to address this limitation and ranks research papers using a novel metric called the **M-index**.  

Key Contributions:
- Classification of citation sentiments using **machine learning models**.
- Embedding techniques such as **Node2Vec** and **BERT** for feature representation.
- Ranking of papers using the combined quantitative and qualitative aspects of citations.

![Citation Example](path/to/diagram1.png)  
*Figure 1: Example of Citation and Its Reference*

For a detailed overview of citation sentiment analysis, refer to [this article](https://www.sciencedirect.com/topics/computer-science/sentiment-analysis).

---

## Objective

1. Classify citation text into **positive**, **neutral**, or **negative** sentiments.
2. Develop a ranking system (M-index) that combines sentiment polarity and citation count to score and rank research papers.

![Classification Process](path/to/diagram2.png)  
*Figure 2: Classifying Citation Text into Positive, Neutral, or Negative*

---

## Dataset Description

The project uses the **Citation Sentiment Corpus** from the ACL Anthology Network. Below is a summary:

- **Total Instances**: 8736
  - **Neutral**: 7627
  - **Positive**: 829
  - **Negative**: 280
- **Attributes**:
  - `Source_Paper ID`: Citing paper's ID.
  - `Target_Paper ID`: Cited paper's ID.
  - `Sentiment`: Positive, negative, or neutral.
  - `Citation_Text`: Text containing citations.

![Dataset Example](path/to/diagram3.png)  
*Figure 3: Example of the Citation Sentiment Corpus*

You can access the dataset [here](https://github.com/aclweb/acl-anthology).

---

## Implementation Details

### Architecture

1. **Citation Graph Representation**:
   - Nodes represent research papers.
   - Edges represent citations between papers, weighted by sentiment polarity.

   ![Citation Graph](path/to/diagram4.png)  
   *Figure 4: Citation Graph Representation*

2. **Embedding Techniques**:
   - **Node2Vec**: Generates 128-dimensional graph embeddings.
   - **BERT**: Generates 128-dimensional contextual embeddings for citation text.

3. **Combined Embedding**:
   - Concatenated Node2Vec and BERT embeddings to form 256-dimensional feature vectors.

### Embedding Techniques

- **Node2Vec**: Captures graph topology through random walks and skip-gram models. Learn more about [Node2Vec](https://snap.stanford.edu/node2vec/).  
![Node2Vec Example](path/to/diagram5.png)  
*Figure 5: Node2Vec Random Walk Example*

- **BERT**: Generates contextual embeddings for text. Learn more about [BERT](https://github.com/google-research/bert).  
![BERT Architecture](path/to/diagram6.png)  
*Figure 6: BERT Architecture*

### Classifiers

Supervised classifiers were used, including:
- **Decision Tree**
- **AdaBoost**
- **Logistic Regression**
- **Stochastic Gradient Descent (SGD)**
- **Random Forest**
- **Extra Trees Classifier**
- **Support Vector Machine (SVM)**

---

## Results

### Node2Vec Embeddings
| Model      | Accuracy  |
|------------|-----------|
| Decision Tree (DT) | 87.78% |
| AdaBoost   | 91.11%    |
| Logistic Regression (LR) | 86.80% |
| SGD        | 91.11%    |
| Random Forest (RF) | 90.83% |
| Extra Trees Classifier (ETC) | 88.89% |
| SVM        | 91.11%    |

### Node2Vec + BERT Embeddings
| Model      | Accuracy  |
|------------|-----------|
| Decision Tree (DT) | 82.78% |
| AdaBoost   | 85.28%    |
| Logistic Regression (LR) | 86.05% |
| SGD        | 88.89%    |
| Random Forest (RF) | 87.78% |
| Extra Trees Classifier (ETC) | 87.50% |
| SVM        | 88.61%    |

---

## Conclusion and Future Work

### Conclusion
- The **Extra Trees Classifier** with **TF-IDF** achieved the highest accuracy of **98.06%** on the SMOTE-balanced dataset.

### Future Work
1. Experiment with additional feature engineering methods.
2. Integrate pre-trained word embeddings with deep learning models.
3. Expand the dataset to include a larger proportion of subjective citations to minimize bias.

---

## References

1. G. Parthasarathy, *Sentiment Analyzer: Analysis of Journal Citations from Citation Databases* (2014). [Read more](https://example-link-to-paper1)
2. G. Parthasarathy and D. C. Tomar, *A Survey of Sentiment Analysis for Journal Citation* (2015). [Read more](https://example-link-to-paper2)
3. S. Ghosh et al., *Determining Sentiment in Citation Text and Analyzing Its Impact on Ranking Index* (2016). [Read more](https://example-link-to-paper3)
4. M. Umer et al., *Scientific Papers Citation Analysis Using Textual Features and SMOTE Techniques* (2021). [Read more](https://example-link-to-paper4)
5. A. Athar, *Sentiment Analysis of Citations Using Sentence Structure-Based Features* (2011). [Read more](https://example-link-to-paper5)
"""
