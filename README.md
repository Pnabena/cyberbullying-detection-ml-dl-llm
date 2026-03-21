# cyberbullying-detection-ml-dl-llm.github
Cyberbullying Detection using Machine Learning, Hybrid Deep Learning, and LLM Fine-Tuning

Raw Tweets
    ↓
Text Cleaning & Preprocessing
    ↓
Tokenization / Embeddings
    ↓
---------------------------------------
| ML Models | CNN-LSTM | DistilBERT | TinyLlama |
---------------------------------------
    ↓
Evaluation (Accuracy, F1)
    ↓
Fairness & Bias Analysis

This project investigates cyberbullying detection as a **multi-class natural language processing task**, comparing traditional machine learning, hybrid deep learning, and large language model (LLM) approaches.

The study focuses on how different models capture **contextual and nuanced forms of harmful language** across multiple categories, while also examining **fairness and bias in classification performance**.

---

## Research Motivation

Online platforms face increasing challenges in detecting subtle and context-dependent cyberbullying. Traditional approaches often fail to capture implicit or context-rich abusive language.

This project explores whether:
- Deep learning improves contextual understanding
- LLMs outperform traditional models in nuanced classification
- Model performance varies across different bullying categories (fairness)

---

## Dataset

- **Source:** Kaggle Cyberbullying Twitter Dataset  
- **Link:** https://www.kaggle.com/datasets/ashiqnazir/cbtweets  
- **Type:** Multi-class text dataset  

### Class Labels:
- Age
- Gender
- Religion
- Ethnicity
- Not Cyberbullying
- Other Cyberbullying

---

## Methodology

### 1. Baseline Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

### 2. Deep Learning Model
- Hybrid **CNN-LSTM architecture**
  - CNN for feature extraction
  - LSTM for sequence modelling
    
### Transformer Model
- DistilBERT (fine-tuned for classification)

### 3. Large Language Model
- Fine-tuned **TinyLlama**
  - Used for contextual classification of cyberbullying types

---

## Workflow

1. Data cleaning and preprocessing  
2. Exploratory data analysis (EDA)  
3. Text vectorisation and feature extraction  
4. Baseline model training  
5. CNN-LSTM training  
6. TinyLlama fine-tuning  
7. Model evaluation  
8. Fairness and bias analysis  

---

## Results

*(Replace with your actual values)*

| Model | Accuracy | F1-score (Weighted) |
|------|----------|---------------------|
| Logistic Regression | 0.81 | 0.81 |
| SVM | 0.81 | 0.81 |
| Random Forest | 0.81 | 0.81 |
| CNN-LSTM | ~0.82 | ~0.82 |
| DistilBERT | 0.87 | 0.87 |
| TinyLlama | 0.61 | 0.61 |

DistilBERT achieved the best overall performance, demonstrating strong contextual understanding and significantly outperforming both traditional machine learning and hybrid deep learning models.

## Per-Class Performance Insights

- **High-performing classes:** Age, Ethnicity, Religion (F1 ≈ 0.97–0.99)
  
- **Moderate:** Gender (F1 ≈ 0.90)
  
- **Challenging classes:**
  - Not Cyberbullying → F1 = 0.66
  - Other Cyberbullying → F1 = 0.74

These results indicate that models perform best when clear lexical patterns exist, but struggle with ambiguous or context-dependent classifications.

---

## Key Findings

- Traditional ML models provide strong baselines but struggle with **context-dependent bullying**
- DistilBERT significantly outperformed all other models, demonstrating strong contextual understanding of cyberbullying language
- CNN-LSTM improves **sequence awareness and contextual understanding**
- TinyLlama achieved moderate overall performance (~0.61), but failed entirely on certain classes, demonstrating that aggregate metrics can mask critical classification failures
- Performance varies across bullying categories, highlighting **fairness concerns in classification**

---

## Fairness & Bias Analysis

This project evaluates how models perform across different bullying types, revealing:

DistilBERT achieved near-perfect performance on categories with explicit linguistic patterns (e.g., age, religion, ethnicity), while performance dropped significantly for more ambiguous classes such as "Not Cyberbullying" (F1 = 0.66) and "Other Cyberbullying" (F1 = 0.74).

- Certain categories (e.g., implicit bullying) are harder to detect
- TinyLlama failed to detect certain classes (F1 = 0.00 for "Other Cyberbullying" and "Not Cyberbullying"), indicating poor generalisation across the full label space.

---

## Repository Structure

notebooks/ → Model development and experiments
src/ → Reusable scripts
models/ → Trained models
results/ → Evaluation outputs and visualisations
data/ → Dataset references and preprocessing
docs/ → Project summary or research notes

---

## Technologies Used

- Python  
- pandas, numpy  
- scikit-learn  
- TensorFlow / PyTorch *(update based on your notebook)*  
- HuggingFace Transformers  
- matplotlib / seaborn  

---

## Future Work

- Extend to **multilingual cyberbullying detection**
- Explore **explainability for LLM predictions**
- Apply methods to **real-world conversational systems**
- Improve fairness-aware training techniques

---

## Research Direction

This project forms part of a broader research interest in:

> **Natural Language Processing for understanding complex, real-world, and culturally nuanced language across domains such as social media, healthcare, and human interaction systems.**
