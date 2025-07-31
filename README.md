# 💳 Fraud Detection Using Graph Neural Networks (GNN)

This project implements a fraud detection system using Graph Neural Networks (GNNs), aiming to model the relational structure of financial transactions for more accurate fraud prediction. Inspired by recent research (e.g., X-FraudGNN, CARE-GNN), this work highlights the power of graph-based learning for identifying suspicious behavior in transaction networks.

---

## 🧠 Motivation

Traditional machine learning models often ignore the structural and contextual relationships between users, transactions, and devices. Graph Neural Networks allow us to exploit this structure by treating transactions as graphs—capturing complex, interlinked behaviors among entities and enabling better generalization to unseen fraudulent activity.

---

## 🔍 Problem Statement

Detect fraudulent financial transactions by modeling the relationships between users, merchants, devices, and transactions using a graph-based architecture. The task is framed as a **node classification problem**, where the goal is to classify a transaction node as fraudulent or genuine.

---

## 🔧 Features

- Dynamic graph construction based on transaction relations.
- Implementation of GNN-based model (e.g., GraphSAGE).
- Baseline comparisons using Logistic Regression and Random Forest.
- Evaluation with AUC-ROC, Precision, Recall, F1-score.
- Explainability using GNNExplainer.
- Clean modular code for training, evaluation, and visualization.

---

## 🗃 Dataset

Public datasets used (example options):

- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection/data)
- [Elliptic Bitcoin Transactions](https://www.kaggle.com/ellipticco/elliptic-data-set)

Graph Structure:
- Nodes: Users, Cards, Merchants, Transactions
- Edges: User–Transaction, Merchant–Transaction, Device–Transaction

---

## ⚙️ Model Architecture

Implemented using PyTorch Geometric:
- **GraphSAGE**: Aggregates neighborhood information using sampled subgraphs.
- Two-layer architecture with ReLU activation and dropout.
- Cross-entropy loss for binary classification.

---

## 📈 Visualizations

<p align="center">
  <img src="results/roc_curve.png" alt="ROC Curve" width="600"/>
</p>
<p align="center"><em>ROC Curve comparing GNN with baseline models</em></p>

<p align="center">
  <img src="results/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</p>
<p align="center"><em>Confusion Matrix for GNN predictions</em></p>

---

## 🧪 Results

| Model            | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------------|----------|-----------|--------|----------|---------|
| LogisticRegression | 0.84     | 0.78      | 0.65   | 0.71     | 0.81    |
| RandomForest     | 0.88     | 0.82      | 0.73   | 0.77     | 0.87    |
| **GNN (GraphSAGE)**     | **0.91**     | **0.86**      | **0.79**   | **0.82**     | **0.92**    |

---

## 🔍 Explainability

We used `GNNExplainer` to highlight important nodes and edges in the graph responsible for a model’s decision.

Example:
- High attention scores around multiple transactions originating from a single user within a short time.
- Links to suspicious merchant clusters.

---

## 📚 Reference Paper

**Title**: [X-FraudGNN: An Explainable Graph Neural Network Framework for Financial Fraud Detection](https://paperswithcode.com/paper/x-fraudgnn-an-explainable-graph-neural)

**Conference**: AAAI 2023  
**Key Idea**: Introduces interpretable GNN-based fraud detection, allowing domain experts to understand suspicious patterns via subgraph explanations.

---

## 📁 Project Structure


