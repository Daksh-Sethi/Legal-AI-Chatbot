# 🏛️ Legal AI System

This repository contains a **Legal AI System** designed to streamline legal document analysis, enhance legal research efficiency, and provide accurate answers to legal queries. The system utilizes state-of-the-art Natural Language Processing (NLP) models like **BERT** and **T5** for understanding, processing, and generating domain-specific legal insights.

---

## **Overview**
Legal professionals, judges, students, and even non-experts often find legal data overwhelming. The **Legal AI System** simplifies this by:
- Extracting insights from large legal documents.
- Providing precise answers to legal queries.
- Summarizing lengthy legal texts effectively.

**Key Use Cases**:
1. **For Legal Professionals**: Assisting in document analysis, contract review, and legal research.
2. **For Judiciary Members**: Simplifying precedent searches and document summaries.
3. **For Students**: Helping with legal education by providing quick insights into complex topics.

---

## **How It Works**
### **1. Initial Approach with BERT**
- We first used **BERT** (Bidirectional Encoder Representations from Transformers) for **Masked Language Modeling (MLM)** to:
  - Pre-train the model on 4GB of legal data.
  - Fine-tune BERT to predict masked tokens in legal texts and understand domain-specific jargon.

**Challenges with BERT**:
- BERT's input token limit (512 tokens) made it difficult to process long legal documents.
- Summarization and open-ended question answering tasks were limited in performance.

### **2. Shift to T5 for Question Answering**
- To overcome BERT's limitations, we transitioned to **T5 (Text-to-Text Transfer Transformer)**, which treats every NLP task as a text-to-text problem.
- **T5 Capabilities**:
  - Handles large legal documents by generating answers from summarized contexts.
  - Better suited for open-ended question answering and summarization tasks.
- **Training Pipeline**:
  1. Pre-trained T5 with MLM datasets.
  2. Fine-tuned T5 on a curated dataset of **15,000 question-answer pairs** from legal texts.

---

## **Files in the Repository**
**pynb file with the model training and evaluations with bert and t5 model both**
**app.py file and zip folder containing frontend files to deploy model using a website**
