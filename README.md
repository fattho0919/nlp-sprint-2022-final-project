# NLP Final Project - Advertising Regulation Classification (2022 Spring)

This repository contains the implementation and results for a Natural Language Processing final project (teamwork) focusing on advertising regulation classification and illegal advertisement detection tasks.

## Project Background

Consumers are exposed to various forms of advertising on a daily basis, including television, websites, flyers, videos, and more. Advertising has become the quickest way for consumers to learn about the features of a new product. However, the issue of whether advertising involves exaggeration or deception is often a headache for consumers.

This project addresses the legal basis for determining illegal advertising (判定非法廣告的法律依據) through computational linguistics approaches.

## Project Structure

### Task 1: Classification of Regulations for Unlawful Ads Based on Government Notice
**Location:** `Task1/`
**(政府公告的違規廣告之法條分類)**

- **Objective:** Text classification of advertising regulations using transformer-based models (BLOOM, LLaMA)
- **Dataset:** COS domain advertising regulation classification
- **Approach:** Fine-tuning pre-trained language models with LoRA (Low-Rank Adaptation) to classify government notices of unlawful advertisements

**Key Components:**
- `src/model_training/`: Core model implementation
  - `model.py`: Main model architecture with LLaMA tokenizer
  - `model_BLOOM.py`: BLOOM-based implementation for multilingual advertisement classification
  - `train_POS.py`: Part-of-speech training utilities for legal text processing
- `src/chatgpt_api_calling/`: ChatGPT API integration for legal data preprocessing
- `data/`: Training and test datasets for regulation classification (COS_train.csv, COS_test.csv)
- `results/`: Model output files with different experiment versions for regulation classification

### Task 2: Detection of Illegal Parts of Ads and Provision of Legal Basis
**Location:** `Task2/`
**(廣告非法部分的偵測與法律依據提供)**

- **Objective:** Prompt design and evaluation for detecting illegal advertisement content and providing legal basis
- **Approach:** Few-shot learning with carefully designed prompts for MED and COS advertisement analysis

**Key Components:**
- `main.py`: Core processing script for MED dataset sampling and illegal content detection
- `json_extract.py`: JSON data extraction utilities for legal document processing
- `prompt_design.json`: Structured prompt templates for illegal advertisement detection
- `COS_prompt_*.txt`: COS domain prompts for advertisement regulation (3 variants)
- `MED_output`: Medical advertisement domain processed outputs
- Reports: `COS_report.pdf`, `MED_report.pdf` - Analysis of illegal advertisement detection results

## Technical Implementation

### Models Used
- **LLaMA-7B**: Chinese language model with LoRA fine-tuning for legal text classification
- **BLOOM**: Multilingual language model for cross-lingual advertisement analysis
- **ChatGPT API**: For legal data preprocessing and advertisement content evaluation

### Key Technologies
- PyTorch for model training and legal text processing
- Transformers library (Hugging Face) for pre-trained model fine-tuning
- PEFT (Parameter Efficient Fine-Tuning) with LoRA for efficient adaptation to legal domain
- TensorBoard for training monitoring and performance analysis

## Results
The project includes multiple experiment runs with different configurations for advertising regulation tasks:
- Task 1: Various model versions for regulation classification (0531_v1, 0602_v1, 0602_v5, 0603_v1-v3)
- Task 2: Comparative analysis of prompt designs for illegal advertisement detection across domains

This teamwork project demonstrates proficiency in applying modern NLP techniques to legal domain problems, including transformer fine-tuning for regulation classification, prompt engineering for illegal content detection, and cross-domain advertisement analysis.