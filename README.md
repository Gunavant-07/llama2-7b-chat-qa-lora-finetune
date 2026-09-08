# llama2-7b-chat-qa-lora-finetune
Fine-tuned Llama-2-7b-chat-hf on a Q&amp;A/instruction dataset using LoRA (PEFT). Model pushed to Hugging Face Hub.

# Llama-2-7B-Chat Fine-Tuned for Q&A / Instruction Following

This project fine-tunes Meta's `Llama-2-7b-chat-hf` model on a Q&A / instruction-following 
dataset sourced from Kaggle, using **LoRA (Low-Rank Adaptation)** via the Hugging Face PEFT library. 
The fine-tuned model has been pushed to the Hugging Face Hub for public use.

## 🔍 Overview
- **Base Model:** [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- **Fine-tuning Method:** LoRA / PEFT
- **Dataset Source:** Hugging Face Hub — [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
- **Task:** Question Answering / Instruction Following
- **Output:** Fine-tuned model hosted on Hugging Face — [your-hf-username/model-name]

📊 Dataset
Source: Hugging Face Hub — mlabonne/guanaco-llama2-1k

A 1,000-sample subset of the Guanaco dataset, reformatted into the Llama-2 chat prompt template (<s>[INST] ... [/INST] ... </s>).
Loaded directly in-notebook via datasets.load_dataset("mlabonne/guanaco-llama2-1k", split="train").
Used for lightweight LoRA fine-tuning of Llama-2-7b-chat-hf.

## ⚙️ Fine-Tuning Details
- **Base Model:** NousResearch/Llama-2-7b-chat-hf
- **Technique:** QLoRA (4-bit quantization + LoRA)
  - LoRA rank (r): 64
  - LoRA alpha: 16
  - LoRA dropout: 0.1
  - 4-bit quantization type: NF4, compute dtype: float16
- **Library:** Hugging Face `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes`, `accelerate`
- **Epochs:** 1
- **Batch size:** 1 (per device), gradient accumulation steps: 4 (effective batch size: 4)
- **Optimizer:** paged_adamw_8bit
- **Learning rate:** 2e-4 (cosine scheduler)
- **Hardware:** Kaggle Notebook — NVIDIA T4 GPU
- 
🚀 Workflow
Loaded base model NousResearch/Llama-2-7b-chat-hf in 4-bit precision (QLoRA) using BitsAndBytesConfig.
Loaded and tokenized the mlabonne/guanaco-llama2-1k dataset.
Fine-tuned using SFTTrainer (from trl) with a LoRA adapter (LoraConfig).
Saved the LoRA adapter and tokenizer locally.
Ran inference to sanity-check the fine-tuned model's responses.
Merged the LoRA adapter into the base model weights (merge_and_unload) to produce a standalone model.
Pushed both the LoRA adapter and the merged model to the Hugging Face Hub.
## 🚀 Model on Hugging Face
The fine-tuned model is available here: 
👉 [https://huggingface.co/Gunavant07/Llama-2-7b-chat-finetune-1](#)

## 📁 Repo Structure
├── finetune-llama2-demo_hf.ipynb   # Full fine-tuning notebook (Kaggle)
├── README.md
