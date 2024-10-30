import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from adapter_transformers import AdapterConfig  # Import from adapter_transformers

# Load Model and Tokenizer Paths
model_path = "finetuned_model/finetuned-bert-lora-news-sentiment"
tokenizer_path = "finetuned_model/finetuned-bert-lora-tokenizer"

# Load Model
def load_model(model_path, tokenizer_path):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load model with adapters
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    # Load adapter configuration and adapter
    adapter_config = AdapterConfig.load("pfeiffer")
    model.load_adapter(model_path, config=adapter_config)
    
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# Predict function and rest of Streamlit code remains the same...
