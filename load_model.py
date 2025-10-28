import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model(model_path, processed_data):
   
    model = load_model(model_path)

    eng_tokenizer = processed_data['eng_tokenizer']
    gloss_tokenizer = processed_data['gloss_tokenizer']
    max_length = processed_data['max_length']

    return model, eng_tokenizer, gloss_tokenizer, max_length
