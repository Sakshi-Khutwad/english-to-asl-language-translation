import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """Enhanced cleaning for ASL gloss notation"""
    text = text.lower()
    
    # Preserve special ASL markers but normalize them
    text = re.sub(r'desc-', 'desc_', text)  # Replace hyphen with underscore
    text = re.sub(r'x-', 'x_', text)        # Replace hyphen with underscore
    
    # Clean the rest
    text = re.sub(r'[^a-zA-Z0-9\s\?\.\!_]', '', text)  # Allow underscores
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_asl_gloss(text):
    """Normalize ASL-specific notations"""
    # Standardize DESC and X markers
    text = re.sub(r'\bdesc[_-]', 'desc_', text)
    text = re.sub(r'\bx[_-]', 'x_', text)
    
    # Handle common variations
    text = re.sub(r'\bdesc\b', 'desc_', text)  # Standalone 'desc'
    text = re.sub(r'\bx\b', 'x_', text)        # Standalone 'x'
    
    return text

def tokenize_text(texts, lower=True):
    """Enhanced tokenizer for ASL glosses"""
    tokenizer = Tokenizer(
        num_words=10000,
        oov_token='<UNK>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',  # Keep underscores
        lower=lower,
        split=' '
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer

def text_to_sequences(tokenizer, texts, max_len=25):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def add_special_tokens(texts):
    return [f'<start> {text} <end>' for text in texts]

def build_vocabulary(tokenizer):
    word_index = tokenizer.word_index
    index_word = {index: word for word, index in word_index.items()}
    vocab_size = len(word_index) + 1
    return word_index, index_word, vocab_size

def analyze_token_distribution(tokenizer, texts):
    """Analyze how well DESC and X tokens are learned"""
    desc_count = 0
    x_count = 0
    total_tokens = 0
    
    sequences = tokenizer.texts_to_sequences(texts)
    
    for seq in sequences:
        for token_id in seq:
            if token_id == 0:  # padding
                continue
            word = tokenizer.index_word.get(token_id, '')
            if word.startswith('desc_'):
                desc_count += 1
            elif word.startswith('x_'):
                x_count += 1
            total_tokens += 1
    
    print(f"📊 Token Distribution Analysis:")
    print(f"   DESC_ tokens: {desc_count} ({desc_count/max(total_tokens,1)*100:.1f}%)")
    print(f"   X_ tokens: {x_count} ({x_count/max(total_tokens,1)*100:.1f}%)")
    print(f"   Total tokens: {total_tokens}")

def preprocess_asl_data(english_texts, gloss_texts):
    """Complete preprocessing pipeline for ASL data"""
    
    # Clean and normalize
    cleaned_english = [clean_text(text) for text in english_texts]
    cleaned_gloss = [normalize_asl_gloss(clean_text(text)) for text in gloss_texts]
    
    print("Sample processed pairs:")
    for i in range(min(3, len(cleaned_english))):
        print(f"  English: {cleaned_english[i]}")
        print(f"  Gloss:   {cleaned_gloss[i]}")
        print()
    
    return cleaned_english, cleaned_gloss