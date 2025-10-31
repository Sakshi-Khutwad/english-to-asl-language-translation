import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'desc-', 'desc_', text)  
    text = re.sub(r'x-', 'x_', text)        
    
    text = re.sub(r'[^a-zA-Z0-9\s\?\.\!_]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_asl_gloss(text):

    text = str(text).upper()

    text = re.sub(r'\bDESC[_-]', 'DESC_', text)
    text = re.sub(r'\bX[_-]', 'X_', text)
    
    text = re.sub(r'\bDESC\b', 'DESC_', text)
    text = re.sub(r'\bX\b', 'X_', text)
    
    return text

def tokenize_text(texts, lower=True):
    tokenizer = Tokenizer(
        num_words=10000,
        oov_token='<UNK>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', 
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
    desc_count = 0
    x_count = 0
    total_tokens = 0
    
    sequences = tokenizer.texts_to_sequences(texts)
    
    for seq in sequences:
        for token_id in seq:
            if token_id == 0:
                continue
            word = tokenizer.index_word.get(token_id, '')
            if word.startswith('desc_'):
                desc_count += 1
            elif word.startswith('x_'):
                x_count += 1
            total_tokens += 1

def preprocess_asl_data(english_texts, gloss_texts):
    
    cleaned_english = [clean_text(text) for text in english_texts]
    cleaned_gloss = [normalize_asl_gloss(clean_text(text)) for text in gloss_texts]
    
    print("Sample processed pairs:")
    for i in range(min(3, len(cleaned_english))):
        print(f"  English: {cleaned_english[i]}")
        print(f"  Gloss:   {cleaned_gloss[i]}")
        print()
    
    return cleaned_english, cleaned_gloss