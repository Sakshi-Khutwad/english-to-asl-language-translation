import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\?\.\!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(texts, lower=True):
    tokenizer = Tokenizer(
        num_words = 10000,
        oov_token = '<UNK>',
        filters='',
        lower=lower
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer

def text_to_sequences(tokenizer, texts, max_len=25):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def add_special_tokens(texts):
    return [f'<START> {text} <END>' for text in texts]

def build_vocabulary(tokenizer):
    word_index = tokenizer.word_index
    index_word = {index: word for word, index in word_index.items()}
    vocab_size = len(word_index) + 1
    return word_index, index_word, vocab_size

def create_decoder_targets():
    pass

