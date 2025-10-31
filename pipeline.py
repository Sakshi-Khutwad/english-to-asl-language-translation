import pandas as pd
from text_processing import (
    clean_text, tokenize_text, text_to_sequences, add_special_tokens, normalize_asl_gloss, analyze_token_distribution,
    preprocess_asl_data
)
from dataset_prep import prepare_dataset
import numpy as np
from get_max_len import get_optimal_max_length, analyze_sequence_lengths


def preprocessing_pipeline():

    dataset_splits = prepare_dataset()

    X_train, y_train = dataset_splits['train']
    X_val, y_val = dataset_splits['val']
    X_test, y_test = dataset_splits['test']

    all_english_texts = X_train + X_val + X_test
    all_gloss_texts = y_train + y_val + y_test

    cleaned_all_english_texts, cleaned_all_gloss_texts = preprocess_asl_data(all_english_texts, all_gloss_texts)
    
    eng_lengths, gloss_lengths, all_lengths = analyze_sequence_lengths(cleaned_all_english_texts, cleaned_all_gloss_texts)
    max_len = get_optimal_max_length(cleaned_all_english_texts, cleaned_all_gloss_texts, coverage=0.95, buffer=2)

    eng_tokenizer = tokenize_text(cleaned_all_english_texts, lower=True)
    gloss_with_tokens = add_special_tokens(cleaned_all_gloss_texts)
    gloss_tokenizer = tokenize_text(gloss_with_tokens, lower=False)

    analyze_token_distribution(gloss_tokenizer, cleaned_all_gloss_texts)
    
    sample_gloss = cleaned_all_gloss_texts[:3]
    for i, text in enumerate(sample_gloss):
        tokens = gloss_tokenizer.texts_to_sequences([text])[0]
        token_words = [gloss_tokenizer.index_word.get(t, '?') for t in tokens if t != 0]
        print(f"  {i+1}. '{text}' -> {token_words}")

    
    train_eng_text = [clean_text(text) for text in X_train]
    encoder_input_train = text_to_sequences(eng_tokenizer, train_eng_text, max_len=max_len)

    val_eng_text = [clean_text(text) for text in X_val]
    encoder_input_val = text_to_sequences(eng_tokenizer, val_eng_text, max_len=max_len)

    test_eng_text = [clean_text(text) for text in X_test]
    encoder_input_test = text_to_sequences(eng_tokenizer, test_eng_text, max_len=max_len)

    
    train_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_train]
    train_gloss_with_tokens = add_special_tokens(train_gloss_text)  
    decoder_input_train = text_to_sequences(gloss_tokenizer, train_gloss_with_tokens, max_len=max_len)

    decoder_target_train = np.zeros_like(decoder_input_train)
    decoder_target_train[:, :-1] = decoder_input_train[:, 1:]  

    val_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_val]
    val_gloss_with_tokens = add_special_tokens(val_gloss_text) 
    decoder_input_val = text_to_sequences(gloss_tokenizer, val_gloss_with_tokens, max_len=max_len)
    
    decoder_target_val = np.zeros_like(decoder_input_val)
    decoder_target_val[:, :-1] = decoder_input_val[:, 1:] 

    test_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_test]
    test_gloss_with_tokens = add_special_tokens(test_gloss_text) 
    decoder_input_test = text_to_sequences(gloss_tokenizer, test_gloss_with_tokens, max_len=max_len)
    
    decoder_target_test = np.zeros_like(decoder_input_test)
    decoder_target_test[:, :-1] = decoder_input_test[:, 1:]
 
    desc_tokens = [word for word in gloss_tokenizer.word_index if word.startswith('desc_')]
    x_tokens = [word for word in gloss_tokenizer.word_index if word.startswith('x_')]
    print(f'Unique DESC tokens: {len(desc_tokens)}')
    print(f'Unique X tokens: {len(x_tokens)}')
    
    print(f'Max sequence length: {max_len}')
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Test samples: {len(X_test)}')

    print(f'Train encoder: {encoder_input_train.shape}')
    print(f'Train decoder: {decoder_input_train.shape}')
    print(f'Train targets: {decoder_target_train.shape}')
    print(f'Validation encoder: {encoder_input_val.shape}')
    print(f'Validation decoder: {decoder_input_val.shape}')

    return {
        'train_encoder_inputs': encoder_input_train,
        'train_decoder_inputs': decoder_input_train,
        'train_decoder_targets': decoder_target_train,

        'val_encoder_inputs': encoder_input_val,
        'val_decoder_inputs': decoder_input_val,
        'val_decoder_targets': decoder_target_val,

        'test_encoder_inputs': encoder_input_test,
        'test_decoder_inputs': decoder_input_test,
        'test_decoder_targets': decoder_target_test,

        'eng_tokenizer': eng_tokenizer,
        'gloss_tokenizer': gloss_tokenizer,

        'eng_vocab_size': len(eng_tokenizer.word_index) + 1,
        'gloss_vocab_size': len(gloss_tokenizer.word_index) + 1,
        'max_length': max_len,

        'raw_test': (X_test, y_test),
        'processed_test_gloss': test_gloss_text
    }

def analyze_data_balance(processed_data):
    gloss_tokenizer = processed_data['gloss_tokenizer']
    train_targets = processed_data['train_decoder_targets']
    
    all_tokens = train_targets.flatten()
    all_tokens = all_tokens[all_tokens != 0] 
    
    desc_count = 0
    x_count = 0
    other_count = 0
    
    for token_id in all_tokens:
        word = gloss_tokenizer.index_word.get(token_id, '')
        if word.startswith('desc_'):
            desc_count += 1
        elif word.startswith('x_'):
            x_count += 1
        else:
            other_count += 1
    
    total = len(all_tokens)
    return desc_count, x_count, other_count