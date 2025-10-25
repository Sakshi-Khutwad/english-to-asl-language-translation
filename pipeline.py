import pandas as pd
from text_processing import clean_text, tokenize_text, text_to_sequences, add_special_tokens, build_vocabulary
from dataset_prep import prepare_dataset
import numpy as np
from get_max_len import get_optimal_max_length, analyze_sequence_lengths


def preprocessing_pipeline():

    print('Loading and splitting dataset')
    dataset_splits = prepare_dataset()

    X_train, y_train = dataset_splits['train']
    X_val, y_val = dataset_splits['val']
    X_test, y_test = dataset_splits['test']

    all_english_texts = X_train + X_val + X_test
    all_gloss_texts = y_train + y_val + y_test

    eng_lengths, gloss_lengths, all_lengths = analyze_sequence_lengths(all_english_texts, all_gloss_texts)
    max_len = get_optimal_max_length(all_english_texts, all_gloss_texts, coverage=0.95, buffer=2)

    cleaned_all_english_texts = [clean_text(text) for text in all_english_texts]
    clean_all_gloss = [str(text).strip() for text in all_gloss_texts]

    eng_tokenizer = tokenize_text(cleaned_all_english_texts, lower=True)
    gloss_with_tokens = add_special_tokens(clean_all_gloss)
    gloss_tokenizer = tokenize_text(gloss_with_tokens, lower=False)

    # english text processing
    
    # training data
    train_eng_text = [clean_text(text) for text in X_train]
    encoder_input_train = text_to_sequences(eng_tokenizer, train_eng_text, max_len=max_len)

    # validation data
    val_eng_text = [clean_text(text) for text in X_val]
    encoder_input_val = text_to_sequences(eng_tokenizer, val_eng_text, max_len=max_len)

    # test data
    test_eng_text = [clean_text(text) for text in X_test]
    encoder_input_test = text_to_sequences(eng_tokenizer, test_eng_text, max_len=max_len)

    # gloss text processing
    
    # training data
    train_gloss_text = [str(text).strip() for text in y_train]
    train_gloss_with_tokens = add_special_tokens(train_gloss_text)  
    decoder_input_train = text_to_sequences(gloss_tokenizer, train_gloss_with_tokens, max_len=max_len)

    decoder_target_train = np.zeros_like(decoder_input_train)
    decoder_target_train[:, :-1] = decoder_input_train[:, 1:]  

    # Validation data  
    val_gloss_text = [str(text).strip() for text in y_val]
    val_gloss_with_tokens = add_special_tokens(val_gloss_text) 
    decoder_input_val = text_to_sequences(gloss_tokenizer, val_gloss_with_tokens, max_len=max_len)
    
    decoder_target_val = np.zeros_like(decoder_input_val)
    decoder_target_val[:, :-1] = decoder_input_val[:, 1:] 

    # Test data
    test_gloss_text = [str(text).strip() for text in y_test]
    test_gloss_with_tokens = add_special_tokens(test_gloss_text) 
    decoder_input_test = text_to_sequences(gloss_tokenizer, test_gloss_with_tokens, max_len=max_len)
    
    decoder_target_test = np.zeros_like(decoder_input_test)
    decoder_target_test[:, :-1] = decoder_input_test[:, 1:]

    print('Eng Vocabulary size: ', len(eng_tokenizer.word_index) + 1)
    print('Gloss Vocabulary size: ', len(gloss_tokenizer.word_index) + 1)
    print('Max len: ', max_len)
    
    print('Train dataset length: ', len(X_train))
    print('Validation dataset length: ', len(X_val))
    print('Test dataset length: ', len(X_test))

    return {
        # Training data
        'train_encoder_inputs': encoder_input_train,
        'train_decoder_inputs': decoder_input_train,
        'train_decoder_targets': decoder_target_train,

        # Validation data
        'val_encoder_inputs': encoder_input_val,
        'val_decoder_inputs': decoder_input_val,
        'val_decoder_targets': decoder_target_val,

        # Test data
        'test_encoder_inputs': encoder_input_test,
        'test_decoder_inputs': decoder_input_test,
        'test_decoder_targets': decoder_target_test,

        # Tokenizers
        'eng_tokenizer': eng_tokenizer,
        'gloss_tokenizer': gloss_tokenizer,

        # Vocab sizes
        'eng_vocab_size': len(eng_tokenizer.word_index) + 1,
        'gloss_vocab_size': len(gloss_tokenizer.word_index) + 1,
        'max_length': max_len,
        
        # Raw data for reference
        'raw_test': (X_test, y_test)
    }    



