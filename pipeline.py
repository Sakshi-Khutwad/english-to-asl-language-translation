import pandas as pd
from text_processing import clean_text, tokenize_text, text_to_sequences, add_special_tokens, build_vocabulary
from dataset_prep import prepare_dataset
import numpy as np


def preprocessing_pipeline():
    print('Loading and splitting dataset')
    dataset_splits = prepare_dataset()
    X_train, y_train = dataset_splits['train']
    X_val, y_val = dataset_splits['val']
    X_test, y_test = dataset_splits['test']

    # preparing the training data
    print('Preparing training data')
    print('Cleaning English texts')
    cleaned_english_texts = [clean_text(text) for text in X_train]
    clean_gloss = [str(text).strip() for text in y_train]

    print('Creating tokenizers')
    # english
    eng_tokenizer = tokenize_text(cleaned_english_texts, lower=True)
    # gloss
    gloss_with_tokens = add_special_tokens(clean_gloss)
    gloss_tokenizer = tokenize_text(gloss_with_tokens, lower=False)

    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    gloss_vocab_size = len(gloss_tokenizer.word_index) + 1

    print('Converting texts to sequences')
    # english
    encoder_input_sequences = text_to_sequences(eng_tokenizer, cleaned_english_texts)
    # gloss
    decoder_input_sequences = text_to_sequences(gloss_tokenizer, gloss_with_tokens)

    # create decoder target sequences
    decoder_target = np.zeros_like(decoder_input_sequences)
    decoder_target[:, :-1] = decoder_input_sequences[:, 1:]

    # preparing the testing data
    print('Preparing testing data')
    print('Cleaning English texts')
    cleaned_english_texts_test = [clean_text(text) for text in X_test]
    clean_gloss_test = [str(text).strip() for text in y_test]

    print('Creating tokenizers')
    # english
    eng_tokenizer_test = tokenize_text(cleaned_english_texts_test, lower=True)
    # gloss
    gloss_with_tokens_test = add_special_tokens(clean_gloss_test)
    gloss_tokenizer_test = tokenize_text(gloss_with_tokens_test, lower=False)

    # eng_vocab_size = len(eng_tokenizer_test.word_index) + 1
    # gloss_vocab_size = len(gloss_tokenizer_test.word_index) + 1

    print('Converting texts to sequences')
    # english
    encoder_input_sequences_test = text_to_sequences(eng_tokenizer_test, cleaned_english_texts_test)
    # gloss
    decoder_input_sequences_test = text_to_sequences(gloss_tokenizer_test, gloss_with_tokens_test)

    # create decoder target sequences
    decoder_target_test = np.zeros_like(decoder_input_sequences_test)
    decoder_target_test[:, :-1] = decoder_input_sequences_test[:, 1:]

    # preparing the validation data
    print('Preparing validation data')
    print('Cleaning English texts')
    cleaned_english_texts_val = [clean_text(text) for text in X_val]
    clean_gloss_val = [str(text).strip() for text in y_val]

    print('Creating tokenizers')
    # english
    eng_tokenizer_val = tokenize_text(cleaned_english_texts_val, lower=True)
    # gloss
    gloss_with_tokens_val = add_special_tokens(clean_gloss_val)
    gloss_tokenizer_val = tokenize_text(gloss_with_tokens_val, lower=False)

    # eng_vocab_size = len(eng_tokenizer_val.word_index) + 1
    # gloss_vocab_size = len(gloss_tokenizer_val.word_index) + 1

    print('Converting texts to sequences')
    # english
    encoder_input_sequences_val = text_to_sequences(eng_tokenizer_val, cleaned_english_texts_val)
    # gloss
    decoder_input_sequences_val = text_to_sequences(gloss_tokenizer_val, gloss_with_tokens_val)

    # create decoder target sequences
    decoder_target_val = np.zeros_like(decoder_input_sequences_val)
    decoder_target_val[:, :-1] = decoder_input_sequences_val[:, 1:]

    print('Data preprocessing completed')
    print(f'   Training examples: {len(encoder_input_sequences)}')
    print(f'   Validation examples: {len(encoder_input_sequences_val)}')
    print(f'   English vocab size: {eng_vocab_size}')
    print(f'   Gloss vocab size: {gloss_vocab_size}')

    return {
        # Training data
        'train_encoder_inputs': encoder_input_sequences,
        'train_decoder_inputs': decoder_input_sequences,
        'train_decoder_targets': decoder_target,

        # Validation data
        'val_encoder_inputs': encoder_input_sequences_val,
        'val_decoder_inputs': decoder_input_sequences_val,
        'val_decoder_targets': decoder_target_val,

        # Testing data
        'test_encoder_inputs': encoder_input_sequences_test,
        'test_decoder_inputs': decoder_input_sequences_test,
        'test_decoder_targets': decoder_target_test,

        # Tokenizers
        'eng_tokenizer': eng_tokenizer,
        'gloss_tokenizer': gloss_tokenizer,
        'eng_tokenizer_test': eng_tokenizer_test,
        'gloss_tokenizer_test': gloss_tokenizer_test,
        'eng_tokenizer_val': eng_tokenizer_val,
        'gloss_tokenizer_val': gloss_tokenizer_val,

        # Vocab sizes
        'eng_vocab_size': eng_vocab_size,
        'gloss_vocab_size': gloss_vocab_size,
        'max_length': 25,
        
        # raw test data
        'raw_test': (X_test, y_test)
    }



