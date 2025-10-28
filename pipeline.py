import pandas as pd
from text_processing import (
    clean_text, tokenize_text, text_to_sequences, add_special_tokens, 
    build_vocabulary, normalize_asl_gloss, analyze_token_distribution,
    preprocess_asl_data
)
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

    # Enhanced preprocessing for ASL data
    print("🔄 Preprocessing ASL data with enhanced normalization...")
    cleaned_all_english_texts, cleaned_all_gloss_texts = preprocess_asl_data(all_english_texts, all_gloss_texts)
    
    # Analyze sequence lengths with cleaned data
    eng_lengths, gloss_lengths, all_lengths = analyze_sequence_lengths(cleaned_all_english_texts, cleaned_all_gloss_texts)
    max_len = get_optimal_max_length(cleaned_all_english_texts, cleaned_all_gloss_texts, coverage=0.95, buffer=2)

    # Tokenize with enhanced processing
    eng_tokenizer = tokenize_text(cleaned_all_english_texts, lower=True)
    gloss_with_tokens = add_special_tokens(cleaned_all_gloss_texts)
    gloss_tokenizer = tokenize_text(gloss_with_tokens, lower=False)

    # Analyze token distribution for DESC/X patterns
    print("\n📊 Analyzing ASL token patterns...")
    analyze_token_distribution(gloss_tokenizer, cleaned_all_gloss_texts)
    # desc_patterns, x_patterns = check_training_data_balance(cleaned_all_gloss_texts)
    
    # Print sample tokenization
    print("\n🔍 Sample Tokenization:")
    sample_gloss = cleaned_all_gloss_texts[:3]
    for i, text in enumerate(sample_gloss):
        tokens = gloss_tokenizer.texts_to_sequences([text])[0]
        token_words = [gloss_tokenizer.index_word.get(t, '?') for t in tokens if t != 0]
        print(f"  {i+1}. '{text}' -> {token_words}")

    # English text processing
    
    # Training data
    train_eng_text = [clean_text(text) for text in X_train]
    encoder_input_train = text_to_sequences(eng_tokenizer, train_eng_text, max_len=max_len)

    # Validation data
    val_eng_text = [clean_text(text) for text in X_val]
    encoder_input_val = text_to_sequences(eng_tokenizer, val_eng_text, max_len=max_len)

    # Test data
    test_eng_text = [clean_text(text) for text in X_test]
    encoder_input_test = text_to_sequences(eng_tokenizer, test_eng_text, max_len=max_len)

    # Gloss text processing with enhanced normalization
    
    # Training data
    train_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_train]
    train_gloss_with_tokens = add_special_tokens(train_gloss_text)  
    decoder_input_train = text_to_sequences(gloss_tokenizer, train_gloss_with_tokens, max_len=max_len)

    decoder_target_train = np.zeros_like(decoder_input_train)
    decoder_target_train[:, :-1] = decoder_input_train[:, 1:]  

    # Validation data  
    val_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_val]
    val_gloss_with_tokens = add_special_tokens(val_gloss_text) 
    decoder_input_val = text_to_sequences(gloss_tokenizer, val_gloss_with_tokens, max_len=max_len)
    
    decoder_target_val = np.zeros_like(decoder_input_val)
    decoder_target_val[:, :-1] = decoder_input_val[:, 1:] 

    # Test data
    test_gloss_text = [normalize_asl_gloss(clean_text(str(text))) for text in y_test]
    test_gloss_with_tokens = add_special_tokens(test_gloss_text) 
    decoder_input_test = text_to_sequences(gloss_tokenizer, test_gloss_with_tokens, max_len=max_len)
    
    decoder_target_test = np.zeros_like(decoder_input_test)
    decoder_target_test[:, :-1] = decoder_input_test[:, 1:]

    # Enhanced vocabulary analysis
    print('\n📚 Vocabulary Analysis:')
    print(f'English Vocabulary size: {len(eng_tokenizer.word_index) + 1}')
    print(f'Gloss Vocabulary size: {len(gloss_tokenizer.word_index) + 1}')
    
    # Count DESC and X tokens in vocabulary
    desc_tokens = [word for word in gloss_tokenizer.word_index if word.startswith('desc_')]
    x_tokens = [word for word in gloss_tokenizer.word_index if word.startswith('x_')]
    print(f'Unique DESC tokens: {len(desc_tokens)}')
    print(f'Unique X tokens: {len(x_tokens)}')
    
    print(f'Max sequence length: {max_len}')
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Test samples: {len(X_test)}')

    # Verify data shapes
    print('\n✅ Data Shapes:')
    print(f'Train encoder: {encoder_input_train.shape}')
    print(f'Train decoder: {decoder_input_train.shape}')
    print(f'Train targets: {decoder_target_train.shape}')
    print(f'Validation encoder: {encoder_input_val.shape}')
    print(f'Validation decoder: {decoder_input_val.shape}')

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
        
        # Pattern analysis
        # 'desc_patterns': desc_patterns,
        # 'x_patterns': x_patterns,
        
        # Raw data for reference
        'raw_test': (X_test, y_test),
        'processed_test_gloss': test_gloss_text
    }

def analyze_data_balance(processed_data):
    """Additional analysis function for data balance"""
    gloss_tokenizer = processed_data['gloss_tokenizer']
    train_targets = processed_data['train_decoder_targets']
    
    # Flatten all target tokens
    all_tokens = train_targets.flatten()
    all_tokens = all_tokens[all_tokens != 0]  # Remove padding
    
    # Count DESC vs X tokens
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
    print(f"\n🎯 Training Data Token Distribution:")
    print(f"   DESC tokens: {desc_count} ({desc_count/total*100:.1f}%)")
    print(f"   X tokens: {x_count} ({x_count/total*100:.1f}%)")
    print(f"   Other tokens: {other_count} ({other_count/total*100:.1f}%)")
    
    return desc_count, x_count, other_count

# Optional: Data augmentation for balancing
# def augment_rare_patterns(processed_data, min_count=10):
    # """Augment data for rare DESC/X patterns"""
    # gloss_tokenizer = processed_data['gloss_tokenizer']
    # train_targets = processed_data['train_decoder_targets']
    
    # # Identify rare patterns
    # token_counts = {}
    # for token_id in train_targets.flatten():
    #     if token_id != 0:
    #         token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    # rare_desc_tokens = []
    # for token_id, count in token_counts.items():
    #     word = gloss_tokenizer.index_word.get(token_id, '')
    #     if word.startswith('desc_') and count < min_count:
    #         rare_desc_tokens.append(token_id)
    
    # print(f"🔍 Found {len(rare_desc_tokens)} rare DESC tokens")
    # return rare_desc_tokens