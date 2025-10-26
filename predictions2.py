import numpy as np
from pipeline import preprocessing_pipeline
from tensorflow.keras.models import load_model

def beam_search_translate(model, english_sentence, processed_data, beam_width=5):
    """Beam search to find better sequences"""
    
    eng_tokenizer = processed_data['eng_tokenizer']
    gloss_tokenizer = processed_data['gloss_tokenizer']
    max_length = processed_data['max_length']
    
    SOS_TOKEN = 1
    EOS_TOKEN = 2
    
    # Preprocess input
    from text_processing import clean_text, text_to_sequences
    cleaned = clean_text(english_sentence)
    encoder_input = text_to_sequences(eng_tokenizer, [cleaned], max_len=max_length)
    
    # Initialize beams: (sequence, score)
    beams = [([SOS_TOKEN], 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            # Stop if sequence ended
            if sequence[-1] == EOS_TOKEN:
                new_beams.append((sequence, score))
                continue
            
            # Prepare decoder input
            decoder_input = np.zeros((1, max_length))
            for i, token in enumerate(sequence):
                if i < max_length:
                    decoder_input[0, i] = token
            
            # Get predictions
            predictions = model.predict([encoder_input, decoder_input], verbose=0)
            current_probs = predictions[0, len(sequence) - 1, :]
            
            # Get top beam_width candidates
            top_tokens = np.argsort(current_probs)[-beam_width:][::-1]
            
            for token in top_tokens:
                new_sequence = sequence + [token]
                # Use log probabilities to avoid underflow
                new_score = score + np.log(current_probs[token] + 1e-10)
                new_beams.append((new_sequence, new_score))
        
        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Stop if all beams ended
        if all(seq[-1] == EOS_TOKEN for seq, score in beams):
            break
    
    # Get best sequence
    best_sequence = beams[0][0]
    
    # Convert to words
    reverse_index = {v: k for k, v in gloss_tokenizer.word_index.items()}
    words = []
    for token in best_sequence[1:]:  # Skip SOS
        if token == EOS_TOKEN:
            break
        word = reverse_index.get(token, '<UNK>')
        if word not in ['<start>', '<end>', '<pad>']:
            words.append(word)
    
    return ' '.join(words) if words else '<NO OUTPUT>'

model = load_model('asl_translation_model.h5')
processed_data = preprocessing_pipeline()

# Test with beam search
test_words = ["hello", "you", "thank", "what"]
for word in test_words:
    translation = beam_search_translate(model, word, processed_data, beam_width=5)
    print(f"'{word}' → '{translation}'")