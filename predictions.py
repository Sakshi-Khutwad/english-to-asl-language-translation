from tensorflow.keras.models import load_model
import numpy as np
from pipeline import preprocessing_pipeline

def correct_translate(model, english_sentence, processed_data, max_output_length=20):
    """Properly implemented translation with step-by-step decoding"""
    
    eng_tokenizer = processed_data['eng_tokenizer']
    gloss_tokenizer = processed_data['gloss_tokenizer']
    max_length = processed_data['max_length']
    
    # Token mappings
    SOS_TOKEN = gloss_tokenizer.word_index.get('<start>', 1)
    EOS_TOKEN = gloss_tokenizer.word_index.get('<end>', 2)
    
    # Reverse index for converting tokens back to words
    reverse_gloss_index = {v: k for k, v in gloss_tokenizer.word_index.items()}
    
    # STEP 1: Preprocess input
    from text_processing import clean_text, text_to_sequences
    cleaned = clean_text(english_sentence)
    encoder_input = text_to_sequences(eng_tokenizer, [cleaned], max_len=max_length)
    
    print(f"📥 Input: '{english_sentence}'")
    print(f"🔢 Encoder input shape: {encoder_input.shape}")
    
    # STEP 2: Run encoder once to get states
    # For inference, we need to get encoder states
    # But if your model doesn't return states, we use teacher forcing trick
    
    # Create initial decoder input with SOS token
    decoder_input = np.zeros((1, max_length))
    decoder_input[0, 0] = SOS_TOKEN
    
    print(f"🎯 Starting decoding with SOS token: {SOS_TOKEN}")
    
    # STEP 3: Decode step by step
    predicted_tokens = []
    
    for time_step in range(min(max_output_length, max_length - 1)):
        # Get predictions for current time step
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        
        # Get predictions for the CURRENT time step only
        current_step_predictions = predictions[0, time_step, :]
        predicted_token = np.argmax(current_step_predictions)
        probability = current_step_predictions[predicted_token]
        
        # Convert to word for debugging
        predicted_word = reverse_gloss_index.get(predicted_token, f'<UNK:{predicted_token}>')
        
        print(f"   Step {time_step}: token={predicted_token} ('{predicted_word}'), prob={probability:.4f}")
        
        # Stop if EOS token is predicted
        if predicted_token == EOS_TOKEN:
            print(f"   🛑 Stopping at EOS token")
            break
        
        # Add to output (skip SOS token)
        if predicted_token != SOS_TOKEN:
            predicted_tokens.append(predicted_token)
        
        # Update decoder input for next time step
        if time_step + 1 < max_length:
            decoder_input[0, time_step + 1] = predicted_token
    
    # STEP 4: Convert tokens to text
    gloss_words = []
    for token in predicted_tokens:
        word = reverse_gloss_index.get(token, '<UNK>')
        if word not in ['<start>', '<end>', '<pad>']:
            gloss_words.append(word)
    
    final_translation = ' '.join(gloss_words) if gloss_words else '<NO OUTPUT>'
    
    print(f"📤 Final: {final_translation}")
    print(f"🔢 Raw tokens: {predicted_tokens}")
    
    return final_translation

# Test it
model = load_model('asl_translation_model.h5')
processed_data = preprocessing_pipeline()

test_sentences = ["What is your name", "I eat apple", "thank you"]

for sentence in test_sentences:
    print("=" * 50)
    translation = correct_translate(model, sentence, processed_data)
    print(f"English: '{sentence}' → ASL: '{translation}'")
    print()