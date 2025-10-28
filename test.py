import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import numpy as np
from pipeline import preprocessing_pipeline
import tensorflow as tf

def translate_sentence(model, encoder_input, eng_tokenizer, gloss_tokenizer, max_length):
    """Translate a single sentence using greedy decoding"""
    # Start with SOS token
    start_token = gloss_tokenizer.word_index.get('<start>', 1)
    end_token = gloss_tokenizer.word_index.get('<end>', 2)
    
    decoder_input = np.zeros((1, max_length))
    decoder_input[0, 0] = start_token
    
    translated_tokens = []
    
    for step in range(max_length - 1):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        next_token = np.argmax(predictions[0, step, :])
        
        if next_token == end_token:
            break
            
        translated_tokens.append(next_token)
        decoder_input[0, step + 1] = next_token
    
    return translated_tokens

def calculate_bleu_score(model, test_data, eng_tokenizer, gloss_tokenizer, max_length):
    """Calculate BLEU score for the entire test set"""
    
    # Get reverse tokenizers for decoding
    reverse_gloss = {v: k for k, v in gloss_tokenizer.word_index.items()}
    
    references = []  # List of reference translations (each is list of words)
    hypotheses = []  # List of hypothesis translations
    
    print("🔍 Calculating BLEU score...")
    
    for i in range(len(test_data['val_encoder_inputs'])):
        if i % 100 == 0:
            print(f"Processing {i}/{len(test_data['val_encoder_inputs'])}")
        
        # Get encoder input
        encoder_input = test_data['val_encoder_inputs'][i:i+1]
        
        # Get reference translation (convert tokens to words)
        reference_tokens = test_data['val_decoder_targets'][i]
        reference_words = []
        for token in reference_tokens:
            if token == 0:  # Skip padding
                continue
            word = reverse_gloss.get(token, '')
            if word and word not in ['<start>', '<end>', '<pad>']:
                reference_words.append(word)
        
        if reference_words:  # Only add if we have valid reference
            references.append([reference_words])  # BLEU expects list of lists
            
            # Get model translation
            predicted_tokens = translate_sentence(model, encoder_input, eng_tokenizer, gloss_tokenizer, max_length)
            predicted_words = [reverse_gloss.get(token, '') for token in predicted_tokens]
            predicted_words = [word for word in predicted_words if word and word not in ['<start>', '<end>', '<pad>']]
            
            hypotheses.append(predicted_words)
    
    # Calculate BLEU scores
    smoothie = SmoothingFunction().method4
    
    print("\n📊 BLEU Score Results:")
    print("=" * 40)
    
    # Individual BLEU scores
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if len(hyp) > 0:
            bleu1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu2 = sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            bleu3 = sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
    
    # Corpus BLEU scores
    corpus_bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    corpus_bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    corpus_bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    corpus_bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    print(f"Corpus BLEU-1: {corpus_bleu1:.4f}")
    print(f"Corpus BLEU-2: {corpus_bleu2:.4f}")
    print(f"Corpus BLEU-3: {corpus_bleu3:.4f}")
    print(f"Corpus BLEU-4: {corpus_bleu4:.4f}")
    
    if bleu1_scores:
        print(f"\nAverage Sentence BLEU-1: {np.mean(bleu1_scores):.4f}")
        print(f"Average Sentence BLEU-2: {np.mean(bleu2_scores):.4f}")
        print(f"Average Sentence BLEU-3: {np.mean(bleu3_scores):.4f}")
        print(f"Average Sentence BLEU-4: {np.mean(bleu4_scores):.4f}")
    
    return {
        'corpus_bleu1': corpus_bleu1,
        'corpus_bleu2': corpus_bleu2,
        'corpus_bleu3': corpus_bleu3,
        'corpus_bleu4': corpus_bleu4,
        'references': references,
        'hypotheses': hypotheses
    }

def evaluate_with_beam_search(model, test_data, eng_tokenizer, gloss_tokenizer, max_length, beam_width=3):
    """Evaluate with beam search for better translations"""
    
    def beam_search_translate(encoder_input, beam_width=3, max_length=20):
        start_token = gloss_tokenizer.word_index.get('<start>', 1)
        end_token = gloss_tokenizer.word_index.get('<end>', 2)
        
        # Initialize beams: (tokens, score)
        beams = [([start_token], 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for tokens, score in beams:
                if tokens[-1] == end_token or len(tokens) >= max_length:
                    new_beams.append((tokens, score))
                    continue
                
                # Prepare decoder input
                decoder_input = np.zeros((1, max_length))
                for i, token in enumerate(tokens):
                    if i < max_length:
                        decoder_input[0, i] = token
                
                # Get predictions
                predictions = model.predict([encoder_input, decoder_input], verbose=0)
                step_probs = predictions[0, len(tokens) - 1, :]
                
                # Get top beam_width candidates
                top_tokens = np.argsort(step_probs)[-beam_width:][::-1]
                
                for token in top_tokens:
                    if token in [start_token, end_token]:
                        continue
                    
                    token_prob = step_probs[token]
                    new_score = score + np.log(token_prob + 1e-8)
                    new_tokens = tokens + [token]
                    
                    new_beams.append((new_tokens, new_score))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop if all beams have EOS
            if all(seq[-1] == end_token for seq, _ in beams):
                break
        
        # Return best sequence (without start token)
        best_tokens, best_score = beams[0]
        return [token for token in best_tokens if token not in [start_token, end_token]]
    
    # Calculate BLEU with beam search
    reverse_gloss = {v: k for k, v in gloss_tokenizer.word_index.items()}
    references = []
    hypotheses = []
    
    print("🔍 Calculating BLEU with Beam Search...")
    
    for i in range(min(100, len(test_data['val_encoder_inputs']))):  # Test on first 100 for speed
        encoder_input = test_data['val_encoder_inputs'][i:i+1]
        
        # Reference
        reference_tokens = test_data['val_decoder_targets'][i]
        reference_words = []
        for token in reference_tokens:
            if token == 0:
                continue
            word = reverse_gloss.get(token, '')
            if word and word not in ['<start>', '<end>', '<pad>']:
                reference_words.append(word)
        
        if reference_words:
            references.append([reference_words])
            
            # Hypothesis with beam search
            predicted_tokens = beam_search_translate(encoder_input, beam_width, max_length)
            predicted_words = [reverse_gloss.get(token, '') for token in predicted_tokens]
            predicted_words = [word for word in predicted_words if word and word not in ['<start>', '<end>', '<pad>']]
            
            hypotheses.append(predicted_words)
    
    # Calculate BLEU
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    
    print(f"Beam Search BLEU-4: {bleu4:.4f}")
    
    return bleu4, references, hypotheses

def print_sample_translations(results, num_samples=5):
    """Print sample translations for manual inspection"""
    print(f"\n🔍 Sample Translations (first {num_samples}):")
    print("=" * 60)
    
    for i in range(min(num_samples, len(results['references']))):
        ref = ' '.join(results['references'][i][0])
        hyp = ' '.join(results['hypotheses'][i])
        
        print(f"Sample {i+1}:")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {hyp}")
        print(f"  BLEU-1: {sentence_bleu(results['references'][i], results['hypotheses'][i], weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4):.4f}")
        print("-" * 40)

# Usage in your main code:
def evaluate_model_bleu(model_path, processed_data):
    """Main function to evaluate your trained model with BLEU"""
    
    # Load model and tokenizers
    model = tf.keras.models.load_model(model_path)
    eng_tokenizer = processed_data['eng_tokenizer']
    gloss_tokenizer = processed_data['gloss_tokenizer']
    max_length = processed_data['max_length']
    
    print("🚀 Starting BLEU Evaluation")
    print("=" * 50)
    
    # Calculate BLEU scores
    results = calculate_bleu_score(model, processed_data, eng_tokenizer, gloss_tokenizer, max_length)
    
    # Optional: Beam search evaluation
    beam_bleu, _, _ = evaluate_with_beam_search(model, processed_data, eng_tokenizer, gloss_tokenizer, max_length)
    print(f"\n🌟 Beam Search (width=3) BLEU-4: {beam_bleu:.4f}")
    
    # Print sample translations
    print_sample_translations(results)
    
    return results

# Add this to your main training script after training:
processed_data = preprocessing_pipeline()
results = evaluate_model_bleu('regularized_asl_model.h5', processed_data)