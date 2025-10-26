import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Add your existing imports here
from text_processing import clean_text, text_to_sequences
from pipeline import preprocessing_pipeline

class NeuralASLTranslator:
    def __init__(self, model_path):
        print("🚀 Loading model and data...")
        self.model = load_model(model_path)
        self.processed_data = preprocessing_pipeline()
        
        self.eng_tokenizer = self.processed_data['eng_tokenizer']
        self.gloss_tokenizer = self.processed_data['gloss_tokenizer']
        self.max_length = self.processed_data['max_length']
        
        # Get token mappings
        self.SOS_TOKEN = self.gloss_tokenizer.word_index.get('<start>', 1)
        self.EOS_TOKEN = self.gloss_tokenizer.word_index.get('<end>', 2)
        self.UNK_TOKEN = self.gloss_tokenizer.word_index.get('<unk>', 0)
        
        print(f"✅ Loaded: SOS={self.SOS_TOKEN}, EOS={self.EOS_TOKEN}, MaxLen={self.max_length}")
        print(f"📚 Vocab sizes: Eng={len(self.eng_tokenizer.word_index)}, Gloss={len(self.gloss_tokenizer.word_index)}")
        
        # Build word mapping dictionaries
        self.reverse_eng = {v: k for k, v in self.eng_tokenizer.word_index.items()}
        self.reverse_gloss = {v: k for k, v in self.gloss_tokenizer.word_index.items()}
    
    def analyze_model_performance(self):
        """Analyze model performance on training data"""
        print("\n🔍 ANALYZING MODEL PERFORMANCE")
        print("=" * 50)
        
        # Test on training data itself
        train_encoder = self.processed_data['train_encoder_inputs'][:3]
        train_decoder = self.processed_data['train_decoder_inputs'][:3]
        train_targets = self.processed_data['train_decoder_targets'][:3]
        
        for i in range(3):
            print(f"\nTraining Example {i}:")
            
            # Show actual training pair
            eng_tokens = [self.reverse_eng.get(t, '?') for t in train_encoder[i] if t != 0]
            gloss_tokens = [self.reverse_gloss.get(t, '?') for t in train_targets[i] if t != 0]
            
            print(f"  Input:  {' '.join(eng_tokens)}")
            print(f"  Target: {' '.join(gloss_tokens)}")
            
            # Get model prediction on training data
            encoder_input = train_encoder[i:i+1]
            decoder_input = train_decoder[i:i+1]
            
            predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
            
            # Analyze prediction quality
            pred_tokens = []
            confidence_scores = []
            
            for step in range(min(8, predictions.shape[1])):
                token = np.argmax(predictions[0, step, :])
                prob = predictions[0, step, token]
                word = self.reverse_gloss.get(token, f'?{token}?')
                
                if word not in ['<start>', '<end>', '<pad>']:
                    pred_tokens.append(word)
                    confidence_scores.append(prob)
            
            print(f"  Model prediction: {' '.join(pred_tokens)}")
            print(f"  Confidence: {np.mean(confidence_scores):.4f}")
    
    def translate(self, english_sentence, max_output_length=10, min_confidence=0.1):
        """
        Pure neural translation without any rule-based fallback
        """
        print(f"\n🎯 Translating: '{english_sentence}'")
        
        try:
            # Preprocess input
            cleaned = clean_text(english_sentence)
            encoder_input = text_to_sequences(self.eng_tokenizer, [cleaned], max_len=self.max_length)
            
            # Initialize decoder with SOS token
            decoder_input = np.zeros((1, self.max_length))
            decoder_input[0, 0] = self.SOS_TOKEN
            
            predicted_tokens = []
            confidence_scores = []
            
            for step in range(min(max_output_length, self.max_length)):
                predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
                current_probs = predictions[0, step, :]
                
                # Get top prediction
                top_token = np.argmax(current_probs)
                top_prob = current_probs[top_token]
                top_word = self.reverse_gloss.get(top_token, '<UNK>')
                
                # Stop conditions
                if (top_token == self.EOS_TOKEN or 
                    top_word in ['<end>', '<pad>'] or
                    step >= max_output_length - 1):
                    break
                
                # Only add if confidence is sufficient and not special tokens
                if (top_prob >= min_confidence and 
                    top_token != self.SOS_TOKEN and 
                    top_word not in ['<start>', '<unk>'] and
                    not top_word.startswith('DESC-') and
                    not top_word.startswith('X-')):
                    
                    predicted_tokens.append(top_token)
                    confidence_scores.append(top_prob)
                    
                    print(f"  Step {step}: '{top_word}' (prob: {top_prob:.4f})")
                    
                    # Update decoder input for next step
                    if step + 1 < self.max_length:
                        decoder_input[0, step + 1] = top_token
                
                elif top_prob < min_confidence:
                    print(f"  Step {step}: Low confidence ({top_prob:.4f}), stopping")
                    break
            
            # Convert tokens to words
            words = []
            for token in predicted_tokens:
                word = self.reverse_gloss.get(token, '')
                if word and word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    words.append(word)
            
            result = ' '.join(words) if words else "<NO CONFIDENT PREDICTION>"
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            print(f"🤖 Neural Translation: {result}")
            print(f"📊 Average Confidence: {avg_confidence:.4f}")
            
            return result, avg_confidence
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return "<TRANSLATION ERROR>", 0
    
    def translate_with_beam_search(self, english_sentence, beam_width=3, max_length=8):
        """
        Advanced translation with beam search for better results
        """
        print(f"\n🎯 Beam Search Translation: '{english_sentence}'")
        
        try:
            # Preprocess input
            cleaned = clean_text(english_sentence)
            encoder_input = text_to_sequences(self.eng_tokenizer, [cleaned], max_len=self.max_length)
            
            # Initialize beam search
            beams = [([self.SOS_TOKEN], 1.0)]  # (tokens, cumulative_prob)
            
            for step in range(max_length):
                new_beams = []
                
                for tokens, cum_prob in beams:
                    # Skip if sequence is complete
                    if tokens and tokens[-1] == self.EOS_TOKEN:
                        new_beams.append((tokens, cum_prob))
                        continue
                    
                    # Prepare decoder input
                    decoder_input = np.zeros((1, self.max_length))
                    for i, token in enumerate(tokens):
                        if i < self.max_length:
                            decoder_input[0, i] = token
                    
                    # Get predictions
                    predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
                    step_probs = predictions[0, len(tokens) - 1, :]
                    
                    # Get top beam_width candidates
                    top_tokens = np.argsort(step_probs)[-beam_width:][::-1]
                    
                    for token in top_tokens:
                        prob = step_probs[token]
                        word = self.reverse_gloss.get(token, '<UNK>')
                        
                        # Skip unwanted tokens
                        if (word in ['<start>', '<pad>', '<unk>'] or
                            word.startswith('DESC-') or word.startswith('X-')):
                            continue
                        
                        new_tokens = tokens + [token]
                        new_prob = cum_prob * prob
                        
                        new_beams.append((new_tokens, new_prob))
                
                # Keep top beam_width beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Stop if all beams end with EOS
                if all(tokens[-1] == self.EOS_TOKEN for tokens, _ in beams):
                    break
            
            # Convert best beam to text
            best_tokens, best_prob = beams[0]
            words = []
            
            for token in best_tokens:
                word = self.reverse_gloss.get(token, '')
                if word and word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    words.append(word)
            
            result = ' '.join(words) if words else "<NO VALID TRANSLATION>"
            
            print(f"🤖 Beam Search Result: {result}")
            print(f"📊 Sequence Probability: {best_prob:.6f}")
            
            return result, best_prob
            
        except Exception as e:
            print(f"❌ Beam search failed: {e}")
            return "<BEAM SEARCH ERROR>", 0
    
    def interactive_translation(self):
        """Interactive translation using only neural model"""
        print("\n" + "="*60)
        print("💬 PURE NEURAL TRANSLATION")
        print("="*60)
        print("Type 'quit' to exit, 'analyze' to check model, 'beam' for beam search")
        
        while True:
            user_input = input("\n🌍 Enter English sentence: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'analyze':
                self.analyze_model_performance()
                continue
            elif user_input.lower() == 'beam':
                sentence = input("Enter sentence for beam search: ").strip()
                if sentence:
                    self.translate_with_beam_search(sentence)
                continue
            elif not user_input:
                continue
            
            # Try both standard and beam search
            print("\n--- Standard Translation ---")
            standard_result, std_conf = self.translate(user_input)
            
            print("\n--- Beam Search Translation ---")
            beam_result, beam_prob = self.translate_with_beam_search(user_input)
            
            print(f"\n🎯 Final Results:")
            print(f"Standard: {standard_result} (conf: {std_conf:.4f})")
            print(f"Beam:     {beam_result} (prob: {beam_prob:.6f})")
    
    def batch_translate(self, sentences, use_beam_search=False):
        """Translate multiple sentences using neural model only"""
        print("\n" + "="*60)
        print("📝 NEURAL BATCH TRANSLATION RESULTS")
        print("="*60)
        
        results = []
        for sentence in sentences:
            if use_beam_search:
                translation, confidence = self.translate_with_beam_search(sentence)
            else:
                translation, confidence = self.translate(sentence)
            
            results.append((sentence, translation, confidence))
            print(f"English: {sentence}")
            print(f"ASL:     {translation}")
            print(f"Conf:    {confidence:.4f}")
            print("-" * 40)
        
        return results

def main():
    """Main function to run neural translation"""
    
    # Model path - adjust if needed
    model_path = 'regularized_asl_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the model file exists in the current directory.")
        return
    
    try:
        # Create translator
        translator = NeuralASLTranslator(model_path)
        
        # First, analyze model performance
        translator.analyze_model_performance()
        
        # Test sentences
        test_sentences = [
            "hello",
            "thank you", 
            "what is your name",
            "where is bathroom",
            "how are you",
            "good morning",
            "please help",
            "sorry"
        ]
        
        # Run batch translation
        print("\n" + "="*50)
        print("STANDARD TRANSLATION")
        print("="*50)
        translator.batch_translate(test_sentences, use_beam_search=False)
        
        print("\n" + "="*50)
        print("BEAM SEARCH TRANSLATION") 
        print("="*50)
        translator.batch_translate(test_sentences, use_beam_search=True)
        
        # Start interactive mode
        translator.interactive_translation()
    
    except Exception as e:
        print(f"❌ Failed to initialize translator: {e}")
        print("Please check:")
        print("1. Model file exists and is valid")
        print("2. All required packages are installed")
        print("3. Preprocessing pipeline works correctly")

if __name__ == "__main__":
    main()