import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Add your existing imports here
from text_processing import clean_text, text_to_sequences
from pipeline import preprocessing_pipeline

class FixedASLTranslator:
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
    
    def diagnose_model_issue(self):
        """Diagnose why the model is failing"""
        print("\n🔍 DIAGNOSING MODEL ISSUE")
        print("=" * 50)
        
        # Test on training data itself
        train_encoder = self.processed_data['train_encoder_inputs'][:2]
        train_decoder = self.processed_data['train_decoder_inputs'][:2]
        train_targets = self.processed_data['train_decoder_targets'][:2]
        
        for i in range(2):
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
            
            # Check if model can even predict training data
            pred_tokens = []
            for step in range(min(5, predictions.shape[1])):
                token = np.argmax(predictions[0, step, :])
                prob = predictions[0, step, token]
                word = self.reverse_gloss.get(token, f'?{token}?')
                if word not in ['<start>', '<end>', '<pad>']:
                    pred_tokens.append((word, prob))
            
            print(f"  Model prediction: {pred_tokens}")
            
            # Check if probabilities are reasonable
            avg_prob = np.mean([prob for _, prob in pred_tokens])
            print(f"  Average probability: {avg_prob:.4f}")
            
            if avg_prob < 0.1:
                print("  ❌ MODEL IS BROKEN - Probabilities too low!")
    
    def create_simple_translator(self):
        """Create a simple rule-based translator as fallback"""
        print("\n🔄 Creating simple rule-based translator...")
        
        # Common word mappings (you can expand this)
        word_mapping = {
            'hello': 'HELLO',
            'hi': 'HI', 
            'you': 'YOU',
            'thank': 'THANK',
            'thanks': 'THANK',
            'what': 'WHAT',
            'name': 'NAME',
            'how': 'HOW',
            'are': 'ARE',
            'where': 'WHERE',
            'bathroom': 'BATHROOM',
            'please': 'PLEASE',
            'sorry': 'SORRY',
            'good': 'GOOD',
            'morning': 'MORNING',
            'night': 'NIGHT'
        }
        
        def simple_translate(sentence):
            words = sentence.lower().split()
            translated = []
            for word in words:
                if word in word_mapping:
                    translated.append(word_mapping[word])
                else:
                    # Try to find similar words
                    found = False
                    for key, value in word_mapping.items():
                        if key in word or word in key:
                            translated.append(value)
                            found = True
                            break
                    if not found:
                        translated.append(word.upper())
            return ' '.join(translated)
        
        return simple_translate
    
    def translate_with_fallback(self, english_sentence):
        """
        Try neural translation first, fall back to rule-based if it fails
        """
        print(f"\n🎯 Translating: '{english_sentence}'")
        
        # First try neural translation
        neural_result = self._try_neural_translation(english_sentence)
        
        if neural_result and neural_result != "<NO OUTPUT>" and not all(word == '<UNK>' for word in neural_result.split()):
            print(f"🤖 Neural: {neural_result}")
            return neural_result
        else:
            # Fall back to rule-based
            simple_translator = self.create_simple_translator()
            rule_result = simple_translator(english_sentence)
            print(f"📝 Rule-based: {rule_result}")
            return rule_result
    
    def _try_neural_translation(self, english_sentence):
        """Try neural translation with aggressive filtering"""
        try:
            # Preprocess input
            cleaned = clean_text(english_sentence)
            encoder_input = text_to_sequences(self.eng_tokenizer, [cleaned], max_len=self.max_length)
            
            # Start with SOS
            decoder_input = np.zeros((1, self.max_length))
            decoder_input[0, 0] = self.SOS_TOKEN
            
            predicted_tokens = []
            
            for step in range(min(8, self.max_length)):  # Limit steps
                predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
                current_probs = predictions[0, step, :]
                
                # Get top 10 predictions
                top_tokens = np.argsort(current_probs)[-10:][::-1]
                
                chosen_token = None
                chosen_word = None
                chosen_prob = 0
                
                # Aggressive filtering
                for token in top_tokens:
                    word = self.reverse_gloss.get(token, '<UNK>')
                    prob = current_probs[token]
                    
                    # Only accept reasonable predictions
                    if (prob > 0.1 and  # Minimum confidence
                        token != self.EOS_TOKEN and
                        token != self.SOS_TOKEN and 
                        token != self.UNK_TOKEN and
                        word not in ['<start>', '<end>', '<pad>', '<unk>'] and
                        len(word) > 1 and  # Avoid single characters
                        not word.startswith('DESC-') and  # Avoid descriptor tokens
                        not word.startswith('X-')):  # Avoid pronoun markers
                        
                        chosen_token = token
                        chosen_word = word
                        chosen_prob = prob
                        break
                
                if chosen_token is None:
                    # If no good token, try less strict criteria
                    for token in top_tokens:
                        word = self.reverse_gloss.get(token, '<UNK>')
                        prob = current_probs[token]
                        
                        if (prob > 0.05 and
                            token != self.EOS_TOKEN and
                            word not in ['<start>', '<end>', '<pad>']):
                            
                            chosen_token = token
                            chosen_word = word
                            chosen_prob = prob
                            break
                
                if chosen_token is None:
                    break
                    
                print(f"  Step {step}: '{chosen_word}' (prob: {chosen_prob:.4f})")
                
                # Stop if we have reasonable output
                if len(predicted_tokens) >= 3:  # Don't generate too long
                    break
                    
                predicted_tokens.append(chosen_token)
                
                # Update decoder
                if step + 1 < self.max_length:
                    decoder_input[0, step + 1] = chosen_token
            
            # Convert to text
            words = []
            for token in predicted_tokens:
                word = self.reverse_gloss.get(token, '<UNK>')
                if word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    words.append(word)
            
            result = ' '.join(words) if words else "<NO OUTPUT>"
            return result
            
        except Exception as e:
            print(f"❌ Neural translation failed: {e}")
            return "<NO OUTPUT>"
    
    def interactive_translation(self):
        """Interactive translation with fallback"""
        print("\n" + "="*60)
        print("💬 INTERACTIVE TRANSLATION (with fallback)")
        print("="*60)
        print("Type 'quit' to exit, 'diagnose' to check model")
        
        while True:
            user_input = input("\n🌍 Enter English sentence: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'diagnose':
                self.diagnose_model_issue()
                continue
            elif not user_input:
                continue
            
            translation = self.translate_with_fallback(user_input)
            print(f"🖐️  ASL Gloss: {translation}")
    
    def batch_translate(self, sentences):
        """Translate multiple sentences"""
        print("\n" + "="*60)
        print("📝 BATCH TRANSLATION RESULTS")
        print("="*60)
        
        results = []
        for sentence in sentences:
            translation = self.translate_with_fallback(sentence)
            results.append((sentence, translation))
            print(f"English: {sentence}")
            print(f"ASL:     {translation}")
            print("-" * 40)
        
        return results

def main():
    """Main function to run everything"""
    
    # Model path - adjust if needed
    model_path = 'asl_translation_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the model file exists in the current directory.")
        return
    
    try:
        # Create translator
        translator = FixedASLTranslator(model_path)
        
        # First, diagnose the issue
        translator.diagnose_model_issue()
        
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
        translator.batch_translate(test_sentences)
        
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