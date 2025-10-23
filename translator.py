import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

class ASLTranslator:
    def __init__(self, model_path, eng_tokenizer, gloss_tokenizer, max_len):
        self.model = load_model(model_path)
        self.eng_tokenizer = eng_tokenizer
        self.gloss_tokenizer = gloss_tokenizer
        self.max_len = max_len

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s\?\.\!]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def translate_sentence(self, eng_sentence):
        clean_sentences = self.clean_text(eng_sentence)
        sequence = self.eng_tokenizer.texts_to_sequences([clean_sentences])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')

        prediction = self.model.predict([padded, padded], verbose=0)
        predicted_indices = np.argmax(prediction, axis=-1)[0]

        gloss_words = []
        for idx in predicted_indices:
            if idx > 0 and idx in self.gloss_tokenizer.index_word:
                word = self.gloss_tokenizer.index_word[idx]
                if word not in ['<START>', '<END>']:
                    gloss_words.append(word)
            if idx in self.gloss_tokenizer.index_word and self.gloss_tokenizer.index_word[idx] == '<END>':
                break

        return ' '.join(gloss_words) if gloss_words else 'UNKNOWN'
    
    def evaluate_on_test_set(self, X_test, y_test, num_samples=20):
        
        print('Model Evaluation')

        correct = 0
        total = min(num_samples, len(X_test))

        for i in range(total):
            english = X_test[i]
            expected_gloss = y_test[i]
            predicted_gloss = self.translate_sentence(english)

            is_correct = (predicted_gloss == expected_gloss)
            if is_correct:
                correct += 1
                status = '✅'
            else:
                status = '❌'
            
            print('English: ', english)
            print('Expected gloss: ', expected_gloss)
            print('Predicted Gloss: ', predicted_gloss)
            print()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.1%} ({correct}/{total})')

        return accuracy
    
    # def interactive_translation(self):
    #     """Interactive mode for translating custom sentences"""
    #     print("\n💬 INTERACTIVE TRANSLATION MODE")
    #     print("Type 'quit' to exit\n")
        
    #     while True:
    #         user_input = input("Enter English sentence: ").strip()
    #         if user_input.lower() == 'quit':
    #             break
    #         if user_input:
    #             translation = self.translate_sentence(user_input)
    #             print(f"🔤 ASL Gloss: {translation}\n")

def main():
    from pipeline import preprocessing_pipeline
    from dataset_prep import prepare_dataset

    # dataset_splits = prepare_dataset()
    # X_test, y_test = dataset_splits['test']

    processed_data = preprocessing_pipeline()
    X_test, y_test = processed_data['X_test_raw'], processed_data['y_test_raw'] 

    translator = ASLTranslator(
        model_path='asl_translation_model.h5',
        eng_tokenizer=processed_data['eng_tokenizer'],
        gloss_tokenizer=processed_data['gloss_tokenizer'],
        max_len=processed_data['max_length']
    )

    accuracy = translator.evaluate_on_test_set(X_test, y_test, num_samples=20)

    print('Accuracy on test set: ', accuracy)
    # if accuracy < 0.8:
    #     translator.interactive_translation()
    # else:
    #     print('Model needs more training')

if __name__ == "__main__":
    main()