import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
from text_processing import clean_text, text_to_sequences
from pipeline import preprocessing_pipeline

class StreamlitASLTranslator:
    def __init__(self, model_path):
        st.write("Loading ASL Translation Model...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading model...")
        self.model = load_model(model_path)
        progress_bar.progress(30)

        self.processed_data = preprocessing_pipeline()
        progress_bar.progress(60)
        
        self.eng_tokenizer = self.processed_data['eng_tokenizer']
        self.gloss_tokenizer = self.processed_data['gloss_tokenizer']
        self.max_length = self.processed_data['max_length']
        
        self.SOS_TOKEN = self.gloss_tokenizer.word_index.get('<start>', 1)
        self.EOS_TOKEN = self.gloss_tokenizer.word_index.get('<end>', 2)
        self.UNK_TOKEN = self.gloss_tokenizer.word_index.get('<unk>', 0)
        
        self.reverse_eng = {v: k for k, v in self.eng_tokenizer.word_index.items()}
        self.reverse_gloss = {v: k for k, v in self.gloss_tokenizer.word_index.items()}
        
        progress_bar.progress(100)
        status_text.text("Model loaded successfully!")
        
        st.success(f"**Model Info:** SOS={self.SOS_TOKEN}, EOS={self.EOS_TOKEN}, Max Length={self.max_length}")
        st.info(f"**Vocabulary Sizes:** English={len(self.eng_tokenizer.word_index)}, Gloss={len(self.gloss_tokenizer.word_index)}")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

    def translate_with_visualization(self, english_sentence, max_output_length=15, min_confidence=0.05):
        
        st.write(f"### Translating: '{english_sentence}'")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            step_container = st.container()
            progress_bar = st.progress(0)
            
        with col2:
            metrics_container = st.container()
        
        try:
            cleaned = clean_text(english_sentence)
            encoder_input = text_to_sequences(self.eng_tokenizer, [cleaned], max_len=self.max_length)
            
            decoder_input = np.zeros((1, self.max_length))
            decoder_input[0, 0] = self.SOS_TOKEN
            
            predicted_tokens = []
            confidence_scores = []
            step_predictions = []
            
            for step in range(min(max_output_length, self.max_length)):
                progress = (step + 1) / min(max_output_length, self.max_length)
                progress_bar.progress(progress)
                
                predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
                current_probs = predictions[0, step, :]
                
                # Standard greedy search - just get the best token
                top_token = np.argmax(current_probs)
                top_prob = current_probs[top_token]
                top_word = self.reverse_gloss.get(top_token, '<UNK>')
                
                step_info = {
                    'step': step + 1,
                    'chosen_word': top_word,
                    'chosen_prob': top_prob
                }
                step_predictions.append(step_info)
                
                with step_container:
                    with st.expander(f"Step {step + 1}: '{top_word}' (Confidence: {top_prob:.4f})", expanded=True):
                        st.write(f"**Predicted:** {top_word}")
                        st.write(f"**Confidence:** {top_prob:.4f}")
                        st.write(f"**Token ID:** {top_token}")
                
                stop_conditions = [
                    top_token == self.EOS_TOKEN,
                    top_word in ['<end>', '<pad>'],
                    step >= max_output_length - 1,
                    top_prob < min_confidence and step > 0
                ]
                
                if any(stop_conditions):
                    if top_prob < min_confidence:
                        st.warning(f"Stopping at step {step + 1} due to low confidence ({top_prob:.4f})")
                    break
                
                if (top_prob >= min_confidence and 
                    top_token != self.SOS_TOKEN and 
                    top_word not in ['<start>', '<unk>'] and
                    not top_word.startswith('DESC-') and
                    not top_word.startswith('X-')):
                    
                    predicted_tokens.append(top_token)
                    confidence_scores.append(top_prob)
                    
                    if step + 1 < self.max_length:
                        decoder_input[0, step + 1] = top_token
            
            progress_bar.progress(1.0)
            
            words = []
            for token in predicted_tokens:
                word = self.reverse_gloss.get(token, '')
                if word and word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    words.append(word)
            
            result = ' '.join(words) if words else "<NO CONFIDENT PREDICTION>"
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            with metrics_container:
                st.metric("Average Confidence", f"{avg_confidence:.4f}")
                st.metric("Words Generated", len(words))
                st.metric("Translation Steps", len(step_predictions))
            
            st.markdown("---")
            if result != "<NO CONFIDENT PREDICTION>":
                st.success(f"### Final ASL Translation: **{result}**")
                st.info(f"**Confidence Score:** {avg_confidence:.4f}")
            else:
                st.error("### No confident translation generated")
                st.warning("Try rephrasing your sentence or check if the words are in the vocabulary.")
            
            return result, avg_confidence, step_predictions
            
        except Exception as e:
            st.error(f"### Translation failed: {str(e)}")
            return "<TRANSLATION ERROR>", 0, []

    def batch_translate(self, sentences, use_visualization=False):
        st.write("### Batch Translation")
        
        results = []
        for i, sentence in enumerate(sentences):
            st.write(f"**{i+1}. English:** {sentence}")
            
            if use_visualization:
                result, confidence, steps = self.translate_with_visualization(sentence)
            else:
                result, confidence, _ = self.translate_with_visualization(sentence)
                st.write(f"**ASL:** {result} (Confidence: {confidence:.4f})")
            
            results.append((sentence, result, confidence))
            st.write("---")
        
        return results

def main():
    st.set_page_config(
        page_title="ASL Neural Translator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">ASL Neural Translator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model Path", "best_regularized_model.h5")
        max_output_length = st.slider("Max Output Length", 5, 20, 15)
        min_confidence = st.slider("Minimum Confidence", 0.01, 0.3, 0.05, 0.01)
        
        st.header("Model Info")
        if st.button("Show Model Summary"):
            try:
                model = load_model(model_path)
                with st.expander("Model Architecture"):
                    st.text("Model summary will be displayed here")
            except:
                st.error("Could not load model")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please make sure the model file exists in the current directory.")
        return
    
    try:
        translator = StreamlitASLTranslator(model_path)
        
        st.markdown('<h2 class="sub-header">Translation Modes</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Single Translation", "Batch Translation", "Quick Test"])
        
        with tab1:
            st.markdown("### Single Sentence Translation")
            english_input = st.text_area(
                "Enter English sentence to translate:",
                placeholder="Type your English sentence here...",
                height=100
            )
            
            if st.button("Translate", key="single_translate"):
                if english_input.strip():
                    with st.spinner("Translating..."):
                        result, confidence, steps = translator.translate_with_visualization(
                            english_input, 
                            max_output_length, 
                            min_confidence
                        )
                else:
                    st.warning("Please enter a sentence to translate.")
        
        with tab2:
            st.markdown("### Batch Translation")
            batch_input = st.text_area(
                "Enter multiple sentences (one per line):",
                placeholder="hello\nthank you\nwhat is your name",
                height=150
            )
            
            show_viz = st.checkbox("Show step-by-step visualization", value=False)
            
            if st.button("Translate Batch", key="batch_translate"):
                if batch_input.strip():
                    sentences = [s.strip() for s in batch_input.split('\n') if s.strip()]
                    with st.spinner(f"Translating {len(sentences)} sentences..."):
                        results = translator.batch_translate(sentences, show_viz)
                else:
                    st.warning("Please enter some sentences to translate.")
        
        with tab3:
            st.markdown("### Quick Test Sentences")
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
            
            cols = st.columns(2)
            for i, sentence in enumerate(test_sentences):
                with cols[i % 2]:
                    if st.button(f"Translate: '{sentence}'", key=f"test_{i}"):
                        with st.spinner(f"Translating '{sentence}'..."):
                            result, confidence, steps = translator.translate_with_visualization(
                                sentence, 
                                max_output_length, 
                                min_confidence
                            )
    
    except Exception as e:
        st.error(f"Failed to initialize translator: {str(e)}")
        st.info("""
        **Please check:**
        1. Model file exists and is valid
        2. All required packages are installed  
        3. Preprocessing pipeline works correctly
        4. TensorFlow version is compatible
        """)

if __name__ == "__main__":
    main()