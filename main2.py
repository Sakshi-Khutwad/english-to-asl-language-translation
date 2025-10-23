from pipeline import preprocessing_pipeline
from model2 import build_attention_bidirectional_model, train_model_with_regularization
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def main():
    print('Starting the ASL Translation Pipeline with Advanced Model')
    
    # Step 1: Preprocess data
    processed_data = preprocessing_pipeline()
    
    # Step 2: Build the advanced model
    print("\n" + "="*50)
    print("BUILDING ADVANCED MODEL WITH ATTENTION + BIDIRECTIONAL + REGULARIZATION")
    print("="*50)
    
    model, attention_weights = build_attention_bidirectional_model(
        eng_vocab_size=processed_data['eng_vocab_size'],
        gloss_vocab_size=processed_data['gloss_vocab_size'],
        max_length=processed_data['max_length'],
        embedding_dim=128
    )
    
    # Step 3: Train with anti-overfitting measures
    print("\n" + "="*50)
    print("TRAINING WITH REGULARIZATION TECHNIQUES")
    print("="*50)
    
    history = train_model_with_regularization(
        model=model,
        processed_data=processed_data,
        epochs=100,  # More epochs with early stopping
        batch_size=32,
        patience=10  # Early stopping patience
    )
    
    # Step 4: Save the model
    print("\n" + "="*50)
    print("SAVING TRAINED MODEL")
    print("="*50)
    
    model.save('asl_translation_model_advanced.h5')
    print('Model saved as: asl_translation_model_advanced.h5')
    
    # Step 5: Plot results
    plot_training_results(history)
    
    # Step 6: Evaluate on test set
    evaluate_model(model, processed_data)
    
    return model, processed_data, history

def plot_training_results(history):
    """Enhanced plotting with more metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss\n(Lower is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy\n(Higher is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss-Accuracy comparison
    plt.subplot(1, 3, 3)
    
    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_accuracy = history.history['val_accuracy'][best_epoch]
    
    plt.axhline(y=best_val_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Best Val Acc: {best_val_accuracy:.3f}')
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title(f'Best Model: Epoch {best_epoch}\nVal Loss: {best_val_loss:.3f}, Val Acc: {best_val_accuracy:.3f}')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n=== TRAINING RESULTS ===")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.1:
        print("✅ Good: Minimal overfitting detected!")
    elif overfitting_gap < 0.2:
        print("⚠️  Acceptable: Moderate overfitting")
    else:
        print("❌ High overfitting - consider more regularization")

def evaluate_model(model, processed_data):
    """Evaluate model on test set"""
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    # Prepare test data
    y_test_reshaped = processed_data['test_decoder_targets'].reshape(
        processed_data['test_decoder_targets'].shape[0],
        processed_data['test_decoder_targets'].shape[1],
        1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        [processed_data['test_encoder_inputs'], processed_data['test_decoder_inputs']],
        y_test_reshaped,
        verbose=1
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    generate_sample_predictions(model, processed_data)

def generate_sample_predictions(model, processed_data, num_samples=3):
    """Generate sample predictions for visualization"""
    
    eng_tokenizer = processed_data['eng_tokenizer']
    gloss_tokenizer = processed_data['gloss_tokenizer']
    
    # Get index to word mappings
    eng_index_to_word = {v: k for k, v in eng_tokenizer.word_index.items()}
    gloss_index_to_word = {v: k for k, v in gloss_tokenizer.word_index.items()}
    
    # Add special tokens to mappings
    eng_index_to_word[0] = '<PAD>'
    gloss_index_to_word[0] = '<PAD>'
    
    # Get test samples
    test_encoder_inputs = processed_data['test_encoder_inputs']
    test_decoder_inputs = processed_data['test_decoder_inputs']
    
    for i in range(min(num_samples, len(test_encoder_inputs))):
        print(f"\n--- Sample {i+1} ---")
        
        # Get input sequence
        input_seq = test_encoder_inputs[i]
        input_words = [eng_index_to_word[idx] for idx in input_seq if idx != 0 and idx in eng_index_to_word]
        input_sentence = ' '.join(input_words)
        print(f"Input: {input_sentence}")
        
        # Get target sequence
        target_seq = processed_data['test_decoder_targets'][i]
        target_words = [gloss_index_to_word[idx] for idx in target_seq if idx != 0 and idx in gloss_index_to_word]
        target_sentence = ' '.join(target_words)
        print(f"Target: {target_sentence}")
        
        # Generate prediction
        prediction = model.predict(
            [test_encoder_inputs[i:i+1], test_decoder_inputs[i:i+1]], 
            verbose=0
        )
        
        # Convert prediction to words
        predicted_indices = np.argmax(prediction[0], axis=-1)
        predicted_words = [gloss_index_to_word[idx] for idx in predicted_indices if idx != 0 and idx in gloss_index_to_word]
        predicted_sentence = ' '.join(predicted_words)
        print(f"Predicted: {predicted_sentence}")

if __name__ == '__main__':
    model, processed_data, history = main()