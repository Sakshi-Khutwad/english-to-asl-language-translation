import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, Dropout, Concatenate, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

def build_regularized_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=128):
    """
    Improved model with attention, proper regularization, and dropout
    """
    print('Building REGULARIZED seq2seq model with Attention & Dropout')
    
    # Regularization parameters
    l2_lambda = 1e-4  # L2 regularization strength
    recurrent_l2_lambda = 1e-5  # Lighter regularization for recurrent weights
    dropout_rate = 0.3  # Dropout rate
    recurrent_dropout_rate = 0.2  # Recurrent dropout rate
    
    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    
    # Encoder embedding with regularization
    encoder_embedding = Embedding(
        eng_vocab_size, 
        embedding_dim, 
        mask_zero=False,
        embeddings_regularizer=l2(l2_lambda),
        name='encoder_embedding'
    )(encoder_inputs)
    
    # Add dropout after embedding
    encoder_embedding = Dropout(dropout_rate, name='encoder_embedding_dropout')(encoder_embedding)
    
    # Encoder LSTM with regularization
    encoder_lstm = Bidirectional(
        LSTM(128, 
             return_sequences=True, 
             return_state=True, 
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout_rate,
             kernel_regularizer=l2(l2_lambda),
             recurrent_regularizer=l2(recurrent_l2_lambda),
             bias_regularizer=l2(l2_lambda)),
        name='encoder_bilstm'
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    
    # Add dropout after encoder LSTM
    encoder_outputs = Dropout(dropout_rate, name='encoder_output_dropout')(encoder_outputs)
    
    # Encoder states for decoder initialization
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # ==================== DECODER WITH ATTENTION ====================
    decoder_inputs = Input(shape=(max_length,), name='decoder_input')
    
    # Decoder embedding with regularization
    decoder_embedding = Embedding(
        gloss_vocab_size, 
        embedding_dim, 
        mask_zero=False,
        embeddings_regularizer=l2(l2_lambda),
        name='decoder_embedding'
    )(decoder_inputs)
    
    # Add dropout after decoder embedding
    decoder_embedding = Dropout(dropout_rate, name='decoder_embedding_dropout')(decoder_embedding)
    
    # Decoder LSTM with regularization
    decoder_lstm = LSTM(
        256,  # 2*128 to match bidirectional encoder
        return_sequences=True, 
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        kernel_regularizer=l2(l2_lambda),
        recurrent_regularizer=l2(recurrent_l2_lambda),
        bias_regularizer=l2(l2_lambda),
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Add dropout after decoder LSTM
    decoder_outputs = Dropout(dropout_rate, name='decoder_output_dropout')(decoder_outputs)
    
    # ==================== ATTENTION MECHANISM ====================
    # Add attention between encoder outputs and decoder outputs
    attention = Attention(name='attention_layer')([decoder_outputs, encoder_outputs])
    
    # Concatenate decoder outputs with attention context
    decoder_combined = Concatenate(name='attention_concat')([decoder_outputs, attention])
    
    # Dense layer to combine attention context with regularization
    decoder_combined = Dense(256, 
                           activation='tanh',
                           kernel_regularizer=l2(l2_lambda),
                           bias_regularizer=l2(l2_lambda),
                           name='attention_dense')(decoder_combined)
    
    # Add dropout after attention dense layer
    decoder_combined = Dropout(dropout_rate, name='attention_dense_dropout')(decoder_combined)
    
    # Output layer with stronger regularization
    decoder_dense = TimeDistributed(
        Dense(gloss_vocab_size, 
              activation='softmax',
              kernel_regularizer=l2(1e-3),  # Stronger regularization on output
              bias_regularizer=l2(1e-3),
              name='output_dense')
    )
    decoder_outputs = decoder_dense(decoder_combined)
    
    # ==================== BUILD MODEL ====================
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='regularized_asl_seq2seq'
    )
    
    # Optimizer with good learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    print('✅ Regularized model with attention and dropout built successfully')
    
    # Print regularization summary
    print(f"\n📊 REGULARIZATION SUMMARY:")
    print(f"   L2 Lambda: {l2_lambda}")
    print(f"   Recurrent L2 Lambda: {recurrent_l2_lambda}") 
    print(f"   Dropout Rate: {dropout_rate}")
    print(f"   Recurrent Dropout Rate: {recurrent_dropout_rate}")
    
    return model

def create_regularized_inference_models(trained_model, max_length):
    """
    Create separate encoder and decoder models for proper inference
    """
    # Encoder inference model
    encoder_inputs = trained_model.input[0]
    encoder_outputs = trained_model.get_layer('encoder_bilstm').output[0]
    encoder_states = trained_model.get_layer('encoder_bilstm').output[1:]
    
    encoder_model = Model(encoder_inputs, [encoder_outputs] + list(encoder_states))
    
    # Decoder inference model
    decoder_inputs = trained_model.input[1]
    decoder_embedding_layer = trained_model.get_layer('decoder_embedding')
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    
    # Decoder states input
    decoder_state_input_h = Input(shape=(256,))
    decoder_state_input_c = Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Encoder outputs for attention
    encoder_outputs_input = Input(shape=(max_length, 256))  # 2*128
    
    # Decoder LSTM
    decoder_lstm = trained_model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    
    # Attention
    attention_layer = trained_model.get_layer('attention_layer')
    attention = attention_layer([decoder_outputs, encoder_outputs_input])
    
    # Concatenate
    concat_layer = trained_model.get_layer('attention_concat')
    decoder_combined = concat_layer([decoder_outputs, attention])
    
    # Attention dense layer
    attention_dense = trained_model.get_layer('attention_dense')
    decoder_combined = attention_dense(decoder_combined)
    
    # Output
    decoder_dense = trained_model.get_layer('output_dense')
    decoder_outputs = decoder_dense(decoder_combined)
    
    decoder_model = Model(
        [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    
    return encoder_model, decoder_model

def regularized_train_model(model, processed_data, epochs=25, batch_size=32):
    """
    Improved training with better callbacks, monitoring, and regularization
    """
    print('🚀 Starting REGULARIZED model training')
    
    # Prepare data
    X_train_encoder = processed_data['train_encoder_inputs']
    X_train_decoder = processed_data['train_decoder_inputs']
    y_train = processed_data['train_decoder_targets']
    
    X_val_encoder = processed_data['val_encoder_inputs']
    X_val_decoder = processed_data['val_decoder_inputs']
    y_val = processed_data['val_decoder_targets']
    
    # Reshape targets
    y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_val_reshaped = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
    
    print(f"📊 Training data: {X_train_encoder.shape[0]} samples")
    print(f"📊 Validation data: {X_val_encoder.shape[0]} samples")
    
    # Enhanced callbacks with regularization monitoring
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=12,  # Reduced patience for faster convergence
            restore_best_weights=True,
            verbose=1,
            min_delta=0.002
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,  # More aggressive learning rate reduction
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            'best_regularized_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Custom callback to monitor regularization effects
        RegularizationMonitor(processed_data),
    ]
    
    # REMOVED class_weight calculation - it's causing the error
    # For sequence models, class weights are complex and often not needed
    
    # Train without class weights
    history = model.fit(
        [X_train_encoder, X_train_decoder],
        y_train_reshaped,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            [X_val_encoder, X_val_decoder],
            y_val_reshaped
        ),
        callbacks=callbacks,
        shuffle=True,
        verbose=1
        # REMOVED: class_weight=class_weights
    )
    
    print(f"✅ Training completed at epoch {len(history.history['loss'])}")
    print(f"🎯 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Analyze training results
    analyze_training_results(history)
    
    return history

def analyze_training_results(history):
    """
    Analyze training results and provide recommendations
    """
    print("\n📈 TRAINING ANALYSIS:")
    print("=" * 50)
    
    if len(history.history['loss']) == 0:
        print("No training history available")
        return
    
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
    
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    
    # Check for overfitting
    if 'val_accuracy' in history.history:
        overfitting_gap = train_acc - val_acc
        if overfitting_gap > 0.1:
            print("⚠️  SIGNIFICANT OVERFITTING DETECTED")
            print("   Consider increasing regularization or dropout")
        elif overfitting_gap < 0.02:
            print("✅ Good generalization - train/val gap is small")
    
    # Check if model converged
    if val_acc > 0.7:
        print("🎉 EXCELLENT - Model achieved good accuracy!")
    elif val_acc > 0.6:
        print("✅ GOOD - Model is learning well")
    elif val_acc > 0.5:
        print("⚠️  FAIR - Model needs improvement")
    else:
        print("❌ POOR - Consider architectural changes")

class RegularizationMonitor(tf.keras.callbacks.Callback):
    """Custom callback to monitor regularization effects"""
    def __init__(self, processed_data):
        super().__init__()
        self.processed_data = processed_data
        self.eng_tokenizer = processed_data['eng_tokenizer']
        self.gloss_tokenizer = processed_data['gloss_tokenizer']
        self.reverse_eng = {v: k for k, v in self.eng_tokenizer.word_index.items()}
        self.reverse_gloss = {v: k for k, v in self.gloss_tokenizer.word_index.items()}
        self.best_val_acc = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"\n--- Epoch {epoch} Regularization Report ---")
            
            # Check train/val gap (overfitting indicator)
            train_acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            gap = train_acc - val_acc
            
            print(f"Train/Val Accuracy Gap: {gap:.4f}")
            
            if gap > 0.15:
                print("⚠️  High overfitting - consider increasing dropout")
            elif gap < 0.05:
                print("✅ Good generalization")
            
            # Update best accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"🎯 New best validation accuracy: {val_acc:.4f}")

def data_augmentation_with_regularization(processed_data):
    """
    Data augmentation with regularization considerations
    """
    print("🔄 Applying data augmentation for regularization...")
    
    train_encoder = processed_data['train_encoder_inputs']
    train_decoder = processed_data['train_decoder_inputs'] 
    train_targets = processed_data['train_decoder_targets']
    
    # For regularization, we want more diverse data
    if len(train_encoder) < 15000:  # If dataset is small
        val_encoder = processed_data['val_encoder_inputs']
        val_decoder = processed_data['val_decoder_inputs']
        val_targets = processed_data['val_decoder_targets']
        
        # Use more validation data for small datasets
        split_point = len(val_encoder) // 3
        augmented_encoder = np.concatenate([train_encoder, val_encoder[:split_point*2]])
        augmented_decoder = np.concatenate([train_decoder, val_decoder[:split_point*2]])
        augmented_targets = np.concatenate([train_targets, val_targets[:split_point*2]])
        
        print(f"📈 Augmented data: {len(augmented_encoder)} samples (from {len(train_encoder)})")
        
        return {
            'train_encoder_inputs': augmented_encoder,
            'train_decoder_inputs': augmented_decoder,
            'train_decoder_targets': augmented_targets,
            'val_encoder_inputs': val_encoder[split_point*2:],
            'val_decoder_inputs': val_decoder[split_point*2:],
            'val_decoder_targets': val_targets[split_point*2:],
        }
    
    return processed_data

class RegularizedASLTranslator:
    def __init__(self, model_path, processed_data):
        print("🚀 Loading REGULARIZED model...")
        self.trained_model = tf.keras.models.load_model(model_path)
        self.processed_data = processed_data
        
        self.eng_tokenizer = processed_data['eng_tokenizer']
        self.gloss_tokenizer = processed_data['gloss_tokenizer']
        self.max_length = processed_data['max_length']
        
        # Create inference models
        self.encoder_model, self.decoder_model = create_regularized_inference_models(
            self.trained_model, self.max_length
        )
        
        self.SOS_TOKEN = self.gloss_tokenizer.word_index.get('<start>', 1)
        self.EOS_TOKEN = self.gloss_tokenizer.word_index.get('<end>', 2)
        
        print("✅ Regularized inference models created successfully")
    
    def translate(self, english_sentence):
        """Proper inference with regularization-trained model"""
        from text_processing import clean_text, text_to_sequences
        
        # Preprocess
        cleaned = clean_text(english_sentence)
        encoder_input = text_to_sequences(self.eng_tokenizer, [cleaned], max_len=self.max_length)
        
        # Encode input
        encoder_outputs, state_h, state_c = self.encoder_model.predict(encoder_input, verbose=0)
        states_value = [state_h, state_c]
        
        # Generate empty target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.SOS_TOKEN
        
        # Sampling loop
        stop_condition = False
        decoded_sentence = []
        max_iterations = min(self.max_length, 20)  # Prevent infinite loops
        
        iteration = 0
        while not stop_condition and iteration < max_iterations:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_outputs] + states_value, verbose=0
            )
            
            # Sample token (use temperature=1.0 for deterministic)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.gloss_tokenizer.index_word.get(sampled_token_index, '<UNK>')
            
            # Stop conditions
            if (sampled_token_index == self.EOS_TOKEN or
                len(decoded_sentence) >= max_iterations):
                stop_condition = True
            elif sampled_word not in ['<start>', '<end>', '<pad>', '<UNK>']:
                decoded_sentence.append(sampled_word)
            
            # Update states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
            iteration += 1
        
        return ' '.join(decoded_sentence) if decoded_sentence else "<NO TRANSLATION>"

def retrain_with_regularization():
    """Complete retraining pipeline with regularization"""
    print("🔄 STARTING REGULARIZED RETRAINING PIPELINE")
    print("=" * 60)
    
    # Load your data
    from pipeline import preprocessing_pipeline
    processed_data = preprocessing_pipeline()
    
    # Apply data augmentation
    augmented_data = data_augmentation_with_regularization(processed_data)
    
    # Get vocab sizes
    eng_vocab_size = len(processed_data['eng_tokenizer'].word_index) + 1
    gloss_vocab_size = len(processed_data['gloss_tokenizer'].word_index) + 1
    max_length = processed_data['max_length']
    
    print(f"📊 Model Parameters:")
    print(f"   English vocab: {eng_vocab_size}")
    print(f"   Gloss vocab: {gloss_vocab_size}") 
    print(f"   Max length: {max_length}")
    
    # Build regularized model
    model = build_regularized_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length)
    
    # Train regularized model
    history = regularized_train_model(model, augmented_data, epochs=25, batch_size=32)
    
    # Save the model
    model.save('regularized_asl_model.h5')
    print("✅ Regularized model saved as 'regularized_asl_model.h5'")
    
    # Test the improved translator
    print("\n🧪 TESTING REGULARIZED TRANSLATOR")
    print("=" * 60)
    
    translator = RegularizedASLTranslator('regularized_asl_model.h5', processed_data)
    
    test_sentences = [
        "hello",
        "thank you",
        "what is your name", 
        "where is bathroom",
        "how are you",
        "good morning",
        "please help"
    ]
    
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"ASL:     {translation}")
        print("-" * 40)
    
    return model, history, translator

# Run the complete regularized retraining
if __name__ == "__main__":
    retrain_with_regularization()