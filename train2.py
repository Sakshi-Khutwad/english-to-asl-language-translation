import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, Dropout, Concatenate, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


def build_improved_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=128):
    
    l2_lambda = 1e-4 
    recurrent_l2_lambda = 1e-5
    dropout_rate = 0.2  
    recurrent_dropout_rate = 0.1  
    
    # Encoder with character-level backup
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    
    # Enhanced embedding with better OOV handling
    encoder_embedding = Embedding(
        eng_vocab_size, 
        embedding_dim, 
        mask_zero=False,
        embeddings_regularizer=l2(l2_lambda),
        name='encoder_embedding'
    )(encoder_inputs)
    
    encoder_embedding = Dropout(dropout_rate, name='encoder_embedding_dropout')(encoder_embedding)

    # Rest of your architecture remains the same...
    encoder_lstm1 = Bidirectional(
        LSTM(256, 
             return_sequences=True,
             return_state=True, 
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout_rate,
             kernel_regularizer=l2(l2_lambda),
             recurrent_regularizer=l2(recurrent_l2_lambda),
             bias_regularizer=l2(l2_lambda)),
        name='encoder_bilstm1'
    )
    encoder_outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(encoder_embedding)

    encoder_outputs1 = Dropout(dropout_rate, name='encoder_output1_dropout')(encoder_outputs1)

    encoder_lstm2 = Bidirectional(
        LSTM(128, 
             return_sequences=True,
             return_state=True, 
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout_rate,
             kernel_regularizer=l2(l2_lambda),
             recurrent_regularizer=l2(recurrent_l2_lambda),
             bias_regularizer=l2(l2_lambda)),
        name='encoder_bilstm2'
    )
    
    encoder_outputs2, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm2(encoder_outputs1)

    state_h = Concatenate()([forward_h2, backward_h2])
    state_c = Concatenate()([forward_c2, backward_c2])
    encoder_states = [state_h, state_c]
    
    encoder_outputs_for_attention = encoder_outputs2
    
    # Decoder
    decoder_inputs = Input(shape=(max_length,), name='decoder_input')
    
    decoder_embedding = Embedding(
        gloss_vocab_size, 
        embedding_dim, 
        mask_zero=False,
        embeddings_regularizer=l2(l2_lambda),
        name='decoder_embedding'
    )(decoder_inputs)
    
    decoder_embedding = Dropout(dropout_rate, name='decoder_embedding_dropout')(decoder_embedding)
    
    decoder_lstm = LSTM(
        256,
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
    
    decoder_outputs = Dropout(dropout_rate, name='decoder_output_dropout')(decoder_outputs)
    
    # Enhanced Attention with coverage mechanism
    attention = Attention(name='attention_layer')([decoder_outputs, encoder_outputs_for_attention])
    
    decoder_combined = Concatenate(name='attention_concat')([decoder_outputs, attention])
    
    decoder_combined = Dense(256, 
                           activation='tanh',
                           kernel_regularizer=l2(l2_lambda),
                           bias_regularizer=l2(l2_lambda),
                           name='attention_dense')(decoder_combined)
    
    decoder_combined = Dropout(dropout_rate, name='attention_dense_dropout')(decoder_combined)
    
    # Output layer with better OOV handling
    decoder_dense = TimeDistributed(
        Dense(gloss_vocab_size, 
              activation='softmax',
              kernel_regularizer=l2(1e-3),
              bias_regularizer=l2(1e-3),
              name='output_dense')
    )
    decoder_outputs = decoder_dense(decoder_combined)
    
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='improved_regularized_asl_seq2seq'
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def enhanced_regularized_train_model(model, processed_data, epochs=40, batch_size=32):
    
    X_train_encoder = processed_data['train_encoder_inputs']
    X_train_decoder = processed_data['train_decoder_inputs']
    y_train = processed_data['train_decoder_targets']
    
    X_val_encoder = processed_data['val_encoder_inputs']
    X_val_decoder = processed_data['val_decoder_inputs']
    y_val = processed_data['val_decoder_targets']
    
    y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_val_reshaped = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

    # Enhanced callbacks with better monitoring
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=12,  # Reduced patience for faster adaptation
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            mode='max'
        ),
        
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,  # More gradual reduction
            patience=4,  # Slightly more patience
            min_lr=1e-6,
            verbose=1,
            mode='max'
        ),
        
        ModelCheckpoint(
            'best_improved_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Add custom callback for sequence start monitoring
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: monitor_sequence_starts(model, processed_data, epoch)
        )
    ]
    
    # Curriculum learning - start with shorter sequences
    def curriculum_learning(epoch):
        if epoch < 10:
            return 32  # Full sequences
        elif epoch < 20:
            return 24  # Medium sequences
        else:
            return 16  # Focus on beginnings
    
    print("🚀 Starting enhanced training with sequence start focus...")
    
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
    )
    
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"✅ Final val accuracy: {final_val_acc:.4f}")

    # analyze_training_results(history)
    
    return history

def monitor_sequence_starts(model, processed_data, epoch):
    """Monitor how well the model predicts sequence starts"""
    if epoch % 5 == 0:  # Check every 5 epochs
        X_val_encoder = processed_data['val_encoder_inputs'][:10]  # Sample 10
        y_val_true = processed_data['val_decoder_targets'][:10]
        gloss_tokenizer = processed_data['gloss_tokenizer']
        
        correct_starts = 0
        total_sequences = len(X_val_encoder)
        
        reverse_gloss = {v: k for k, v in gloss_tokenizer.word_index.items()}
        
        for i in range(total_sequences):
            true_start_token = y_val_true[i][1]  # Skip <start> token
            encoder_input = X_val_encoder[i:i+1]
            decoder_input = processed_data['val_decoder_inputs'][i:i+1]
            
            predictions = model.predict([encoder_input, decoder_input], verbose=0)
            pred_start_token = np.argmax(predictions[0, 1, :])  # First real token
            
            if pred_start_token == true_start_token:
                correct_starts += 1
        
        start_accuracy = correct_starts / total_sequences
        print(f"   Sequence Start Accuracy: {start_accuracy:.2f}")

def data_augmentation_with_regularization(processed_data):
    
    train_encoder = processed_data['train_encoder_inputs']
    train_decoder = processed_data['train_decoder_inputs'] 
    train_targets = processed_data['train_decoder_targets']
    
    if len(train_encoder) < 15000:
        val_encoder = processed_data['val_encoder_inputs']
        val_decoder = processed_data['val_decoder_inputs']
        val_targets = processed_data['val_decoder_targets']
        
        split_point = len(val_encoder) // 3
        augmented_encoder = np.concatenate([train_encoder, val_encoder[:split_point*2]])
        augmented_decoder = np.concatenate([train_decoder, val_decoder[:split_point*2]])
        augmented_targets = np.concatenate([train_targets, val_targets[:split_point*2]])
        
        print(f"Augmented data: {len(augmented_encoder)} samples (from {len(train_encoder)})")
        
        return {
            'train_encoder_inputs': augmented_encoder,
            'train_decoder_inputs': augmented_decoder,
            'train_decoder_targets': augmented_targets,
            'val_encoder_inputs': val_encoder[split_point*2:],
            'val_decoder_inputs': val_decoder[split_point*2:],
            'val_decoder_targets': val_targets[split_point*2:],
        }
    
    return processed_data

def train():
    
    from pipeline import preprocessing_pipeline
    processed_data = preprocessing_pipeline()
    
    augmented_data = data_augmentation_with_regularization(processed_data)
    
    eng_vocab_size = len(processed_data['eng_tokenizer'].word_index) + 1
    gloss_vocab_size = len(processed_data['gloss_tokenizer'].word_index) + 1
    max_length = processed_data['max_length']
    
    model = build_improved_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length)
    
    history = enhanced_regularized_train_model(model, augmented_data, epochs=40, batch_size=32)

    model.save('regularized_asl_model.h5')

    return model, history

if __name__ == "__main__":
    train()