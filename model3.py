import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, Dropout, Concatenate, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=256):
    print('Building LSTM model with Attention Mechanism')
    
    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    
    # Encoder embedding
    encoder_embedding = Embedding(
        eng_vocab_size, 
        embedding_dim, 
        name='encoder_embedding'
    )(encoder_inputs)
    
    # Encoder LSTM layers - return sequences for attention
    encoder_lstm1 = Bidirectional(
        LSTM(256, return_sequences=True, dropout=0.2, name='encoder_lstm1')
    )
    encoder_output1 = encoder_lstm1(encoder_embedding)
    
    # Second encoder layer - return sequences for attention
    encoder_lstm2 = Bidirectional(
        LSTM(256, return_sequences=True, return_state=True, dropout=0.2, name='encoder_lstm2')
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm2(encoder_output1)
    
    # Encoder states for decoder initialization
    encoder_state_h = Concatenate()([forward_h, backward_h])
    encoder_state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [encoder_state_h, encoder_state_c]
    
    # ==================== DECODER WITH ATTENTION ====================
    decoder_inputs = Input(shape=(max_length,), name='decoder_input')
    
    # Decoder embedding
    decoder_embedding = Embedding(
        gloss_vocab_size, 
        embedding_dim, 
        name='decoder_embedding'
    )(decoder_inputs)
    
    # First decoder LSTM
    decoder_lstm1 = LSTM(
        512,  # Larger to handle encoder states (2*256=512)
        return_sequences=True, 
        return_state=True,
        dropout=0.2,
        name='decoder_lstm1'
    )
    decoder_output1, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder_states)
    
    # ==================== ATTENTION MECHANISM ====================
    # Bahdanau-style attention
    attention = Attention()([decoder_output1, encoder_outputs])
    
    # Concatenate decoder output with attention context
    decoder_combined = Concatenate()([decoder_output1, attention])
    
    # Optional: Dense layer to combine information
    decoder_combined = Dense(512, activation='tanh')(decoder_combined)
    decoder_combined = Dropout(0.2)(decoder_combined)
    
    # Second decoder LSTM
    decoder_lstm2 = LSTM(
        512, 
        return_sequences=True,
        dropout=0.2,
        name='decoder_lstm2'
    )
    decoder_outputs = decoder_lstm2(decoder_combined)
    
    # Output layer
    decoder_dense = TimeDistributed(
        Dense(gloss_vocab_size, activation='softmax', name='output')
    )
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Build model
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='lstm_attention_seq2seq'
    )
    
    # Optimizer with slightly higher learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    print('LSTM with Attention model built successfully')
    return model

def train_model(model, processed_data, epochs=35, batch_size=64):
    print('Training LSTM with Attention model')
    
    # Reshape targets
    y_train_reshaped = processed_data['train_decoder_targets'].reshape(
        processed_data['train_decoder_targets'].shape[0],
        processed_data['train_decoder_targets'].shape[1],
        1
    )
    
    y_val_reshaped = processed_data['val_decoder_targets'].reshape(
        processed_data['val_decoder_targets'].shape[0],
        processed_data['val_decoder_targets'].shape[1],
        1
    )

    # More aggressive callbacks for attention model
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode='max'  # Important: we want to maximize accuracy
        ),
        
        ReduceLROnPlateau(
            monitor='val_accuracy',  # Monitor accuracy
            factor=0.5,
            patience=5,    # Reduce LR if no improvement for 5 epochs
            min_lr=1e-6,
            verbose=1,
            mode='max'     # We want to maximize accuracy
        ),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            'best_lstm_attention_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    history = model.fit(
        [processed_data['train_encoder_inputs'], processed_data['train_decoder_inputs']],
        y_train_reshaped,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [processed_data['val_encoder_inputs'], processed_data['val_decoder_inputs']],
            y_val_reshaped
        ),
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    print(f"Training stopped at epoch {len(history.history['loss'])}")
    return history