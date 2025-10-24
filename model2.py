import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=128):
    print('Building Improved seq2seq model with Bidirectional LSTM & Regularization')

    # ==================== ENCODER ====================
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    
    # Encoder embedding with regularization
    encoder_embedding = Embedding(
        eng_vocab_size, 
        embedding_dim, 
        mask_zero=True,  # Ignore padding
        embeddings_regularizer=l2(1e-4),  # L2 regularization
        name='encoder_embedding'
    )(encoder_inputs)
    
    # Add dropout to embedding layer
    encoder_embedding = Dropout(0.3)(encoder_embedding)

    # First Bidirectional LSTM with regularization
    encoder_bilstm1 = Bidirectional(
        LSTM(
            256, 
            return_sequences=True, 
            dropout=0.3,           # Dropout for inputs
            recurrent_dropout=0.2, # Dropout for recurrent connections
            kernel_regularizer=l2(1e-4),  # Weight regularization
            recurrent_regularizer=l2(1e-4), # Recurrent weight regularization
            name='encoder_bilstm1'
        )
    )
    encoder_output1 = encoder_bilstm1(encoder_embedding)
    
    # Add dropout between LSTM layers
    encoder_output1 = Dropout(0.3)(encoder_output1)

    # Second Bidirectional LSTM with regularization
    encoder_bilstm2 = Bidirectional(
        LSTM(
            128, 
            return_state=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
            name='encoder_bilstm2'
        )
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm2(encoder_output1)

    # Combine forward and backward states
    encoder_state_h = Concatenate()([forward_h, backward_h])
    encoder_state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [encoder_state_h, encoder_state_c]

    # ==================== DECODER ====================
    decoder_inputs = Input(shape=(max_length,), name='decoder_input')
    
    # Decoder embedding with regularization
    decoder_embedding = Embedding(
        gloss_vocab_size, 
        embedding_dim, 
        mask_zero=True,
        embeddings_regularizer=l2(1e-4),
        name='decoder_embedding'
    )(decoder_inputs)
    
    # Add dropout to decoder embedding
    decoder_embedding = Dropout(0.3)(decoder_embedding)
    
    # First Decoder LSTM - needs 256 units to match bidirectional encoder output (2*128)
    decoder_lstm1 = LSTM(
        256,  # Matches encoder output size (2 * 128)
        return_sequences=True, 
        return_state=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(1e-4),
        recurrent_regularizer=l2(1e-4),
        name='decoder_lstm1'
    )
    decoder_output1, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder_states)
    
    # Dropout between decoder layers
    decoder_output1 = Dropout(0.3)(decoder_output1)
    
    # Second Decoder LSTM
    decoder_lstm2 = LSTM(
        256, 
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(1e-4),
        recurrent_regularizer=l2(1e-4),
        name='decoder_lstm2'
    )
    decoder_outputs = decoder_lstm2(decoder_output1)
    
    # Final dropout before output
    decoder_outputs = Dropout(0.3)(decoder_outputs)

    # Output layer with strong regularization
    decoder_dense = TimeDistributed(
        Dense(
            gloss_vocab_size, 
            activation='softmax',
            kernel_regularizer=l2(1e-3),  # Stronger regularization on output
            bias_regularizer=l2(1e-3),
            name='output'
        )
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='asl_seq2seq_improved'
    )

    # Use Adam with lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    print('Improved Model built successfully with Bidirectional LSTM & Regularization')

    return model

def train_model(model, processed_data, epochs=35, batch_size=64):
    print('Starting model training with anti-overfitting measures')

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

    # Anti-overfitting callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop after 10 epochs without improvement
            restore_best_weights=True,  # Keep the best weights
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,    # Reduce LR by half
            patience=5,    # After 5 epochs without improvement
            min_lr=1e-7,   # Minimum learning rate
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
        callbacks=callbacks,  # Add the anti-overfitting callbacks
        shuffle=True,         # Shuffle training data each epoch
        verbose=1
    )
    
    print(f"Training stopped at epoch {len(history.history['loss'])}")
    return history