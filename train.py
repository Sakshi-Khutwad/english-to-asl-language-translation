import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, Dropout, Concatenate, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

def build_regularized_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=128):
    
    l2_lambda = 1e-4 
    recurrent_l2_lambda = 1e-5
    dropout_rate = 0.2  
    recurrent_dropout_rate = 0.1  
    
    # Encoder
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    
    # Encoder embedding
    encoder_embedding = Embedding(
        eng_vocab_size, 
        embedding_dim, 
        mask_zero=False,
        embeddings_regularizer=l2(l2_lambda),
        name='encoder_embedding'
    )(encoder_inputs)
    
    encoder_embedding = Dropout(dropout_rate, name='encoder_embedding_dropout')(encoder_embedding)
    

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
    
    # Attention Mechanism
    attention = Attention(name='attention_layer')([decoder_outputs, encoder_outputs_for_attention])
    
    decoder_combined = Concatenate(name='attention_concat')([decoder_outputs, attention])
    
    decoder_combined = Dense(256, 
                           activation='tanh',
                           kernel_regularizer=l2(l2_lambda),
                           bias_regularizer=l2(l2_lambda),
                           name='attention_dense')(decoder_combined)
    
    decoder_combined = Dropout(dropout_rate, name='attention_dense_dropout')(decoder_combined)
    
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
        name='regularized_asl_seq2seq_2layer'
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

def regularized_train_model(model, processed_data, epochs=40, batch_size=32):
    
    X_train_encoder = processed_data['train_encoder_inputs']
    X_train_decoder = processed_data['train_decoder_inputs']
    y_train = processed_data['train_decoder_targets']
    
    X_val_encoder = processed_data['val_encoder_inputs']
    X_val_decoder = processed_data['val_decoder_inputs']
    y_val = processed_data['val_decoder_targets']
    
    y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_val_reshaped = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15, 
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            mode='max'
        ),
        
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode='max'
        ),
        
        ModelCheckpoint(
            'best_regularized_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
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

    print(f"Final val accuracy: {final_val_acc:.4f}")

    analyze_training_results(history)
    
    return history

def analyze_training_results(history):
    
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
    
    if 'val_accuracy' in history.history:
        overfitting_gap = train_acc - val_acc
        if overfitting_gap > 0.1:
            print("Overfitting detected")
            print("   Consider increasing regularization or dropout")
        elif overfitting_gap < 0.02:
            print("Good generalization - train/val gap is small")
    
    # Check if model converged
    if val_acc > 0.7:
        print("Good accuracy!")
    elif val_acc > 0.6:
        print("Learning well")
    elif val_acc > 0.5:
        print("Needs improvement")
    else:
        print("Architectural changes needed")

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
    
    model = build_regularized_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length)
    
    history = regularized_train_model(model, augmented_data, epochs=40, batch_size=32)

    model.save('regularized_asl_model.h5')

    return model, history

if __name__ == "__main__":
    train()