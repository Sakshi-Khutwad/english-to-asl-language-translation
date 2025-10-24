import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, RepeatVector, TimeDistributed, Input

def build_seq2seq_model(eng_vocab_size, gloss_vocab_size, max_length, embedding_dim=128):
    print('Building the seq2seq model')

    # Encoder
    encoder_inputs = Input(shape=(max_length,), name='encoder_input')
    encoder_embedding = Embedding(eng_vocab_size, embedding_dim, name='encoder_embedding')(encoder_inputs)
    encoder_lstm1 = LSTM(256, return_sequences=True, return_state=True, name='encoder_lstm1')
    encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)

    encoder_lstm2 = LSTM(128, return_state=True, name='encoder_lstm2')
    encoder_outputs, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_states = [state_h2, state_c2]

    # Decoder
    decoder_inputs = Input(shape=(max_length,), name='decoder_input')
    decoder_embedding = Embedding(gloss_vocab_size, embedding_dim, name='decoder_embedding')(decoder_inputs)
    
    decoder_lstm1 = LSTM(128, return_sequences=True, return_state=True, name='decoder_lstm1')
    decoder_output1, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder_states)
    
    decoder_lstm2 = LSTM(256, return_sequences=True, name='decoder_lstm2')
    decoder_outputs = decoder_lstm2(decoder_output1)

    decoder_dense = TimeDistributed(Dense(gloss_vocab_size, activation='softmax'), name='output')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='asl_seq2seq'
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Model built successfully')

    return model

def train_model(model, processed_data, epochs=50, batch_size=32):

    print('Starting model training')
    y_train_reshaped = processed_data['train_decoder_targets'].reshape(
            processed_data['train_decoder_targets'].shape[0],
            processed_data['train_decoder_targets'].shape[1],
            1
    )
    
    with tf.device('/GPU:0'):  # 👈 Force GPU
        history = model.fit(
            [processed_data['train_encoder_inputs'], processed_data['train_decoder_inputs']],
            y_train_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [processed_data['val_encoder_inputs'], processed_data['val_decoder_inputs']],
                processed_data['val_decoder_targets'].reshape(
                    processed_data['val_decoder_targets'].shape[0],
                    processed_data['val_decoder_targets'].shape[1],
                    1
                )
            ),
            verbose=1
        )
    return history

