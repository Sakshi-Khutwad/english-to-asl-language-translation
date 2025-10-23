from pipeline import preprocessing_pipeline
from model import build_seq2seq_model, train_model
import matplotlib.pyplot as plt

def main():
    print('Starting the ASL Translation Pipeline')
    processed_data = preprocessing_pipeline()

    model = build_seq2seq_model(
        eng_vocab_size=processed_data['eng_vocab_size'],
        gloss_vocab_size=processed_data['gloss_vocab_size'],
        max_length=processed_data['max_length']
    )

    history = train_model(model, processed_data, epochs=50, batch_size=32)

    print('model training completed')
    model.save('asl_translation_model.h5')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return model, processed_data

if __name__ == '__main__':
    model, processed_data = main()
