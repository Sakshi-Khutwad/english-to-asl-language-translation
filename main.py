from pipeline import preprocessing_pipeline
from model2 import build_seq2seq_model, train_model
import matplotlib.pyplot as plt
import tensorflow as tf

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Allow TensorFlow to allocate GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # (Optional) Use only specific GPU if multiple available
        # tf.config.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) available.")
        print("TensorFlow will use GPU for computation.")

        # Log device placement (to confirm ops are on GPU)
        tf.debugging.set_log_device_placement(True)

        # Enable mixed precision for faster training (if supported)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("🚀 Mixed precision enabled for faster GPU training.")

    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("❌ No GPU detected — TensorFlow will use CPU.")


def main():
    print('Starting the ASL Translation Pipeline')
    processed_data = preprocessing_pipeline()

    model = build_seq2seq_model(
        eng_vocab_size=processed_data['eng_vocab_size'],
        gloss_vocab_size=processed_data['gloss_vocab_size'],
        max_length=processed_data['max_length']
    )

    history = train_model(model, processed_data, epochs=35, batch_size=32)

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
