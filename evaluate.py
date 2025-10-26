def evaluate_model(model, processed_data):
    """Evaluate model on test set"""
    
    # Prepare test data
    X_test = processed_data['test_encoder_inputs']
    y_test = processed_data['test_decoder_targets']
    
    # Reshape targets
    y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        [processed_data['test_encoder_inputs'], processed_data['test_decoder_inputs']],
        y_test_reshaped,
        verbose=1
    )
    
    print(f"🎯 Test Loss: {test_loss:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy
