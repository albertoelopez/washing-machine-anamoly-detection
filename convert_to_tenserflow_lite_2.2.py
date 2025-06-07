# convert_to_tflite.py
import tensorflow as tf
import numpy as np
import joblib
import os

def convert_model(model_path, output_dir='model'):
    """Convert a saved Keras model to TensorFlow Lite"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create a representative dataset for quantization
    def representative_data_gen():
        # Load some training data for calibration
        # This should match your training data preprocessing
        scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
        # Generate random data with the same shape as training data
        # Replace this with actual data samples for better results
        for _ in range(100):
            dummy_input = np.random.randn(1, *model.input_shape[1:]).astype(np.float32)
            dummy_input = scaler.transform(dummy_input)
            yield [dummy_input]
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert and save
    tflite_model = converter.convert()
    
    # Save the model
    tflite_model_path = os.path.join(output_dir, 'model_quant.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {tflite_model_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model_path

if __name__ == "__main__":
    convert_model('model/best_model.h5')