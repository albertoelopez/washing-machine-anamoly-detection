# train_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_labeled_data(data_dir='collected_data/labeled'):
    """Load all labeled data files"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and f.startswith('labeled_')]
    if not files:
        raise FileNotFoundError("No labeled data found. Run the labeling tool first.")
    
    dfs = []
    for file in files:
        df = pd.read_parquet(os.path.join(data_dir, file))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def prepare_data(df, test_size=0.2, val_size=0.1):
    """Prepare data for training"""
    # Separate features and labels
    X = df.drop(columns=['label', 'machine_id', 'window_start'], errors='ignore')
    y = pd.get_dummies(df['label'])  # Convert to one-hot encoding
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training set into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_size/(1-test_size), 
        random_state=42,
        stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, scaler)

def create_model(input_shape, num_classes):
    """Create a CNN model for time series classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape[0]//100, 100, 1), 
                              input_shape=(input_shape[0],)),
        
        # First Conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Second Conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train():
    # Load and prepare data
    df = load_labeled_data()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(df)
    
    # Create and compile model
    model = create_model(X_train.shape[1:], y_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save the final model
    model.save('model/final_model.h5')
    
    return model, history

if __name__ == "__main__":
    train()