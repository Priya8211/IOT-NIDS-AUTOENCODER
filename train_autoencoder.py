import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from sklearn.preprocessing import MinMaxScaler
import joblib
import argparse

def build_autoencoder(input_dim):
    # Standard N-BaIoT Autoencoder architecture: 115 -> 64 -> 32 -> 16 -> 32 -> 64 -> 115
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='tanh')(input_layer)
    encoded = Dense(32, activation='tanh')(encoded)
    bottleneck = Dense(16, activation='tanh')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='tanh')(bottleneck)
    decoded = Dense(64, activation='tanh')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

class StreamlitProgressCallback(Callback):
    def __init__(self, epochs, update_fn):
        super().__init__()
        self.total_epochs = epochs
        self.update_fn = update_fn

    def on_epoch_end(self, epoch, logs=None):
        if self.update_fn:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            self.update_fn(epoch + 1, self.total_epochs, loss, val_loss)

def train_model(device_dir, epochs=10, batch_size=256, max_samples=40000, progress_callback=None):
    """
    Trains the N-BaIoT autoencoder on benign traffic dataset in the specified device_dir.
    """
    benign_path = os.path.join(device_dir, "benign_traffic.csv")
    if not os.path.exists(benign_path):
        raise FileNotFoundError(f"Benign traffic file not found at {benign_path}")
        
    print(f"[*] Reading dataset from {benign_path}...")
    # Load csv. Limit samples if specified to save RAM/time
    if max_samples:
        df = pd.read_csv(benign_path, nrows=max_samples)
    else:
        df = pd.read_csv(benign_path)
        
    print(f"[✓] Loaded {len(df)} benign samples. Features: {df.shape[1]}")
    
    # Check shape
    input_dim = df.shape[1]
    
    # Fit MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)
    
    # Split into train/val
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    
    # Build autoencoder
    model = build_autoencoder(input_dim)
    
    # Fit callbacks
    callbacks = []
    if progress_callback:
        cb = StreamlitProgressCallback(epochs, progress_callback)
        callbacks.append(cb)
        
    print("[*] Training Autoencoder model...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    # Save the scaler and model
    scaler_filename = "scaler.save"
    model_filename = "nbaiot_autoencoder.h5"
    
    joblib.dump(scaler, scaler_filename)
    model.save(model_filename)
    
    print(f"[✓] Saved scaler to {scaler_filename}")
    print(f"[✓] Saved autoencoder model to {model_filename}")
    return history.history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train N-BaIoT Autoencoder on IoT device benign traffic.")
    parser.add_argument("--device", type=str, required=True, help="Path to device folder containing benign_traffic.csv")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--samples", type=int, default=40000, help="Max benign samples to train on")
    args = parser.parse_args()
    
    if os.path.exists(args.device):
        train_model(args.device, epochs=args.epochs, batch_size=args.batch, max_samples=args.samples)
    else:
        print(f"[!] Error: device folder {args.device} does not exist.")
