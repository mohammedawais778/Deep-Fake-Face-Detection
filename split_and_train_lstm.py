import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load features
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save splits
os.makedirs("data/features", exist_ok=True)
np.save("data/features/X_train.npy", X_train)
np.save("data/features/X_val.npy", X_val)
np.save("data/features/X_test.npy", X_test)
np.save("data/features/y_train.npy", y_train)
np.save("data/features/y_val.npy", y_val)
np.save("data/features/y_test.npy", y_test)

# Build LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
checkpoint = ModelCheckpoint('model/lstm_model_retrained.keras', save_best_only=True, monitor='val_accuracy')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16, callbacks=[early_stop, checkpoint])

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Predict & Calculate Metrics
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Deepfake", "Real"]))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
