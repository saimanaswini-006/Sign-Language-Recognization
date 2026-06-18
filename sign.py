import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import time

# ── CONFIG ───────────────────────────────────────────────────────────────
GESTURES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
            'OK', 'Peace', 'Thumbs_Up', 'Hello', 'None']
SEQUENCE_LENGTH = 30       # frames per gesture sequence
NUM_LANDMARKS = 21 * 3     # 21 hand landmarks × (x, y, z)
DATA_DIR = "gesture_data/" # folder to save recorded sequences

# ── MEDIAPIPE SETUP ───────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extract_landmarks(frame):
    """Extract 63 landmark coordinates from a frame. Returns zeros if no hand."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        coords = []
        for pt in lm.landmark:
            coords.extend([pt.x, pt.y, pt.z])
        return np.array(coords), result.multi_hand_landmarks[0]
    return np.zeros(NUM_LANDMARKS), None

# ── COLLECT TRAINING DATA ─────────────────────────────────────────────────
def collect_data(gesture_name, num_sequences=50):
    """Record gesture sequences from webcam."""
    os.makedirs(f"{DATA_DIR}/{gesture_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)

    print(f"\n▶️ Recording '{gesture_name}'. Press SPACE to start each sequence. Q to quit.")

    seq_count = 0
    while seq_count < num_sequences:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Gesture: {gesture_name} | Seq: {seq_count}/{num_sequences}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "Press SPACE to record", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            sequence = []
            for _ in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                landmarks, hand_lm = extract_landmarks(frame)
                if hand_lm:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                sequence.append(landmarks)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            np.save(f"{DATA_DIR}/{gesture_name}/seq_{seq_count}.npy", np.array(sequence))
            seq_count += 1
            print(f"  Saved sequence {seq_count}")

    cap.release()
    cv2.destroyAllWindows()

# ── LOAD DATASET ──────────────────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    for label, gesture in enumerate(GESTURES):
        gesture_dir = f"{DATA_DIR}/{gesture}"
        if not os.path.exists(gesture_dir):
            continue
        for file in os.listdir(gesture_dir):
            if file.endswith('.npy'):
                seq = np.load(os.path.join(gesture_dir, file))
                X.append(seq)
                y.append(label)
    return np.array(X), np.array(y)

# ── BUILD LSTM MODEL ──────────────────────────────────────────────────────
def build_lstm_model(num_classes):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_LANDMARKS)),
        layers.Dropout(0.3),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── TRAIN ─────────────────────────────────────────────────────────────────
def train_model():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Dataset: {X.shape}, Labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model(len(GESTURES))
    model.summary()

    callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=16,
              callbacks=callbacks)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    model.save("sign_language_model.h5")
    print("Model saved as sign_language_model.h5")
    return model

# ── REAL-TIME INFERENCE ───────────────────────────────────────────────────
def real_time_recognition(model_path="sign_language_model.h5"):
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)

    sequence = []
    prediction_text = ""
    confidence = 0.0

    print("\n▶️ Real-time Sign Language Recognition started. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, hand_lm = extract_landmarks(frame)

        if hand_lm:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        sequence.append(landmarks)
        sequence = sequence[-SEQUENCE_LENGTH:]  # keep last N frames

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            pred_idx = np.argmax(prediction)
            confidence = prediction[pred_idx] * 100
            if confidence > 70:
                prediction_text = GESTURES[pred_idx]

        # Display
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {prediction_text}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── MAIN ──────────────────────────────────────────────────────────────────
if _name_ == "_main_":
    print("Sign Language Recognition System")
    print("=================================")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Real-time recognition")
    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == '1':
        gesture = input("Enter gesture name (e.g. A, B, Thumbs_Up): ").strip()
        collect_data(gesture)
    elif choice == '2':
        train_model()
    elif choice == '3':
        real_time_recognition()
    else:
        print("Invalid choice.")