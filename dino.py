import numpy as np
import cv2
import pyautogui
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, Flatten
import time
import webbrowser
import random
from collections import deque
import os
import subprocess
import pickle

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configuration
SCREEN_REGION = (50, 200, 500, 300)  # Adjust based on your screen
IMG_SIZE = (80, 80)
SEQ_LENGTH = 10
ACTIONS = [0, 1, 2]  # 0: do nothing, 1: jump, 2: duck
ACTION_MAP = {0: None, 1: 'space', 2: 'down'}
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000
BATCH_SIZE = 32
EPISODES = 1000

memory = deque(maxlen=MEMORY_SIZE)

def build_model(seq_length=SEQ_LENGTH):
    model = Sequential([
        TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(seq_length, IMG_SIZE[0], IMG_SIZE[1], 1)),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    return normalized.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)

def capture_screen(region):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    return preprocess_frame(frame)

def get_score(frame):
    if not OCR_AVAILABLE:
        return frame_count * 0.1
    try:
        score_region = frame[0:40, -80:]
        gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6 digits')
        return float(text.strip()) if text.strip().isdigit() else frame_count * 0.1
    except:
        return frame_count * 0.1

def launch_game():
    try:
        os.system("taskkill /im chrome.exe /f")
        time.sleep(2)

        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        subprocess.Popen([chrome_path])
        time.sleep(5)

        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.5)
        pyautogui.typewrite('chrome://dino', interval=0.05)
        pyautogui.press('enter')
        time.sleep(3)

        pyautogui.hotkey('alt', 'space')
        time.sleep(0.5)
        pyautogui.press('x')
        time.sleep(1)

        pyautogui.press('space')
        time.sleep(2)

    except Exception as e:
        print(f"Error launching game: {e}")

def is_game_over(frame):
    try:
        score_area = frame[20:60, -120:-20]
        mean_value = np.mean(score_area)
        return mean_value < 0.6
    except:
        return False

def save_epsilon(epsilon):
    try:
        with open('epsilon.pkl', 'wb') as f:
            pickle.dump(epsilon, f)
    except Exception as e:
        print(f"Error saving epsilon: {e}")

def load_epsilon():
    try:
        if os.path.exists('epsilon.pkl'):
            with open('epsilon.pkl', 'rb') as f:
                return pickle.load(f)
        return EPSILON  # Default value if file doesn't exist
    except Exception as e:
        print(f"Error loading epsilon: {e}")
        return EPSILON

def train_dino_game():
    global EPSILON, frame_count

    # Load or build the model
    model_path = 'dino_rnn_model.h5'
    if os.path.exists(model_path):
        try:
            print("Loading saved model...")
            model = load_model(model_path)
            # Verify the model's input shape matches the current SEQ_LENGTH
            expected_input_shape = (None, SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1)
            model_input_shape = model.layers[0].input_shape
            if model_input_shape != expected_input_shape:
                print(f"Warning: Model input shape {model_input_shape} does not match expected {expected_input_shape}. Rebuilding model.")
                model = build_model(SEQ_LENGTH)
        except Exception as e:
            print(f"Error loading model: {e}. Building a new model.")
            model = build_model(SEQ_LENGTH)
    else:
        model = build_model(SEQ_LENGTH)
        print("Created new model")

    # Load the last epsilon value to continue exploration strategy
    EPSILON = load_epsilon()
    print(f"Loaded EPSILON: {EPSILON}")

    sequence = deque(maxlen=SEQ_LENGTH)
    frame_count = 0
    last_score = 0

    launch_game()

    for episode in range(EPISODES):
        sequence.clear()
        total_reward = 0
        step = 0
        game_over = False
        frame_count = 0
        last_score = 0

        for _ in range(SEQ_LENGTH):
            frame = capture_screen(SCREEN_REGION)
            sequence.append(frame)

        while not game_over:
            frame_count += 1
            state = np.array(sequence)

            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                q_values = model.predict(state[np.newaxis, ...], verbose=0)
                action = np.argmax(q_values[0])

            if ACTION_MAP[action]:
                pyautogui.press(ACTION_MAP[action])
                time.sleep(0.3)
                if action == 2:
                    pyautogui.keyUp('down')

            next_frame = capture_screen(SCREEN_REGION)
            sequence.append(next_frame)
            next_state = np.array(sequence)

            raw_frame = np.array(pyautogui.screenshot(region=SCREEN_REGION))
            score = get_score(raw_frame)
            score_increase = score - last_score
            last_score = score

            game_over = is_game_over(next_frame)
            reward = 1.0 + score_increase * 0.5 if not game_over else -100
            total_reward += reward

            memory.append((state, action, reward, next_state, game_over))

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states = np.array([exp[0] for exp in batch])
                actions = np.array([exp[1] for exp in batch])
                rewards = np.array([exp[2] for exp in batch])
                next_states = np.array([exp[3] for exp in batch])
                dones = np.array([exp[4] for exp in batch])

                targets = model.predict(states, verbose=0)
                next_q_values = model.predict(next_states, verbose=0)
                for i in range(BATCH_SIZE):
                    targets[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + GAMMA * np.max(next_q_values[i])
                model.fit(states, targets, epochs=1, verbose=0)

            step += 1
            print(f"Step: {step}, Action: {action}, Score: {score:.2f}, Reward: {reward:.2f}, Game Over: {game_over}")

            if game_over:
                print(f"Episode {episode + 1}/{EPISODES}, Game Score: {score:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")
                pyautogui.press('space')
                time.sleep(2)
                sequence.clear()
                for _ in range(SEQ_LENGTH):
                    sequence.append(capture_screen(SCREEN_REGION))
                break

        if (episode + 1) % 10 == 0:
            model.save('dino_rnn_model.h5')
            save_epsilon(EPSILON)
            print("Model and epsilon saved successfully")

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

if __name__ == "__main__":
    try:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 1.0
        print("Minimize all other windows now! Starting in 5 seconds...")
        print("Move your mouse to the top-left corner to abort.")
        time.sleep(5)
        train_dino_game()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}") 