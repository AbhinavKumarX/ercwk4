import cv2
import mediapipe as mp
import numpy as np
import random

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 30
PLAYER_SIZE = 50
BLOCK_FALL_SPEED = 5
BLOCK_SPAWN_RATE = 30  # Lower is more frequent

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize OpenCV window
cv2.namedWindow('Dodge Game')

# Function to check collision
def check_collision(player_pos, block_pos):
    return (player_pos[0] < block_pos[0] + BLOCK_SIZE and
            player_pos[0] + PLAYER_SIZE > block_pos[0] and
            player_pos[1] < block_pos[1] + BLOCK_SIZE and
            player_pos[1] + PLAYER_SIZE > block_pos[1])

# Main game loop
def main():
    cap = cv2.VideoCapture(0)
    player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_SIZE - 10]
    blocks = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Get hand position
        hand_x, hand_y = player_pos
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * SCREEN_WIDTH) - PLAYER_SIZE // 2
            hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * SCREEN_HEIGHT)

        # Update player position
        player_pos[0] = max(0, min(hand_x, SCREEN_WIDTH - PLAYER_SIZE))
        player_pos[1] = hand_y

        # Spawn blocks
        if frame_count % BLOCK_SPAWN_RATE == 0:
            block_x = random.randint(0, SCREEN_WIDTH - BLOCK_SIZE)
            blocks.append([block_x, 0])  # New block at the top

        # Move blocks down
        for block in blocks:
            block[1] += BLOCK_FALL_SPEED

        # Check for collisions
        blocks = [block for block in blocks if block[1] < SCREEN_HEIGHT]  # Remove blocks that are off-screen
        for block in blocks:
            if check_collision(player_pos, block):
                print("Game Over!")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Drawing
        frame.fill(0)  # Clear the frame
        for block in blocks:
            cv2.rectangle(frame, (block[0], block[1]), (block[0] + BLOCK_SIZE, block[1] + BLOCK_SIZE), (0, 255, 0), -1)

        # Draw player
        cv2.rectangle(frame, (player_pos[0], player_pos[1]), (player_pos[0] + PLAYER_SIZE, player_pos[1] + PLAYER_SIZE), (255, 0, 0), -1)

        # Display the frame
        cv2.imshow('Dodge Game', frame)

        frame_count += 1

        # Exit game if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
