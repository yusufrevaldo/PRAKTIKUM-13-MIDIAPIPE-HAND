import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil data tangan kanan/kiri
            label = results.multi_handedness[idx].classification[0].label

            thumb_tip = hand_landmarks.landmark[4]
            pinky_tip = hand_landmarks.landmark[20]

            # Logika depan / belakang
            if label == "Right":
                if thumb_tip.x < pinky_tip.x:
                    posisi = "DEPAN"
                else:
                    posisi = "BELAKANG"
            else:  # Left hand
                if thumb_tip.x > pinky_tip.x:
                    posisi = "DEPAN"
                else:
                    posisi = "BELAKANG"

            cv2.putText(img, posisi, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3)

    cv2.imshow("Deteksi Depan-Belakang", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()