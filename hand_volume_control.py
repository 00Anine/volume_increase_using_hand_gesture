import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ऑडियो कंट्रोल सेटअप
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# वेबकैम सेटअप
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # चौड़ाई सेट करें
cap.set(4, 480)  # ऊंचाई सेट करें

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
            
            # हाथ के लैंडमार्क्स को ड्रा करें
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # थंब और इंडेक्स फिंगर के बीच की दूरी निकालें
            if len(landmarks) > 0:
                x1, y1 = landmarks[4][0], landmarks[4][1]  # थंब
                x2, y2 = landmarks[8][0], landmarks[8][1]  # इंडेक्स फिंगर
                
                # दो उंगलियों के बीच लाइन खींचें
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # दूरी की गणना करें
                length = hypot(x2 - x1, y2 - y1)
                
                # हैंड रेंज 50 - 300
                # वॉल्यूम रेंज -65 - 0
                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                
                # वॉल्यूम बार दिखाएं
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])
                
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                          1, (0, 255, 0), 3)

    cv2.imshow('Hand Volume Control', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
