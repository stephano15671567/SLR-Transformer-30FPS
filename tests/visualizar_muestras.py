import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
from src.data_engine.extractor import FeatureExtractor
from config.hparams import VIDEO_ROOT, CSV_PATH

def auditar_muestras():
    df = pd.read_csv(CSV_PATH)
    extractor = FeatureExtractor()
    # Buscamos un video que exista
    intentos = 0
    while intentos < 20:
        sample = df.sample(1).iloc[0]
        video_path = extractor.get_video_path(VIDEO_ROOT, sample['video_id'])
        if os.path.exists(video_path): break
        intentos += 1

    cap = cv2.VideoCapture(video_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # --- CONFIGURACION DE COLORES (BGR) ---
    # Pose: Naranja
    c_pose = mp_drawing.DrawingSpec(color=(80, 110, 255), thickness=2, circle_radius=1)
    # Mano Izquierda: Verde Cian
    c_izq = mp_drawing.DrawingSpec(color=(121, 255, 76), thickness=2, circle_radius=1)
    # Mano Derecha: Rojo Brillante
    c_der = mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
    # Rostro: Blanco (Sutil)
    c_face = mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Inferencia
        results = extractor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 1. Rostro (Malla fina)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, c_face, c_face)
        # 2. Pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, c_pose, c_pose)
        # 3. Manos (Diferenciadas)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, c_izq, c_izq)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, c_der, c_der)
            
        cv2.imshow('Auditoria Profesional SLR - 30 FPS', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    auditar_muestras()