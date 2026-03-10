import cv2
import mediapipe as mp
import numpy as np
import os

class FeatureExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # Bajamos la complejidad a 0 solo para asegurar que el Xeon arranque fluido
        # y desactivamos el rastreo de rostro refinado.
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_video_path(self, video_root, video_id):
        return os.path.join(video_root, video_id, f"{video_id}.mp4")

    def _normalize_landmarks(self, pose_landmarks, lh_landmarks, rh_landmarks):
        if pose_landmarks:
            hombro_izq = [pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y, pose_landmarks.landmark[11].z]
            hombro_der = [pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y, pose_landmarks.landmark[12].z]
            centro_hombros = np.mean([hombro_izq, hombro_der], axis=0)
        else:
            centro_hombros = np.zeros(3)

        def center_and_flatten(landmarks, num_points, center):
            if not landmarks: return np.zeros(num_points * 3)
            coords = np.array([[l.x - center[0], l.y - center[1], l.z - center[2]] for l in landmarks.landmark])
            return coords.flatten()

        pose = center_and_flatten(pose_landmarks, 33, centro_hombros)
        lh = center_and_flatten(lh_landmarks, 21, centro_hombros)
        rh = center_and_flatten(rh_landmarks, 21, centro_hombros)
        
        return np.concatenate([pose, lh, rh])

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        # Leemos solo los primeros 300 frames para que el test no sea infinito
        # si el video es muy largo
        frame_count = 0
        while cap.isOpened() and frame_count < 300:
            ret, frame = cap.read()
            if not ret: break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Optimizacion de memoria
            results = self.holistic.process(image_rgb)
            
            sequence.append(self._normalize_landmarks(results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks))
            frame_count += 1
            
        cap.release()
        return np.array(sequence)