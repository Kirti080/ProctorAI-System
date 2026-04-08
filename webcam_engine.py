import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import pyttsx3
import threading

class ProctorVoice:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 5 

    def say(self, text):
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown:
            # Running voice in a completely separate daemon thread
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()
            self.last_alert_time = current_time

    def _speak(self, text):
        try:
            # Local initialization inside the thread prevents the main loop from lagging
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop() 
        except:
            pass

def start_smart_proctor():
    voice = ProctorVoice()
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    print("Loading YOLO AI...")
    yolo_model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    last_blink_time = time.time()
    frame_count = 0  # To skip frames for YOLO

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # 1. OPTIMIZED YOLO (Run only every 5 frames to save CPU)
            if frame_count % 5 == 0:
                yolo_results = yolo_model.predict(frame, classes=[67], conf=0.5, verbose=False)
                if len(yolo_results[0].boxes) > 0:
                    voice.say("Phone detected")
                    for box in yolo_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # 2. MEDIAPIPE LOGIC
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(image_rgb)
            hand_results = hands.process(image_rgb)

            # Face Tracking
            if mesh_results.multi_face_landmarks:
                face = mesh_results.multi_face_landmarks[0]
                
                # Check Tilt
                depth_diff = face.landmark[33].z - face.landmark[263].z
                if abs(depth_diff) > 0.030: # Wider threshold for less noise
                    voice.say("Please look at the screen")
                    cv2.putText(frame, "Looking Away", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # Check Blink
                if abs(face.landmark[159].y - face.landmark[145].y) < 0.012:
                    last_blink_time = time.time()
                
                if time.time() - last_blink_time > 15:
                    voice.say("Liveness alert")
                    cv2.putText(frame, "SPOOF ALERT", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(frame, "Student Missing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Hand Tracking
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    if hand_lms.landmark[8].y < hand_lms.landmark[0].y - 0.3:
                        voice.say("Assistance requested")
                        cv2.putText(frame, "HAND RAISED", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow('Fast Smart Proctor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_smart_proctor()