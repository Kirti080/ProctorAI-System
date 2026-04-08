import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import pyttsx3
import threading
from datetime import datetime
import pandas as pd

# --- UI CONFIGURATION ---
st.set_page_config(page_title="ProctorAI Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-family: 'Courier New', monospace; }
    .alert-text { color: #ff4b4b; font-weight: bold; animation: blinker 1.5s linear infinite; font-size: 20px; }
    @keyframes blinker { 50% { opacity: 0; } }
    .exam-box { padding: 20px; border: 1px solid #333; border-radius: 10px; background-color: #111; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
class ProctorVoice:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 4 
    def say(self, text):
        if time.time() - self.last_alert_time > self.cooldown:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()
            self.last_alert_time = time.time()
    def _speak(self, text):
        try:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except: pass

if 'voice' not in st.session_state: st.session_state.voice = ProctorVoice()
if 'stats' not in st.session_state: 
    st.session_state.stats = {"Phone": 0, "Distracted": 0, "Spoofing": 0, "Requests": 0}
if 'logs' not in st.session_state: st.session_state.logs = []

def record_event(msg, category):
    st.session_state.stats[category] += 1
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"| {now} | {msg}")

# --- HEADER & TABS ---
t1, t2 = st.columns([4, 1])
with t1:
    st.title("🛡️ PROCTOR AI : BLUE-HORIZON")
with t2:
    run_proctoring = st.toggle("SYSTEM ACTIVE", value=False, key="active_toggle")

# ADDED STUDENT PORTAL TAB
tab_monitor, tab_student, tab_analytics = st.tabs(["🔴 LIVE MONITOR", "👨‍🎓 STUDENT PORTAL", "📊 ANALYTICS & REPORTS"])

# --- LIVE MONITOR TAB ---
with tab_monitor:
    m1, m2, m3, m4 = st.columns(4)
    met1, met2, met3, met4 = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    st.write("---")
    col_video, col_alerts = st.columns([2.5, 1])
    with col_video:
        video_placeholder = st.empty()
    with col_alerts:
        st.write("#### 📝 SESSION LOGS")
        log_view = st.empty()
        st.write("---")
        st.write("#### ⚡ SYSTEM STATUS")
        status_indicator = st.empty()
        blink_timer_view = st.empty()

# --- STUDENT PORTAL TAB (NEW) ---
with tab_student:
    st.header("Mock Examination: Computer Science 101")
    with st.container():
        st.markdown('<div class="exam-box">', unsafe_allow_html=True)
        st.write("### Question 1")
        st.write("What is the primary purpose of a Virtual Environment in Python?")
        st.radio("Select an answer:", ["To make code run faster", "To isolate project dependencies", "To connect to the internet", "To edit images"], key="q1")
        
        st.write("### Question 2")
        st.write("Which AI model is used in this project for Object Detection?")
        st.selectbox("Select an answer:", ["ResNet", "YOLOv8", "VGG16", "MobileNet"], key="q2")
        
        st.button("Submit Exam", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYTICS & REPORTS TAB (UPDATED) ---
with tab_analytics:
    st.header("Exam Integrity Report")
    df = pd.DataFrame(list(st.session_state.stats.items()), columns=['Violation', 'Count'])
    
    col_chart, col_download = st.columns([2, 1])
    with col_chart:
        if any(st.session_state.stats.values()):
            st.bar_chart(df.set_index('Violation'))
        else:
            st.info("No data collected yet.")
            
    with col_download:
        st.write("### Export Data")
        st.write("Generate a CSV file of all alerts for university records.")
        # THE REPORT GENERATOR
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Exam Report",
            data=csv,
            file_name=f"Exam_Report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            help="Click to download a CSV summary of all proctoring violations."
        )
        st.write("---")
        st.table(df)

# --- THE AI ENGINE ---
if run_proctoring:
    yolo = YOLO('yolov8n.pt')
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    hands_model = mp.solutions.hands.Hands(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    last_blink = time.time()
    f_count = 0

    while cap.isOpened() and run_proctoring:
        ret, frame = cap.read()
        if not ret: break
        f_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO (Phone)
        if f_count % 6 == 0:
            results = yolo.predict(frame, classes=[67], conf=0.5, verbose=False)
            if results[0].boxes:
                st.session_state.voice.say("Phone detected")
                record_event("CRITICAL: Prohibited Device", "Phone")
                status_indicator.markdown("<p class='alert-text'>⚠️ PHONE DETECTED</p>", unsafe_allow_html=True)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # MediaPipe
        mesh_res = face_mesh.process(image_rgb)
        hand_res = hands_model.process(image_rgb)

        if mesh_res.multi_face_landmarks:
            face = mesh_res.multi_face_landmarks[0]
            # Tilt Check
            if abs(face.landmark[33].z - face.landmark[263].z) > 0.035:
                st.session_state.voice.say("Please look at the screen")
                record_event("Student distracted", "Distracted")
            # Blink Check
            if abs(face.landmark[159].y - face.landmark[145].y) < 0.012:
                last_blink = time.time()
            time_since_blink = time.time() - last_blink
            blink_timer_view.write(f"Seconds since last blink: {int(time_since_blink)}s")
            if time_since_blink > 10:
                st.session_state.voice.say("Liveness alert")
                record_event("SPOOFING ALERT: No Blink", "Spoofing")
        else:
            status_indicator.markdown("<p style='color:orange;'>USER NOT DETECTED</p>", unsafe_allow_html=True)

        if hand_res.multi_hand_landmarks:
            if hand_res.multi_hand_landmarks[0].landmark[8].y < hand_res.multi_hand_landmarks[0].landmark[0].y - 0.3:
                st.session_state.voice.say("Assistance requested")
                record_event("Hand Raised", "Requests")

        # UI REFRESH
        final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(final_frame, use_container_width=True)
        met1.metric("PHONE DETECTION", st.session_state.stats["Phone"])
        met2.metric("GAZE ALERTS", st.session_state.stats["Distracted"])
        met3.metric("LIVENESS STATUS", "SECURE" if time_since_blink < 10 else "FAIL")
        met4.metric("PENDING REQ.", st.session_state.stats["Requests"])
        log_view.code("\n".join(st.session_state.logs[:8]))

    cap.release()
else:
    st.info("System Standby. Toggle 'SYSTEM ACTIVE' in the top right to start proctoring.")