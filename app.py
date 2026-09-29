import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time

# Load models
@st.cache_resource(ttl=0)
def load_models():
    emotion_model = tf.keras.models.load_model('emotion_model.h5')
    xgb_model = joblib.load('xgboost_poker.pkl')
    return emotion_model, xgb_model

emotion_model, xgb_model = load_models()

EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Page config
st.set_page_config(page_title="Poker Lie Detector", page_icon="🃏", layout="wide")

st.title("🃏 Poker Lie Detector")
st.markdown("**AI-powered real-time bluff detection using facial expressions and behavioral analysis**")
st.divider()

# Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Hand Information")
    bet_amount = st.slider("Bet Amount ($)", 0, 1000, 100)
    pot_size = st.slider("Pot Size ($)", 1, 2000, 300)
    num_raises = st.number_input("Raises this session", 0, 20, 2)
    action_time = st.slider("Decision time (seconds)", 0.0, 10.0, 2.0)
    hesitation = st.number_input("Hesitation count", 0, 10, 1)
    aggression = st.slider("Aggression factor", 0.0, 2.0, 0.5)
    run = st.toggle("Start Camera")
    st.divider()
    st.subheader("Results")
    emotion_box = st.empty()
    visual_box = st.empty()
    behavioral_box = st.empty()
    final_box = st.empty()
    verdict_box = st.empty()
    tip_box = st.empty()

with col1:
    st.subheader("Live Camera Feed")
    frame_window = st.empty()

# Microlearning tips
TIPS = {
    'Fear': "💡 Fear flash detected — player may be bluffing with a weak hand. Watch for quick bets after this.",
    'Disgust': "💡 Disgust expression — player likely unhappy with their cards. Strong fold or bluff signal.",
    'Surprise': "💡 Surprise detected — unexpected community card may have helped or hurt their hand.",
    'Angry': "💡 Anger expression — player may be on tilt. Likely to make aggressive irrational bets.",
    'Happy': "💡 Happiness detected — player likely has a strong hand. Be cautious calling their bets.",
    'Sad': "💡 Sadness signal — player may be giving up. Watch for weakness in their next action.",
    'Neutral': "💡 Neutral expression — player is experienced at masking emotions. Focus on betting patterns."
}

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found!")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        emotion = "Detecting..."
        visual_score = 0.3

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, landmarks,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_style.get_default_face_mesh_contours_style()
                )

            # Predict emotion
            face_crop = cv2.resize(rgb, (48, 48)) / 255.0
            face_input = np.expand_dims(face_crop, axis=0)
            preds = emotion_model.predict(face_input, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = float(np.max(preds))

            # Visual bluff score: fear + disgust + surprise
            visual_score = float(preds[0]*0.3 + preds[1]*0.4 +
                                  preds[2]*0.5 + preds[5]*0.3)
            visual_score = min(visual_score * 2, 1.0)

            # Draw on frame
            color = (0, 0, 255) if visual_score > 0.5 else (0, 255, 0)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Bluff: {visual_score:.2f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Behavioral score
        beh_features = np.array([[
    action_time,
    bet_amount / max(pot_size, 1),
    num_raises,
    2,
    hesitation,
    aggression,
    0
]])
        behavioral_score = float(xgb_model.predict_proba(beh_features)[0][1])

        # Fuse scores
        final_score = 0.6 * visual_score + 0.4 * behavioral_score

        # Display frame
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)

        # Display results
        emotion_box.metric("Detected Emotion", emotion)
        visual_box.metric("Visual Bluff Score", f"{visual_score:.2f}")
        behavioral_box.metric("Behavioral Score", f"{behavioral_score:.2f}")
        final_box.metric("Final Bluff Probability", f"{final_score:.2f}")

        if final_score > 0.65:
            verdict_box.error("🔴 BLUFF DETECTED")
        elif final_score > 0.4:
            verdict_box.warning("🟡 SUSPICIOUS")
        else:
            verdict_box.success("🟢 Likely Honest")

        tip_box.info(TIPS.get(emotion, "💡 Keep observing player behavior."))

        time.sleep(0.03)

    cap.release()
else:
    frame_window.image(np.zeros((480, 640, 3), dtype=np.uint8),
                      channels="RGB", use_column_width=True)
    verdict_box.info("Toggle 'Start Camera' to begin analysis")