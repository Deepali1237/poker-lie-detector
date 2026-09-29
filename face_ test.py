import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

# Load your trained emotion model
emotion_model = tf.keras.models.load_model('emotion_model.h5')
EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_face = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils
style = mp.solutions.drawing_styles

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        emotion = "No Face"
        visual_score = 0.0

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                draw.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=style.get_default_face_mesh_contours_style()
                )

            # ---- NEW: crop face and predict emotion ----
            h, w = frame.shape[:2]
            face_crop = cv2.resize(rgb, (48, 48)) / 255.0
            face_input = np.expand_dims(face_crop, axis=0)
            preds = emotion_model.predict(face_input, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = np.max(preds)

            # bluff signal: fear + disgust + surprise = suspicion
            visual_score = float(preds[2] + preds[1] + preds[5])
            visual_score = min(visual_score, 1.0)

            # show emotion on screen
            cv2.putText(frame, f"Emotion: {emotion} ({confidence:.2f})",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Bluff Score: {visual_score:.2f}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255) if visual_score > 0.5 else (0,255,0), 2)
        else:
            cv2.putText(frame, "No Face", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Micro Expression Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()