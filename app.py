import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

st.set_page_config(page_title="Track Coach AI", layout="wide")
st.title("🏃‍♂️ Track Coach AI")
st.subheader("AI-Powered Form Analysis for Track & Field")

st.markdown("Upload a video of your sprint, jump, or throw and get AI coaching feedback.")

uploaded_file = st.file_uploader("Upload video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    st.video(uploaded_file)

    if st.button("🔍 Analyze with AI Coach"):
        try:
            with st.spinner("Processing video with MediaPipe Pose..."):
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
                mp_drawing = mp.solutions.drawing_utils

                cap = cv2.VideoCapture(video_path)
                analyzed_frame = None

                # Process only a few frames for speed
                for i in range(30):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    if results.pose_landmarks and i == 15:  # Take middle frame
                        annotated = frame.copy()
                        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        analyzed_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                cap.release()

                if analyzed_frame is not None:
                    st.image(analyzed_frame, caption="Pose Analysis", use_column_width=True)
                else:
                    st.warning("Could not detect pose clearly. Try better lighting or side view.")

                st.success("✅ Analysis Complete!")

                # Feedback
                event = st.selectbox("Select Event", ["Sprint", "Long Jump", "High Jump", "Shot Put", "Other"])
                
                feedbacks = {
                    "Sprint": "• Focus on high knee lift and powerful arm drive.\n• Maintain tall posture with slight forward lean.\n• Work on quick foot turnover.",
                    "Long Jump": "• Keep speed through takeoff.\n• Strong penultimate step.\n• Extend in the air and land with full reach.",
                    "High Jump": "• Smooth curved approach.\n• Explosive knee drive at takeoff.\n• Arch over the bar.",
                    "Shot Put": "• Powerful hip rotation.\n• Push from neck with full arm extension.",
                    "Other": "Tell me the event for more specific advice!"
                }
                
                st.subheader("💡 Coach Feedback")
                st.markdown(feedbacks.get(event, feedbacks["Other"]))

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Try a shorter video or different format.")

        finally:
            try:
                os.unlink(video_path)
            except:
                pass
else:
    st.info("👆 Please upload a video to begin analysis.")

st.caption("Track Coach AI • Powered by MediaPipe")
