import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

st.set_page_config(page_title="Track Coach AI", layout="wide")
st.title("🏃‍♂️ Track Coach AI")
st.subheader("AI-Powered Form Analysis for Track & Field Athletes")

st.markdown("""
Upload a video of your track and field performance (sprint, jump, throw, etc.), and our AI coach will analyze your form and provide improvement suggestions.
""")

# Upload video
uploaded_file = st.file_uploader("Upload your video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()
    
    st.video(uploaded_file)
    
    if st.button("Analyze Video with AI Coach"):
        with st.spinner("Analyzing form using MediaPipe Pose... This may take a moment."):
            # Initialize MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils
            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            analyzed_frames = []
            landmarks_list = []
            
            progress_bar = st.progress(0)
            
            for i in range(min(100, frame_count)):  # Process limited frames for demo speed
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks_list.append(results.pose_landmarks)
                    # Draw landmarks
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    analyzed_frames.append(annotated_frame)
                
                progress_bar.progress((i + 1) / min(100, frame_count))
            
            cap.release()
            
            st.success("Analysis Complete!")
            
            # Display sample analyzed frame
            if analyzed_frames:
                st.image(cv2.cvtColor(analyzed_frames[0], cv2.COLOR_BGR2RGB), caption="Sample Analyzed Frame with Pose Landmarks", use_column_width=True)
            
            # Event selector
            event_type = st.selectbox("What event is this video for?", ["Sprint", "Long Jump", "High Jump", "Shot Put", "Other"])
            
            st.subheader("🧠 AI Coach Feedback")
            
            feedback = {
                "Sprint": """
                - **Arm Drive:** Keep elbows \~90°, pump arms back powerfully and drive forward.
                - **Knee Lift & Stride:** Higher knee drive and quick turnover for better stride length/power.
                - **Posture:** Stay tall with a slight forward lean from the ankles (not waist).
                - **Start/Acceleration:** Explosive first steps — focus on ground force.
                - **Next Steps:** Film starts separately and incorporate hill sprints or resisted runs.
                """,
                "Long Jump": """
                - **Approach:** Consistent speed into the board.
                - **Takeoff:** Strong penultimate step, explosive plant, aim for optimal takeoff angle (\~20°).
                - **Flight:** Hang or hitch-kick technique to maximize distance.
                - **Landing:** Full leg extension and reach forward in the pit.
                - **Drill Idea:** Practice bounding and single-leg hops.
                """,
                "High Jump": """
                - **Approach:** Smooth J-curve run-up with good speed.
                - **Takeoff:** Firm foot plant, aggressive knee drive upward.
                - **Bar Clearance:** Arch back (Fosbury Flop) and lift hips over the bar.
                - **Recommendation:** Work on approach consistency and vertical power.
                """,
                "Shot Put": """
                - **Glide/Spin Technique:** Powerful hip and shoulder rotation/transfer.
                - **Delivery:** Push from the neck, full arm extension, and wrist snap.
                - **Stance:** Solid base and explosive leg drive.
                """,
                "Other": "Select a specific event above or describe it in more detail for tailored advice!"
            }
            
            st.markdown(feedback.get(event_type, feedback["Other"]))
            
            st.info("**Note:** This demo uses MediaPipe for pose detection. Real-world accuracy improves with more frames, better lighting, side/profile views, and event-specific training data. For production, we could add advanced models (e.g., via Torch) or integrate with coaching datasets.")
            
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
else:
    st.info("👆 Upload a video to get started with AI analysis!")

st.sidebar.markdown("### About Track Coach AI")
st.sidebar.write("Helps track & field athletes improve technique using AI pose estimation.")
st.sidebar.write("Technologies: Streamlit + OpenCV + MediaPipe")
st.sidebar.write("Future enhancements: Multi-angle support, slow-motion overlay, performance metrics (e.g., stride rate, joint angles).")
