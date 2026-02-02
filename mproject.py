import cv2
import os

# =========================
# CONFIGURATION
# =========================
video_path = "videoplayback.mp4"      # Path to your video
output_folder = "grayscale_frames"  # Folder to save grayscale frames
frames_to_extract = 40              # Change to None to extract ALL frames

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs(output_folder, exist_ok=True)

# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"🎥 Video FPS: {fps}")
print(f"🎞 Total Frames: {total_frames}")

# =========================
# FRAME EXTRACTION LOGIC
# =========================
interval = 1
if frames_to_extract:
    interval = max(total_frames // frames_to_extract, 1)

frame_index = 0
saved_count = 0

# =========================
# PROCESS VIDEO
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract at fixed interval
    if frame_index % interval == 0:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save grayscale frame
        filename = f"gray_frame_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), gray_frame)

        saved_count += 1

        if frames_to_extract and saved_count >= frames_to_extract:
            break

    frame_index += 1

cap.release()

print(f"✅ Successfully extracted {saved_count} grayscale frames")
