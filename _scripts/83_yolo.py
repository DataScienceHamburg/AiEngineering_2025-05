#%%
from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO('yolo11n.pt')
# %%
results = model('kiki_video.mp4')

# %% 
results[0].show()

# %% show the video with results
results[0].save('C:/Temp/kiki_video_results.mp4')

# %%

# %%
# Load a pretrained YOLOv11 model

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Load the YOLOv11 model

model = YOLO("yolo11n.pt")
 
# Function to process video

def process_video(input_path, output_path=None):

    # Open the video file

    video = cv2.VideoCapture(input_path)

    # Check if video opened successfully

    if not video.isOpened():

        print(f"Error: Could not open video {input_path}")

        return

    # Get video properties

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer if output path is provided

    if output_path:

        # Ensure output directory exists

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create video writer

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    else:

        out = None

    frame_count = 0

    # Process each frame

    while True:

        # Read the next frame

        success, frame = video.read()

        # Break if no more frames

        if not success:

            break

        # Increment frame counter

        frame_count += 1

        # Process frame with YOLO

        results = model(frame)

        # Visualize results on frame

        annotated_frame = results[0].plot()

        # Display progress

        print(f"Processing frame {frame_count}/{total_frames}", end="\r")

        # Write to output video if specified

        if out:

            out.write(annotated_frame)

        # Display frame (comment out for faster processing)

        # cv2.imshow("YOLOv11 Detection", annotated_frame)

        # Break if 'q' is pressed

        # if cv2.waitKey(1) & 0xFF == ord('q'):

        #     break

    # Release resources

    video.release()

    if out:

        out.release()

    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")

    if out:

        print(f"Saved output to {output_path}")
 
#%% Usage example


# Process a single image (using your existing code)


# Process video

input_video = "./kiki_video.mp4"  # Change to your input video path

output_video = "./kiki_video_results.mp4"  # Change to your desired output path

process_video(input_video, output_video)
 
# %%
