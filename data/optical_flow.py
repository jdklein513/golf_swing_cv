import numpy as np
import cv2
import os


# FUNCTIONS
# Function to create object for to write videos to output directory
def save_flow_mp4(directory, video_num, frame_size):
    # Define the codec and create VideoWriter object to save Optical Flow video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Set video location and file name
    output_filename = directory + str(video_num) + '.mp4'
    # Set output parameters
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, frame_size)

    # Return output writing location and properties
    return out


# Function create directory
def create_directory(dir):
    # Check if directory exists
    if not os.path.exists(dir):
        # Otherwise create it
        os.makedirs(dir)


# Init video number counter
video_count = 0

# MAIN ENTRY POINT
if __name__ == '__main__':

    # Create cleaner output folder
    clean_output_dir = 'videos_160_optical_flow/'
    create_directory(clean_output_dir)

    # Set path and get all files
    path = 'videos_160'

    # Get list of videos from directory
    video_file_list = sorted(os.listdir(path), key=lambda f: int(os.path.splitext(f)[0]))

    # Iterate the file list beginning with second file
    for file in video_file_list:
        # Get the file
        video_file = os.path.join(path, file)

        print("Processing:", video_file)

        # Set the video to be captured
        video = cv2.VideoCapture(video_file)

        # Read the video frames
        retrieve_bool, base_frame = video.read()

        # Set output video resolutions
        size = (base_frame.shape[1], base_frame.shape[0])
        # Function call to set output parameters
        cleaner_output_video = save_flow_mp4(clean_output_dir, video_count, size)

        # Converts frame to grayscale
        previous_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)

        # Define HSV (hue, saturation, value) color array
        hsv = np.zeros_like(base_frame)
        # Update the color array second dimension to 'white'
        hsv[..., 1] = 255

        # Define parameters for Gunnar Farneback algorithm
        feature_params = dict(pyr_scale=0.5,
                              levels=3,
                              winsize=15,
                              iterations=3,
                              poly_n=5,
                              poly_sigma=1.2,
                              flags=0)

        # Iterate video frames
        while video.isOpened():
            retrieve_bool, frame = video.read()
            if not retrieve_bool:
                print('Completed!')
                break

            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Define an optical flow object
            flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, **feature_params)

            # Calculate the magnitude and angle the vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Sets image hue (in HSV array) to the optical flow direction
            hsv[..., 0] = angle * 180 / np.pi / 2

            # Set image value (in HSV array) to normalized magnitude
            # Cleaner output
            clean_hsv = hsv.copy()
            clean_hsv[..., 2] = np.minimum(45 * magnitude, 255)
    
            # Convert HSV to RGB (BGR) representation
            clean_rgb = cv2.cvtColor(clean_hsv, cv2.COLOR_HSV2BGR)
    
            # Write video
            cleaner_output_video.write(clean_rgb)

            # Read frames at given milliseconds and listen for key press value
            key = cv2.waitKey(15) & 0xff

            # Key presses to close video or save snapshot
            if key == 27:  # "ESC" key
                break

            # Update the previous frame
            previous_frame = next_frame

        # Increment video counter
        video_count += 1

        # Deallocate memory from any GUI windows
        video.release()
        cv2.destroyAllWindows()