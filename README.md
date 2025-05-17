# Deepfake Video Dataset Processing

## Description

This project processes video datasets (real and fake) to extract faces using the InsightFace RetinaFace model with GPU batch processing. It's designed to efficiently handle large video datasets for deepfake detection research or related applications. The script extracts a specified number of frames from each video, detects faces in those frames, crops them, resizes them, and saves them into an organized directory structure.

## Features

* **Video Processing**: Handles common video formats (MP4, MOV, AVI, MKV).
* **Face Detection**: Utilizes the InsightFace `buffalo_l` model for robust face detection.
* **Batch Processing**: Processes frames in batches for improved GPU utilization and speed.
* **Frame Sampling**: Extracts an evenly spaced, configurable number of frames per video using PyAV.
* **Face Cropping and Resizing**: Crops detected faces and resizes them to a consistent output size.
* **Organized Output**: Saves extracted faces in a structured directory: `extracted_faces/{label}/{video_name}/frame_{frame_idx}_face_{face_idx}.jpg`.
* **GPU Acceleration**: Leverages CUDA for accelerated face detection if a compatible GPU is available. Falls back to CPU if not.
* **Configuration**: Key parameters like paths, frame count, output size, and batch size are configurable.
* **Environment Handling**: Includes logic to automatically zip the output when running in Kaggle or Colab environments.

## Dependencies

The primary dependencies are:

* `insightface`
* `onnxruntime-gpu` (or `onnxruntime` for CPU)
* `numpy`
* `opencv-python`
* `tqdm`
* `torch`
* `av`

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required Python packages:**
    The notebook installs dependencies using pip:
    ```python
    !pip install insightface
    !pip install onnxruntime-gpu  # or onnxruntime if you're on CPU
    !pip install numpy opencv-python tqdm torch av
    ```
    You can create a `requirements.txt` file with the following content and install using `pip install -r requirements.txt`:
    ```
    insightface
    onnxruntime-gpu # or onnxruntime
    numpy
    opencv-python
    tqdm
    torch
    av
    ```

## Usage

1.  **Configure Parameters:**
    Modify the configuration variables at the beginning of the script (`notebookbde5920a27.ipynb`) to suit your dataset and environment:
    ```python
    REAL_PATH = "/path/to/your/real/videos"  # Path to real videos
    FAKE_PATH = "/path/to/your/fake/videos"  # Path to fake videos
    OUTPUT_BASE = "extracted_faces"          # Directory to store extracted faces
    OUTPUT_FACE_SIZE = (200, 200)            # Size to resize cropped face images
    FRAME_COUNT = 15                         # Number of frames to extract per video
    MAX_VIDEOS = 200                         # Limit number of videos to process per category (real/fake)
    BATCH_SIZE = 8                           # Number of frames to process in one batch
    ```

2.  **Run the script:**
    Execute the Jupyter notebook cells or run the equivalent Python script. The main execution starts with:
    ```python
    if __name__ == '__main__':
        main() #
    ```

## Configuration Details

* `REAL_PATH`: Path to the directory containing real videos.
* `FAKE_PATH`: Path to the directory containing fake videos.
* `OUTPUT_BASE`: The root directory where extracted faces will be saved. Subdirectories for 'real' and 'fake' videos, and then further subdirectories for each video, will be created.
* `OUTPUT_FACE_SIZE`: A tuple `(width, height)` specifying the dimensions to which cropped faces will be resized.
* `FRAME_COUNT`: The number of frames to sample from each video. Frames are selected at evenly spaced intervals.
* `MAX_VIDEOS`: The maximum number of videos to process from each of the `REAL_PATH` and `FAKE_PATH` directories. This is useful for testing or quick runs on large datasets.
* `BATCH_SIZE`: The number of frames to group together for batch processing by the face detector. Adjust based on GPU memory.
* `USE_GPU`: Automatically determined by checking `torch.cuda.is_available()`.
* `DEVICE_ID`: Set to `0` for GPU or `-1` for CPU, used by InsightFace.

## Example Extracted Faces

Below are some examples of faces that can be extracted by this script. The script would save these into a structured directory like `extracted_faces/real/video_name/` or `extracted_faces/fake/video_name/`.

**Real Faces:**

<table>
  <tr>
    <td><img src= "![frame_0003_face_00](https://github.com/user-attachments/assets/ecde24c9-9cd6-4967-8c24-22dd7225afa6) alt = Real Face Example 1" width="200"/></td>
    <td><img src= ![frame_0004_face_00](https://github.com/user-attachments/assets/f7363a02-d93f-4c6e-82d0-aa2e268121f8) alt="Real Face Example 2" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Example Real Face 1</td>
    <td align="center">Example Real Face 2</td>
  </tr>
</table>

**Deepfake Faces:**

<table>
  <tr>
    <td><img src= ![frame_0005_face_00](https://github.com/user-attachments/assets/192072f4-1d64-432e-b6ff-c4d37942f44c)
  alt="Deepfake Face Example 1" width="200"/></td>
    <td><img src= ![frame_0004_face_00](https://github.com/user-attachments/assets/9cdb095b-49a5-4fcf-8fd5-9f1c0c57588b)
 alt="Deepfake Face Example 2" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Example Deepfake Face 1</td>
    <td align="center">Example Deepfake Face 2</td>
  </tr>
</table>

*(Note: For the above images to display in your GitHub README, ensure the image files (`frame_0001_face_00.jpg`, `frame_0002_face_00.jpg`, `frame_0004_face_00.jpg`, `frame_0005_face_00.jpg`) are present in the root of your repository or update the `src` paths accordingly if you place them in a subdirectory like `docs/images/`.)*
