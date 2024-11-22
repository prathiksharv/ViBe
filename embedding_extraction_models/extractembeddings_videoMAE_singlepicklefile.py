# import numpy as np
# import torch
# from moviepy.editor import VideoFileClip
# from transformers import VideoMAEImageProcessor, VideoMAEModel
# import pickle
# import pandas as pd

# # Initialize VideoMAE processor and model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
# video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     converted_len = int(clip_len * frame_sample_rate)
#     if converted_len >= seg_len:
#         indices = np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
#     else:
#         end_idx = np.random.randint(converted_len, seg_len)
#         start_idx = end_idx - converted_len
#         indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
#     indices = np.clip(indices, 0, seg_len - 1)
#     return indices

# def read_video_moviepy(video_clip, indices):
#     frames = []
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     video_clip.reader.close()
#     return np.stack(frames)

# def extract_and_save_embeddings(video_paths, labels, output_df_file, num_labels):
#     data = {'Video label': [], 'Video embedding': [], 'Video name': []}
#     for video_path, label in zip(video_paths, labels):
#         video_clip = VideoFileClip(video_path)
#         indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=int(video_clip.duration * video_clip.fps))
#         video_frames = read_video_moviepy(video_clip, indices)
#         inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = video_model(**inputs)
#         last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#         # Convert labels to multi-label binary vector so that it can be easily used in all models, only for RF: convert it back to multiclass label
#         label_vector = np.zeros(num_labels)
#         for l in label:
#             label_vector[l] = 1
#         data['Video label'].append(label_vector)
#         data['Video embedding'].append(last_hidden_states.tolist())
#         data['Video name'].append(video_path)
#         # Print the count of rows after 10 additions
#         if len(data['Video name']) % 10 == 0:
#             print(f"{len(data['Video name'])} videos processed!")

#     df = pd.DataFrame(data)
#     with open(output_df_file, 'wb') as f:
#         pickle.dump(df, f)

# def load_video_labels_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     video_paths = df['video_paths'].tolist()
#     labels = df['labels'].apply(eval).tolist()  # Convert string representation of lists back to lists
#     return video_paths, labels

# # Example usage
# csv_file = 'annotations.csv'
# output_df_file = 'embeddings/videoMAE_video_embeddings_annotations.pkl'
# num_labels = 5  # Total number of unique labels

# # Load video paths and labels from the CSV file
# video_paths, labels = load_video_labels_csv(csv_file)

# # Extract and save embeddings
# extract_and_save_embeddings(video_paths, labels, output_df_file, num_labels)

import os
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import VideoMAEImageProcessor, VideoMAEModel
import pickle
import pandas as pd
import logging

# Set up logging for missing videos and interruptions
logging.basicConfig(filename='VideoMAE_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VideoMAE processor and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
    indices = np.clip(indices, 0, seg_len - 1)
    return indices

def read_video_moviepy(video_clip, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    video_clip.reader.close()
    return np.stack(frames)

def extract_and_save_embeddings(video_paths, labels, output_df_file, num_labels):
    data = {'Video label': [], 'Video embedding': [], 'Video name': []}
    last_saved_index = 0
    
    try:
        for i, (video_path, label) in enumerate(zip(video_paths, labels)):
            if not os.path.exists(video_path):
                logging.warning(f"Video file not found: {video_path}")
                continue
            
            video_clip = VideoFileClip(video_path)
            indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=int(video_clip.duration * video_clip.fps))
            video_frames = read_video_moviepy(video_clip, indices)
            inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = video_model(**inputs)
            last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Convert labels to multi-label binary vector
            label_vector = np.zeros(num_labels)
            for l in label:
                label_vector[l] = 1
            data['Video label'].append(label_vector)
            data['Video embedding'].append(last_hidden_states.tolist())
            data['Video name'].append(video_path)
            
            last_saved_index = i  # Update the last processed index
            
            # Log progress after every 10 videos
            if len(data['Video name']) % 10 == 0:
                print(f"{len(data['Video name'])} videos processed!")
        
    except Exception as e:
        logging.error(f"Process terminated abruptly at video: {video_paths[last_saved_index]}")
        logging.error(f"Error: {e}")
        
    finally:
        # Save the DataFrame to pickle, even if interrupted
        df = pd.DataFrame(data)
        with open(output_df_file, 'wb') as f:
            pickle.dump(df, f)
        logging.info(f"Data saved up to video: {video_paths[last_saved_index]}")

def load_video_labels_csv(csv_file):
    df = pd.read_csv(csv_file)
    video_paths = df['video_paths'].tolist()
    labels = df['labels'].apply(eval).tolist()  # Convert string representation of lists back to lists
    return video_paths, labels

# Example usage
csv_file = 'annotations/annotations.csv'
output_df_file = 'embeddings/videoMAE_video_embeddings_annotations.pkl'
num_labels = 5  # Total number of unique labels

# Load video paths and labels from the CSV file
video_paths, labels = load_video_labels_csv(csv_file)

# Extract and save embeddings
extract_and_save_embeddings(video_paths, labels, output_df_file, num_labels)