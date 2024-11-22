import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
import pandas as pd
import time

# Start the timer
start_time = time.time()

# Load prompts from CSV
prompts_df = pd.read_csv('hf_models/prompts_3.csv')                           #update prompts csv file here
prompts_list = prompts_df['prompt'].tolist()
# prompts_list = ['a panda looking into the camera', 'pink sunset over the sea']

# Iterate over each prompt
i=0
for prompt in prompts_list:
    i+=1
    print("PROMPT = ", i)
    print(f"Generating high-res video for prompt: {prompt}")

    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]     #safe low-res video

    # let's offload the text-to-image model
    pipe.to("cpu")
    del pipe

    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16, revision="refs/pr/17"
    )  # load image-to-image model

    # The VAE consumes A LOT of memory, let's make sure we run it in sliced mode
    pipe.vae.enable_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Convert the frames to uint8 format
    video = [Image.fromarray((frame * 255).astype(np.uint8)).resize((1024, 576)) for frame in video_frames]

    # and denoise it
    video_frames = pipe(prompt, video=video, strength=0.6).frames[0]

    output_video_path_XL = f"hf_models/generated_videos/zeroscopeV2_XL/{prompt.replace(' ', '_')}.mp4"
    video_path = export_to_video(video_frames, output_video_path=output_video_path_XL)
    print(f"High-res video saved at: {video_path}")

    # Clean up
    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()

print("All high-res videos generated successfully.")

# End the timer
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f'The script took {execution_time:.2f} seconds to run.')

