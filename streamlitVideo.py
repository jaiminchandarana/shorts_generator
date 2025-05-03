import os
import requests
from gtts import gTTS
from moviepy.editor import *
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import cv2  
import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='moviepy')
import numpy as np
import tempfile

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.makedirs("assets/images", exist_ok=True)
os.makedirs("assets/audio", exist_ok=True)

def generate_script(topic):
    model = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate.from_template("""
    ###INSTRUCTIONS
    Generate only facts no preamble or anything.
    Use a friendly, energetic, and clear tone. Make sure the pacing fits within 60 seconds, with natural pauses between each fact about the following topic:
    {topic}
    Do not provide preamble.
    ###Facts(NO PREAMBLE):
    """)

    chain = prompt | model
    response = chain.invoke({"topic": topic})
    lines = response.content.strip().split('\n')
    return '\n'.join(lines[1:]) if len(lines) > 1 else response.content.strip()

def fetch_videos(query, num_videos=5):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={num_videos}"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise Exception(f"Pexels API error: {res.text}")

    data = res.json()
    if "videos" not in data or not data["videos"]:
        raise Exception("No videos found.")

    video_paths = []
    for i, video in enumerate(data["videos"]):
        # Pick the highest quality vertical video file
        video_files = sorted(video["video_files"], key=lambda x: x["height"], reverse=True)
        best = next((vf for vf in video_files if vf["width"] <= 1080 and vf["height"] >= 1080), video_files[0])

        video_url = best["link"]
        video_path = f"assets/images/vid_{i}.mp4"

        with open(video_path, "wb") as f:
            f.write(requests.get(video_url).content)

        video_paths.append(video_path)

    return video_paths



def generate_voice(script_text, output_path="assets/audio/voice.mp3"):
    tts = gTTS(text=script_text, lang="hi")
    tts.save(output_path)
    return output_path

def split_script_evenly(script, parts):
    words = script.strip().split()
    chunk_size = len(words) // parts
    chunks = [' '.join(words[i * chunk_size : (i + 1) * chunk_size]) for i in range(parts - 1)]
    chunks.append(' '.join(words[(parts - 1) * chunk_size:]))
    return chunks

def wrap_text(text, width=40):
    words = text.split()
    lines, line = [], []
    for word in words:
        if len(' '.join(line + [word])) <= width:
            line.append(word)
        else:
            lines.append(' '.join(line))
            line = [word]
    lines.append(' '.join(line))
    return lines

def create_video_cv2(video_paths, script_text, voice_path, output_path="youtube_short.mp4"):
    script_chunks = split_script_evenly(script_text, len(video_paths))
    audio_clip = AudioFileClip(voice_path)
    total_duration = audio_clip.duration
    per_video_duration = total_duration / len(video_paths)

    width, height = 1080, 1920
    fps = 30
    out = cv2.VideoWriter("temp_video_no_audio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for video_path, text in zip(video_paths, script_chunks):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(per_video_duration * fps)
        read_frames = 0

        wrapped_text = wrap_text(text, width=35)

        while read_frames < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            y0 = 1600
            for i, line in enumerate(wrapped_text):
                y = y0 + i * 60
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                 x = (frame.shape[1] - text_width) // 2  # Center x
                 cv2.putText(
                     frame, line, (x, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                     (255, 255, 255), 3, cv2.LINE_AA
                 )


            out.write(frame)
            read_frames += 1

        cap.release()

    out.release()

    # Combine video with audio using MoviePy (audio only)
    final = VideoFileClip("temp_video_no_audio.mp4").set_audio(audio_clip)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path

st.title("🎬 YouTube Shorts Generator from Topic")
topic = st.text_input("Enter a topic (Like., Mountain, Galaxy, Oceans)")

if st.button("Generate YouTube Short"):
    if topic:
        try:
            st.info("Generating script...")
            script = generate_script(topic)
            st.success("script generated successfully!")

            st.info("Fetching videos...")
            videos = fetch_videos(topic)

            st.info("Generating voiceover...")
            voice_path = generate_voice(script)

            st.info("Creating video...")
            video_path = create_video_cv2(videos, script, voice_path)

            st.success("Video created successfully!")
            st.video(video_path)

        except Exception as e:
            st.error(f"something went wrong: {e}")
    else:
        st.warning("Please enter a topic first.")
