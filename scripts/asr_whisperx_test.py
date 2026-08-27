import os
import sys
# ffmpeg-setup: problems while working on a Windows system
## Set a path to the necessary ffmpeg .dlls
## Force Windows and Python to discover the necessary .dlls
## Ensure that the .dlls are loaded
ffmpeg_bin_path = r"C:/ffmpeg/ffmpeg-9.0.1-full_build-shared/bin" 
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
if sys.platform == "win32":
    try:
        os.add_dll_directory(ffmpeg_bin_path)
    except Exception:
        pass

import whisperx
import torch

def get_parent_directory() -> str:
    """Get the parent directory for handling csv files.

    Returns:
        string: the path to the directory where directories for csv files are located
    """
    #create relative path for parent
    relative_parent = os.path.join(os.getcwd(), '.')

    #use abspath for absolute parent path
    return str(os.path.abspath(relative_parent)).replace('\\', '/')

directory = get_parent_directory()

with open(f'{directory}/env/hf_token.txt') as f:
    hf_token = f.read()

for video_file in os.listdir(f'{directory}/input_videos/'):
    video_file_path = f'{directory}/input_videos/{video_file}'
    # if the video already has been processed -> do nothing
    video_file = video_file.replace('.mp4', '')
    video_file = video_file+'_whisperx' # !!! HARD CODED FOR TESTS!!!
    if video_file in [i.replace('.txt', '') for i in os.listdir(f'{directory}/output_texts/')]:
        pass
    # if the video has not been processed -> get the transcription of the video
    else:
        output_file_path = f'{directory}/output_texts/{video_file+'.txt'}'

        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 4  
        compute_type = "float16" if device == "cuda" else "int8"

        # Load and read local video/audio directly
        audio = whisperx.load_audio(video_file_path)

        # Transcribe using Fin language settings
        model = whisperx.load_model("medium", device, compute_type=compute_type, language="fi")
        result = model.transcribe(audio, batch_size=batch_size)

        # Align timestamps
        model_a, metadata = whisperx.load_align_model(language_code="fi", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Diarise speakers
        diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize_model(audio)

        # Merge text transcription with speaker IDs
        final_result = whisperx.assign_word_speakers(diarize_segments, result)

        # Save the final file
        with open(output_file_path, "w", encoding="utf-8") as f:
            for segment in final_result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
                text = segment["text"].strip()
                start = segment["start"]
                end = segment["end"]
                f.write(f"[{start:05.1f}s - {end:05.1f}s] {speaker}: {text}\n")