import os
import sys
import librosa
import soundfile as sf
import whisper
import subprocess

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

video_file_path = f'{directory}/input_videos/annin_lempielain_on_pupu.mp4'
cmd = f"ffmpeg -i {video_file_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {video_file_path.replace('.mp4', '.wav')} -y"
subprocess.run(cmd, shell=True, check=True, capture_output=True)


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
        diarize_model = whisperx.diarize.DiarizationPipeline(token=whisperx_token, device=device)
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