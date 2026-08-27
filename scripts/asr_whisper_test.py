import os
import sys
import numpy as np
import pandas as pd
#import librosa
#import soundfile as sf
ffmpeg_bin_path = r"C:/ffmpeg/ffmpeg-9.0.1-full_build-shared/bin" 
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
if sys.platform == "win32":
    try:
        os.add_dll_directory(ffmpeg_bin_path)
    except Exception:
        pass

import whisper
import subprocess
import torch
from pyannote.audio import Pipeline

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

os.environ['HF_TOKEN'] = hf_token

# For tests with different models
WHISPER_MODEL = 'medium'

for video_file in os.listdir(f'{directory}/input_videos/'):
    video_file_path = f'{directory}/input_videos/{video_file}'
    # if audio has already been ripped from the video -> do nothing
    if video_file.replace('.mp4', '.wav') in os.listdir(f'{directory}/input_audios/'):
        pass
    # process the video
    else:
        audio_output_path = f'{directory}/input_audios/{video_file.replace('.mp4', '.wav')}'
        cmd = f"ffmpeg -i {video_file_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_output_path} -y"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)

for audio_file in os.listdir(f'{directory}/input_audios/'):
    audio_file_path = f'{directory}/input_audios/{audio_file}'
    output_file = f'{audio_file.replace('.wav', '')}_Whisper_{WHISPER_MODEL}.csv'
    output_file_path = f'{directory}/output_files/{output_file}'
    if output_file in os.listdir(output_file_path):
        pass
    else:
        if audio_file == 'turku-1377525844-26.8.2013.wav':

            # load and split audio
            audio, sr = librosa.load(audio_file_path, sr=16000)
            # bypass torchcodec!
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_dict = {'waveform': audio_tensor, 'sample_rate': sr}
            
            # diarise the audio
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            diarization = pipeline(audio_dict)

            # transcribe
            whisper_model = whisper.load_model(name=WHISPER_MODEL)
            whisper_transcript = whisper_model.transcribe(audio_file_path, language='fi', word_timestamps=True)

            # Pyannote diarization into a structured list: speaker_timeline
            speaker_timeline = []
            for turn, speaker in diarization.speaker_diarization:
                speaker_timeline.append({'start':turn.start, 'end':turn.end, 'speaker':speaker})

            # Store the results into a df with columns: start_time, end_time, speaker, speech
            df = pd.DataFrame(columns=['start_time','end_time','speaker','speech'], dtype='str')

            for segment in whisper_transcript['segments']:
                # Extract segment info from Whisper transcript
                w_segment_start = segment['start']
                w_segment_end = segment['end']
                w_speech = segment['text'].strip()

                assigned_speaker = "UNKNOWN_SPEAKER"
                maximum_overlap_duration = 0.0

                # Calculate which Pyannote speaker owns the most duration of this text clip
                for spk_segment in speaker_timeline:
                    intersection_start = max(w_segment_start, spk_segment['start'])
                    intersection_end = min(w_segment_end, spk_segment['end'])
                    overlap_duration = max(0.0, intersection_end - intersection_start)
                    
                    # Map speaker if they have the largest overlapping time chunk
                    if overlap_duration > maximum_overlap_duration:
                        maximum_overlap_duration = overlap_duration
                        assigned_speaker = spk_segment['speaker']
                
                # Concat each segment into a dataframe -> each segment is a row with start time and end time timestamps, speaker id and speech 
                df = pd.concat([df, pd.DataFrame.from_dict({'start_time': [f'{int(w_segment_start // 60)}:{int(w_segment_start % 60)}'], 'end_time': [f'{int(w_segment_end // 60)}:{int(w_segment_end % 60)}'], 'speaker': [assigned_speaker], 'speech': [w_speech]}, orient='columns')], axis=0, ignore_index=True)
            
            # Store the full df as a csv
            df.to_csv(output_file_path, header=True, encoding='utf-8', sep=';')
