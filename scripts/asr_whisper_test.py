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
    output_file = f'{audio_file.replace('.wav', '.csv')}'
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
            whisper_model = whisper.load_model(name='medium')
            whisper_transcript = whisper_model.transcribe(audio_file_path, language="fi", word_timestamps=True)

            # Pyannote diarization into a structured list: speaker_timeline
            speaker_timeline = []
            for turn, speaker in diarization.speaker_diarization:
                speaker_timeline.append({'start':turn.start, 'end':turn.end, 'speaker':speaker})
                #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

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
        
        # Convert raw floating-point seconds into human-readable MM:SS format
        #timestamp_minutes = int(w_segment_start // 60)
        #timestamp_seconds = int(w_segment_end % 60)
        #formatted_timestamp = f"[{timestamp_minutes:02d}:{timestamp_seconds:02d}]"
        
            # Structure line to conform to one segment per row requirement
            df = pd.concat([df, pd.DataFrame.from_dict({'start_time': [f'{int(w_segment_start // 60)}:{int(w_segment_start % 60)}'], 'end_time': [f'{int(w_segment_end // 60)}:{int(w_segment_end % 60)}'], 'speaker': [assigned_speaker], 'speech': [w_speech]}, orient='columns')], axis=0, ignore_index=True)
        #output_row = f"{formatted_timestamp} {assigned_speaker}: {text_content}\n"
            # Store as a csv
            df.to_csv(output_file_path, header=True, encoding='utf-8', sep=';')


#        save_file_name = audio_file.replace('.wav', '')
#        save_file_name = save_file_name+'_whisper'
#        save_file_name = save_file_name+'.txt'

# This requires matching timestamps between diarization and transcription
# For simplicity, just print both:


print(diarization)

help(diarization)

for i in diarization.serialize():
    print(i)


print("=== DIARIZATION ===")
for turn, _, speaker in diarization.serialize():
    print(f"{turn.start:.1f}s-{turn.end:.1f}s: {speaker}")

print("\n=== TRANSCRIPTION ===")
for segment in result["segments"]:
    print(f"{segment['start']:.1f}s: {segment['text']}")  

help(diarization)

#Save the diarization output to a file
with open(f'{directory}/output_texts/{save_file_name}', "w") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        f.write(f"{turn.start:.2f} - {turn.end:.2f}: Speaker {speaker}\n")

print("Speaker diarization completed.") 

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