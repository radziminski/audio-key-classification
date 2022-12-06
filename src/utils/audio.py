import contextlib
from audiofile import AudioFile
from pydub import AudioSegment


def split_to_intervals(interval_length, input_filename, output_filename_prefix):
    # Open the audio file
    with contextlib.closing(AudioFile(input_filename)) as audio_file:
        # Get the audio parameters
        num_channels = audio_file.channels
        sample_width = audio_file.sample_width
        frame_rate = audio_file.frame_rate
        num_frames = audio_file.frames

        # Calculate the duration of the audio file in seconds
        duration = num_frames / frame_rate

        # Calculate the number of 20-second intervals
        num_intervals = duration // interval_length

        # Loop through each 20-second interval
        for i in range(num_intervals):
            # Calculate the starting and ending frame for this interval
            start_frame = i * 20 * frame_rate
            end_frame = start_frame + (20 * frame_rate)

            # Read the frames from the audio file
            frames = audio_file.read_frames(end_frame - start_frame)

            # Convert the frames to a pydub.AudioSegment
            audio = AudioSegment(
                data=frames,
                sample_width=sample_width,
                channels=num_channels,
                frame_rate=frame_rate,
            )

            # Save the interval as an MP3 file
            audio.export(f'interval{i}.mp3', format='mp3')
