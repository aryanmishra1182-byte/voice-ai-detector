import base64
import tempfile
from pydub import AudioSegment

def base64_to_wav(base64_string):
    # Remove whitespace and newlines
    base64_string = "".join(base64_string.split())

    # Fix padding
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)

    audio_bytes = base64.b64decode(base64_string)

    # Save MP3
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
        mp3_file.write(audio_bytes)
        mp3_path = mp3_file.name

    # Convert MP3 → WAV
    wav_path = mp3_path.replace(".mp3", ".wav")
    sound = AudioSegment.from_file(mp3_path, format="mp3")
    sound.export(wav_path, format="wav")

    return wav_path
