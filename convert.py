import base64

with open("dataset\human\english\common_voice_en_14716.mp3", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

print(b64)
