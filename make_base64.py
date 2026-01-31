import base64

input_path = "test.mp3"
output_path = "audio_base64.txt"

with open(input_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

with open(output_path, "w") as out:
    out.write(b64)

print("Base64 saved to audio_base64.txt")
