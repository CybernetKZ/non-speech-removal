import requests


no_speech_threshold = 0.5

with open("before.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process-audio",
        files={"file": f},
        params={
            "no_speech_threshold": no_speech_threshold,
        }
    )


if response.status_code == 200:
    
    with open("after.wav", "wb") as out:
        out.write(response.content)
    print("Audio processed successfully! Saved as 'after.wav'")
else:
    print(f"Error: {response.status_code}")
    print(f"Details: {response.text}")
