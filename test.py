import whisper

#model = whisper.load_model("base")
model = whisper.load_model("medium")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
#audio = whisper.load_audio("engineKGSO1-Twr-Aug-28-2024-2330Z.mp3")
audio = whisper.load_audio("ditchKORL-Twr-Aug-16-2024-1900Z.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
device = "cuda:0"# if torch.cuda.is_available() else "cpu"
#mel = whisper.log_mel_spectrogram(audio).to(model.device)
mel = whisper.log_mel_spectrogram(audio).to(device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
