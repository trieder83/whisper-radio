import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "primeline/distil-whisper-large-v3-german"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#dataset = Dataset.from_dict({"audio": ["health-german.mp3"]}).cast_column("audio", Audio())
#sample = dataset[0]["audio"]
#result = pipe(sample)
#x = librosa.load("health-german.mp3", sr=None)
x = librosa.load("health-german.mp3", sr=44000).cast_column("audio", Audio())
result = pipe(x)
print(result["text"])
