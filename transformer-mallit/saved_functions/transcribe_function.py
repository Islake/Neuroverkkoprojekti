import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_with_whisper(audio_path, model_size="small"):
    # Määrittele laite
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "mps" if torch.backends.mps.is_available() else device

    device = "cpu"
    print(f"Using device: {device}")

    # Lataa malli
    print(f"Loading Whisper {model_size} model...")
    model_name = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    # Lataa ääni
    y, sr = librosa.load(audio_path, sr=16000)

    # Preprosessi
    input_features = processor(
        y, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="fi", task="transcribe")

    # Transkriptio
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription