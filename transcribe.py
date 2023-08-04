from flask import Flask, request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)

# Load pre-trained model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return {"error": "audio file not found in request"}, 400
    
    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    # Load audio
    audio_input, _ = sf.read(file_path)

    # Process the audio input
    input_values = processor(audio_input, return_tensors="pt", padding=True).input_values
    input_values = input_values.float()
    input_values = input_values.to(device)
    # Generate logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Compute predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the ids to obtain the transcription
    transcription = processor.decode(predicted_ids[0])
    
    # Remove temporary file
    os.remove(file_path)

    return {"transcription": transcription}

if __name__ == '__main__':
    app.run(port=5000, debug=True)
