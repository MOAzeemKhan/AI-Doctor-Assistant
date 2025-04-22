from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import soundfile as sf
import os
import re
import uuid

app = FastAPI()

# Load Whisper model (optimized for CPU)
model = WhisperModel("small", device="cpu", compute_type="int8")

# Entity Extraction Function
def extract_entities(text):
    patient_name = None
    disease = None
    treatment = None

    name_patterns = [
        r"my name is ([A-Za-z\s]+)",
        r"i am ([A-Za-z\s]+)",
        r"this is ([A-Za-z\s]+)"
    ]
    disease_patterns = [
        r"i have (.*?)[\.,]",
        r"suffering from (.*?)[\.,]",
        r"diagnosed with (.*?)[\.,]"
    ]
    treatment_patterns = [
        r"prescribe (.*?)[\.,]",
        r"take (.*?)[\.,]",
        r"medicine (.*?)[\.,]"
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            patient_name = match.group(1).strip()
            break

    for pattern in disease_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            disease = match.group(1).strip()
            break

    for pattern in treatment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            treatment = match.group(1).strip()
            break

    return {
        "patient_name": patient_name,
        "disease": disease,
        "treatment": treatment
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_filename = f"temp_{file_id}.wav"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Run transcription
    segments, info = model.transcribe(temp_filename, beam_size=1, vad_filter=True)
    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "

    # Clean up temp file
    os.remove(temp_filename)

    # Extract entities
    extracted = extract_entities(transcription)

    # Return response
    return {
        "transcription": transcription.strip(),
        "extracted": extracted
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
