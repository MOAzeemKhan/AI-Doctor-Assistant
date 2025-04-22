import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# Load the model
model = WhisperModel("small", device="cpu")  # or device="cuda" if GPU available

# Audio recording parameters
samplerate = 16000  # Whisper expects 16kHz audio
blocksize = 8000    # Approx 0.5 seconds of audio at a time
channels = 1        # Mono

# Queue to collect recorded audio
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, blocksize=blocksize):
        print("\nRecording... Press Ctrl+C to stop.\n")
        while True:
            sd.sleep(1000)

def transcribe_audio():
    buffer = np.empty((0, channels), dtype=np.float32)

    while True:
        chunk = audio_queue.get()
        buffer = np.vstack((buffer, chunk))

        # If enough audio accumulated, transcribe
        if len(buffer) >= samplerate * 5:  # 5 seconds window
            audio_data = np.squeeze(buffer[:samplerate * 5])
            buffer = buffer[samplerate * 5:]

            # Save temporary
            temp_path = "temp_audio.wav"
            from scipy.io.wavfile import write
            write(temp_path, samplerate, audio_data)

            segments, info = model.transcribe(temp_path, beam_size=5)
            print("\n[Transcription]")
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            print("-" * 40)

if __name__ == "__main__":
    # Start recording and transcription in parallel
    record_thread = threading.Thread(target=record_audio)
    transcribe_thread = threading.Thread(target=transcribe_audio)

    record_thread.start()
    transcribe_thread.start()

    record_thread.join()
    transcribe_thread.join()
