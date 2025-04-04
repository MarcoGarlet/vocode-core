import queue
import whisper as openai_whisper
import numpy as np
import torch
from vocode.streaming.models.transcriber import WhisperTranscriberConfig, Transcription
from vocode.streaming.transcriber.base_transcriber import BaseThreadAsyncTranscriber


class WhisperTranscriber(BaseThreadAsyncTranscriber[WhisperTranscriberConfig]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self, config: WhisperTranscriberConfig):
        super().__init__(config)
        self.config = config
        print(f"model = {config.model}")
        self.model = openai_whisper.load_model(config.model or "tiny", device=WhisperTranscriber.device)
        self._ended = False

    def _run_loop(self):
        stream = self.generator()
        for chunk in stream:
            audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if np.all(audio_np == 0.0):
                print("[WhisperTranscriber] 💤 skip silent chunk.")
                continue
            if WhisperTranscriber.device == "cuda":
                torch.cuda.empty_cache()
            result = self.model.transcribe(
                audio_np, language=self.config.language, fp16=WhisperTranscriber.device == 'cuda'
            )

            message = result["text"]
            print(f"[WhisperTranscriber] 📝 text: {message}")
            rms = np.sqrt(np.mean(audio_np**2))
            print(f"[Chunk] 📈 Energy RMS: {rms:.6f}")
            self.produce_nonblocking(
                Transcription(message=message, confidence=1.0, is_final=True)
            )

            if self._ended:
                break

    def generator(self):
        """
        PCM stream bytes.
        """
        while not self._ended:
            chunk = self.input_janus_queue.sync_q.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self.input_janus_queue.sync_q.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

    async def terminate(self):
        self._ended = True
        await super().terminate()
