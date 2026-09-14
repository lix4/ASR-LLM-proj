import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from earnings_asr.audio import SAMPLE_RATE, load_audio, segment
from earnings_asr.inference import transcribe_file
from earnings_asr.service import create_app


class FakeBackend:
    def __init__(self, config):
        self.metadata = {"test_double": True}
        self.calls = []

    def synchronize(self):
        pass

    def transcribe_batch(self, arrays):
        self.calls.extend(arrays)
        return ["hello hello" for _ in arrays]


@pytest.fixture
def config():
    return {"chunk_seconds": 1.0, "boundary_search_seconds": 0.2,
            "silence_rms": 0.0001, "batch_size": 2}


def test_segmentation_covers_each_sample_once_and_keeps_tail():
    audio = np.ones(16000 * 5 + 31, dtype=np.float32) * 0.1
    audio[15000:15500] = 0
    chunks = list(segment(audio, 1, 0.2))
    assert chunks[0].start == 0 and chunks[-1].end == len(audio)
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))
    assert all(0 < c.end - c.start <= SAMPLE_RATE for c in chunks)
    assert np.array_equal(np.concatenate([audio[c.start:c.end] for c in chunks]), audio)


def test_silence_skips_model_and_real_repetitions_remain(tmp_path, config):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000), 16000)
    backend = FakeBackend(config)
    result = transcribe_file(backend, path, config)
    assert result["text"] == "" and not backend.calls
    sf.write(path, np.ones(32000) * 0.1, 16000)
    assert transcribe_file(backend, path, config)["text"] == "hello hello hello hello"


def test_resampling_and_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.ones((8000, 2)) * 0.1, 8000)
    assert len(load_audio(path)) == 16000


def test_api_upload_bad_audio_and_demo(config):
    app = create_app(config, FakeBackend)
    stream = io.BytesIO()
    sf.write(stream, np.ones(16000) * 0.1, 16000, format="WAV")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["status"] == "ready"
        result = client.post("/transcribe", files={"file": ("sample.wav", stream.getvalue())})
        assert result.status_code == 200
        assert result.json()["text"] == "hello hello"
        assert result.json()["segments"][0]["start"] == 0
        assert client.post("/transcribe", files={"file": ("bad.wav", b"not audio")}).status_code == 422
        assert client.post("/transcribe", files={"file": ("bad.mp3", b"bad")}).status_code == 415
