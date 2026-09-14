"""Single-process local offline upload service; model requests are serialized."""

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from .common import read_config
from .inference import QwenBackend, transcribe_file

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earnings Call Transcription</title>
<style>body{font:17px system-ui;max-width:800px;margin:60px auto;padding:0 20px;background:#f5f7fb;color:#182335}
button{padding:10px 18px;background:#164eaa;color:white;border:0;border-radius:6px;cursor:pointer}
pre{white-space:pre-wrap;background:white;padding:22px;border-radius:8px}small{color:#546174}</style>
<h1>Earnings Call Transcription</h1><p>Upload an English WAV or FLAC recording.</p>
<p><small>Offline transcription · up to 100 MiB and 2 hours · segment timestamps</small></p>
<form id="upload"><input name="file" type="file" accept=".wav,.flac" required>
<button>Transcribe</button></form><p id="status" role="status"></p><pre id="result"></pre>
<script>document.querySelector('form').onsubmit=async e=>{e.preventDefault();
const button=document.querySelector('button'), status=document.querySelector('#status');
button.disabled=true;status.textContent='Transcribing…';document.querySelector('#result').textContent='';
try{const r=await fetch('/transcribe',{method:'POST',body:new FormData(e.target)}),d=await r.json();
if(!r.ok)throw Error(d.detail||'Transcription failed');
status.textContent=`Completed in ${d.request_seconds.toFixed(1)} seconds`;
document.querySelector('#result').textContent=d.text+'\\n\\n'+d.segments.map(s=>`[${s.start.toFixed(2)}–${s.end.toFixed(2)}] ${s.silent?'(silence)':s.text}`).join('\\n');
}catch(e){status.textContent=e.message}finally{button.disabled=false}};</script></html>"""


def create_app(config=None, backend_factory=QwenBackend):
    config = config or read_config()

    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = await run_in_threadpool(backend_factory, config)
        app.state.lock = asyncio.Lock()
        yield

    app = FastAPI(title="Earnings ASR", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/health")
    def health():
        return {"status": "ready", "mode": "offline", "model": app.state.backend.metadata}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile):
        start = time.perf_counter()
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".wav", ".flac"}:
            await file.close()
            raise HTTPException(415, "Please upload WAV or FLAC audio")
        with tempfile.TemporaryDirectory(prefix="earnings-asr-") as temporary:
            path = Path(temporary) / ("audio" + suffix)
            try:
                total = 0
                with path.open("wb") as stream:
                    while block := await file.read(1024 * 1024):
                        total += len(block)
                        if total > 100 * 1024**2:
                            raise HTTPException(413, "Audio upload exceeds 100 MiB")
                        stream.write(block)
                queued = time.perf_counter()
                async with app.state.lock:
                    queue_seconds = time.perf_counter() - queued
                    try:
                        result = await run_in_threadpool(transcribe_file, app.state.backend, path, config, 7200)
                    except (ValueError, RuntimeError) as error:
                        import soundfile
                        if isinstance(error, (ValueError, soundfile.LibsndfileError)):
                            raise HTTPException(422, "Invalid, unsupported, empty, or overlong audio") from error
                        raise HTTPException(503, "Transcription failed; check server resources and logs") from error
                return {**result, "queue_seconds": queue_seconds,
                        "request_seconds": time.perf_counter() - start}
            finally:
                await file.close()
    return app
