#!/home/qnguyen3/.venvs/comfyui/bin/python
"""
Video generation API on top of ComfyUI.

    ./api.py                      # serves http://127.0.0.1:8000 (open it for a simple web form)
    ./api.py --host 0.0.0.0       # reachable from other machines

    POST /jobs        {"prompt": "...", "negative_prompt": "...", "pipeline": "wan"}  -> job with status_url
    GET  /jobs/{id}   -> status, progress, and video_url once finished
    GET  /videos/{id}.mp4, /images/{id}.png
    GET  /pipelines   -> which pipelines have all their model files
    POST /jobs?wait=true blocks until the video is ready (up to --max-wait seconds).

Environment: COMFY_URL (default http://127.0.0.1:8188; a local server is started automatically),
PUBLIC_BASE_URL (what links should start with, default: the request's own host).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import threading
import time
import urllib.error
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import cartoon
import pipelines

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "jobs"
JOBS_FILE = JOBS_DIR / "jobs.json"
COMFY_URL = os.environ.get("COMFY_URL", f"http://{cartoon.HOST}:{cartoon.PORT}").rstrip("/")
COMFY_IS_LOCAL = COMFY_URL == f"http://{cartoon.HOST}:{cartoon.PORT}"
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

app = FastAPI(title="Cartoon Video API", version="1.0",
              description="Submit a prompt, get back a link to a generated video.")


# ----------------------------------------------------------------------------- comfy client

def comfy(path: str, data: dict | None = None, timeout: float = 30):
    import urllib.request
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def comfy_up() -> bool:
    try:
        comfy("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_comfy() -> None:
    if comfy_up():
        return
    if not COMFY_IS_LOCAL:
        raise RuntimeError(f"ComfyUI at {COMFY_URL} is not reachable")
    cartoon.start_server_background(cartoon.PORT, [])


# ----------------------------------------------------------------------------- jobs

class Dialogue(BaseModel):
    character_a: str = Field(..., min_length=1, max_length=200, examples=["Mia, a young woman with short red hair"])
    character_b: str = Field(..., min_length=1, max_length=200, examples=["Tom, an older man with a grey beard"])
    line_a: str = Field(..., min_length=1, max_length=400, description="what A says to B")
    line_b: str = Field(..., min_length=1, max_length=400, description="what B replies")
    scene: str = Field("", max_length=400, description="where they are")
    style: str = Field("", max_length=200, description="visual style, e.g. 'pixar style 3D animation'")


class JobRequest(BaseModel):
    prompt: str = Field("", max_length=4000, examples=["a red fox running through snow"])
    negative_prompt: str = Field("", max_length=4000)
    pipeline: Literal["wan", "ltx25", "flux2_ltx25", "flux2_image", "dialogue"] = "wan"
    dialogue: Dialogue | None = Field(None, description="required for the dialogue pipeline")
    seed: int | None = None
    seconds: float | None = Field(None, gt=0, le=20)
    width: int | None = Field(None, ge=256, le=1536)
    height: int | None = Field(None, ge=256, le=1536)
    fps: int | None = Field(None, ge=8, le=60)
    steps: int | None = Field(None, ge=1, le=100)
    cfg: float | None = Field(None, ge=0, le=30)
    two_stage: bool = False

    def check(self) -> None:
        if self.pipeline == "dialogue" and self.dialogue is None:
            raise HTTPException(422, "the dialogue pipeline needs a 'dialogue' object with character_a, character_b, line_a, line_b")
        if self.pipeline != "dialogue" and not self.prompt.strip():
            raise HTTPException(422, "prompt must not be empty")


class Job:
    def __init__(self, req: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = req.model_dump()
        self.status = "queued"            # queued | running | done | failed
        self.stage = ""
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.error: str | None = None
        self.video_path: str | None = None
        self.image_path: str | None = None
        self.segments: list[str] = []
        self.meta: dict = {}

    def to_dict(self, base: str = "") -> dict:
        d = {
            "id": self.id, "status": self.status, "stage": self.stage, "pipeline": self.request["pipeline"],
            "prompt": self.request["prompt"], "negative_prompt": self.request["negative_prompt"],
            "dialogue": self.request.get("dialogue"),
            "seed": self.request.get("seed"), "created": self.created, "started": self.started,
            "finished": self.finished, "error": self.error, "meta": self.meta,
            "status_url": f"{base}/jobs/{self.id}",
            "video_url": f"{base}/videos/{self.id}.mp4" if self.video_path else None,
            "image_url": f"{base}/images/{self.id}.png" if self.image_path else None,
        }
        if self.status == "running":
            d["progress"] = cartoon.last_progress() if COMFY_IS_LOCAL else ""
            d["elapsed"] = round(time.time() - (self.started or time.time()))
        return d


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()
QUEUE: "queue.Queue[Job]" = queue.Queue()


def save_jobs() -> None:
    JOBS_DIR.mkdir(exist_ok=True)
    with LOCK:
        data = {j.id: {**j.to_dict(), "video_path": j.video_path, "image_path": j.image_path,
                       "segments": j.segments, "request": j.request}
                for j in JOBS.values()}
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(JOBS_FILE)


def load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    for jid, d in json.loads(JOBS_FILE.read_text()).items():
        j = Job(JobRequest(**d["request"]))
        j.id = jid
        j.status = d["status"] if d["status"] in ("done", "failed") else "failed"
        j.error = d.get("error") or ("server restarted before the job finished" if j.status == "failed" and d["status"] != "failed" else None)
        j.created, j.started, j.finished = d["created"], d.get("started"), d.get("finished")
        j.video_path, j.image_path, j.meta = d.get("video_path"), d.get("image_path"), d.get("meta", {})
        j.segments = d.get("segments", [])
        JOBS[jid] = j


def run_graph(graph: dict, output_node: str, job: Job) -> list[Path]:
    """Queue one ComfyUI graph, wait for it, return the files its output node saved."""
    try:
        pid = comfy("/prompt", {"prompt": graph})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError("ComfyUI rejected the workflow: " + e.read().decode(errors="replace")[:1500])
    while True:
        time.sleep(2)
        hist = comfy(f"/history/{pid}")
        if pid in hist:
            break
    entry = hist[pid]
    if entry["status"].get("status_str") == "error":
        for m in entry["status"].get("messages", []):
            if m[0] == "execution_error":
                raise RuntimeError(f"{m[1].get('node_type')}: {m[1].get('exception_message')}")
        raise RuntimeError("ComfyUI execution failed")
    out = entry["outputs"].get(output_node, {})
    files = []
    for item in out.get("images", []) + out.get("video", []) + out.get("videos", []):
        files.append(pipelines.COMFY_DIR / "output" / item.get("subfolder", "") / item["filename"])
    if not files:
        raise RuntimeError("workflow finished but produced no file")
    return files


def free_comfy_memory() -> None:
    """Ask ComfyUI to drop every loaded model and cached output. The WSL VM only has ~8 GB of RAM,
    so models left over from a previous stage (Flux before LTX) can push it over the edge."""
    try:
        comfy("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


def run_job(job: Job) -> None:
    o = pipelines.Options(name=f"job_{job.id}", **job.request)
    job.status, job.started = "running", time.time()
    save_jobs()
    try:
        ensure_comfy()
        for i, stage in enumerate(pipelines.stages(o), 1):
            free_comfy_memory()
            if stage.get("deferred") == "ltx25_i2v":
                # hand the keyframe to LTX via ComfyUI's input folder
                src = Path(job.image_path)
                dst = pipelines.COMFY_DIR / "input" / f"job_{job.id}_keyframe.png"
                shutil.copyfile(src, dst)
                stage = pipelines.build_ltx25(o, image_file=dst.name)
            elif stage.get("deferred") == "concat":
                job.stage = f"{i}: joining {len(job.segments)} shots"
                save_jobs()
                out = pipelines.COMFY_DIR / "output" / "video" / f"job_{job.id}_dialogue.mp4"
                concat_videos([Path(p) for p in job.segments], out)
                job.video_path = str(out)
                continue
            job.stage = f"{i}: {stage['kind']} ({stage['graph'][stage['output_node']]['class_type']})"
            if stage.get("meta", {}).get("speaker"):
                job.stage += f" - {stage['meta']['speaker']} speaking"
            save_jobs()
            files = run_graph(stage["graph"], stage["output_node"], job)
            job.meta.update(stage.get("meta", {}))
            if stage["kind"] == "image":
                job.image_path = str(files[0])
            else:
                job.video_path = str(files[0])
                job.segments.append(str(files[0]))
        job.status = "done"
    except Exception as e:  # noqa: BLE001 - we want every failure to land in the job record
        job.status, job.error = "failed", str(e)[:2000]
    job.finished = time.time()
    job.stage = ""
    job.request["seed"] = o.seed
    save_jobs()


def concat_videos(parts: list[Path], out: Path) -> None:
    """Join mp4 clips (video + audio) into one file, re-encoding to H.264 / AAC 48 kHz stereo."""
    import av
    import numpy as np
    frames, audio, fps = [], [], None
    for p in parts:
        with av.open(str(p)) as c:
            v = c.streams.video[0]
            fps = fps or float(v.average_rate)
            frames += [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        with av.open(str(p)) as c:
            if c.streams.audio:
                rs = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=48000)
                for f in c.decode(audio=0):
                    audio += [r.to_ndarray() for r in rs.resample(f)]
                audio += [r.to_ndarray() for r in rs.resample(None)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), "w") as o:
        # Both streams must exist before the first mux() writes the container header; a stream
        # added afterwards has no time base and muxing fails with "Cannot rebase to zero time".
        vs = o.add_stream("libx264", rate=round(fps))
        vs.width, vs.height, vs.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        vs.options = {"crf": "18", "preset": "medium"}
        a = o.add_stream("aac", rate=48000, layout="stereo") if audio else None
        for i, img in enumerate(frames):
            fr = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
            fr.pts = i
            o.mux(vs.encode(fr))
        o.mux(vs.encode(None))
        if a is not None:
            wav = np.concatenate(audio, axis=1).astype(np.float32)
            wav = wav[:, : int(len(frames) / fps * 48000)]
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wav), format="fltp", layout="stereo")
            fr.sample_rate, fr.pts = 48000, 0
            o.mux(a.encode(fr))
            o.mux(a.encode(None))


def worker() -> None:
    while True:
        job = QUEUE.get()
        run_job(job)
        QUEUE.task_done()


# ----------------------------------------------------------------------------- http

def base_url(request: Request) -> str:
    return PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


@app.get("/health")
def health():
    return {"ok": True, "comfy_url": COMFY_URL, "comfy_up": comfy_up(), "queued": QUEUE.qsize()}


@app.get("/pipelines")
def list_pipelines():
    return pipelines.pipeline_status()


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest, request: Request, wait: bool = False, max_wait: int = 1800):
    req.check()
    missing = pipelines.missing_files(req.pipeline)
    if missing:
        raise HTTPException(409, {"error": f"pipeline {req.pipeline} is missing model files", "missing_files": missing})
    job = Job(req)
    with LOCK:
        JOBS[job.id] = job
    save_jobs()
    QUEUE.put(job)
    if wait:
        deadline = time.time() + max_wait
        while job.status in ("queued", "running") and time.time() < deadline:
            time.sleep(2)
    return JSONResponse(job.to_dict(base_url(request)), status_code=200 if wait else 202)


@app.get("/jobs")
def list_jobs(request: Request, limit: int = 50):
    with LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)[:limit]
    return [j.to_dict(base_url(request)) for j in jobs]


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.to_dict(base_url(request))


@app.api_route("/videos/{job_id}.mp4", methods=["GET", "HEAD"])
def get_video(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.video_path or not Path(job.video_path).exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(job.video_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.api_route("/images/{job_id}.png", methods=["GET", "HEAD"])
def get_image(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.image_path or not Path(job.image_path).exists():
        raise HTTPException(404, "image not ready")
    return FileResponse(job.image_path, media_type="image/png")


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Video API</title>
<style>body{font:15px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
textarea,select,input{width:100%;box-sizing:border-box;margin:.25rem 0 .75rem;padding:.5rem;font:inherit}
button{padding:.6rem 1.2rem;font:inherit;cursor:pointer}#out{margin-top:1rem;white-space:pre-wrap}
.tabs button{margin-right:.5rem;background:#eee;border:1px solid #ccc}.tabs button.on{background:#dbe7ff}
.row{display:flex;gap:1rem}.row>label{flex:1}video,img{max-width:100%;margin-top:1rem}.hide{display:none}</style></head><body>
<h2>Make a video</h2>
<div class="tabs"><button id="tab-prompt" class="on">From a prompt</button><button id="tab-dialogue">Two characters talking</button></div>
<div id="pane-prompt">
<label>Pipeline <select id="pipeline"></select></label>
<label>Prompt <textarea id="prompt" rows="3">A cheerful orange cat wearing a tiny blue scarf walks through a sunny meadow full of flowers</textarea></label>
<label>Negative prompt (optional) <textarea id="neg" rows="2"></textarea></label>
</div>
<div id="pane-dialogue" class="hide">
<div class="row"><label>Character A <input id="char_a" placeholder="Mia, a young woman with short red hair"></label>
<label>Character B <input id="char_b" placeholder="Tom, an older man with a grey beard"></label></div>
<label>A says to B <textarea id="line_a" rows="2" placeholder="Did you finish the report?"></textarea></label>
<label>B replies <textarea id="line_b" rows="2" placeholder="Almost. Give me ten more minutes."></textarea></label>
<div class="row"><label>Scene (optional) <input id="scene" placeholder="a sunny office with large windows"></label>
<label>Style (optional) <input id="style" placeholder="pixar style 3D animation"></label></div>
<p style="color:#555;font-size:.9em">Renders one LTX 2.5 shot per line with spoken audio and lip motion, then joins them. About 3 to 4 minutes per line on this machine.</p>
</div>
<button id="go">Generate</button>
<div id="out"></div><div id="media"></div>
<script>
const $=id=>document.getElementById(id); let mode='prompt';
const setMode=m=>{mode=m;$('pane-prompt').classList.toggle('hide',m!=='prompt');$('pane-dialogue').classList.toggle('hide',m!=='dialogue');
 $('tab-prompt').classList.toggle('on',m==='prompt');$('tab-dialogue').classList.toggle('on',m==='dialogue');};
$('tab-prompt').onclick=()=>setMode('prompt');$('tab-dialogue').onclick=()=>setMode('dialogue');
fetch('/pipelines').then(r=>r.json()).then(p=>{for(const [k,v] of Object.entries(p)){if(k==='dialogue')continue;const o=document.createElement('option');
 o.value=k;o.textContent=k+(v.available?'':'  (missing models)')+' - '+v.description;o.disabled=!v.available;$('pipeline').appendChild(o);}});
$('go').onclick=async()=>{$('media').innerHTML='';$('out').textContent='submitting...';
 const body=mode==='prompt'?{prompt:$('prompt').value,negative_prompt:$('neg').value,pipeline:$('pipeline').value}
  :{pipeline:'dialogue',dialogue:{character_a:$('char_a').value,character_b:$('char_b').value,line_a:$('line_a').value,line_b:$('line_b').value,scene:$('scene').value,style:$('style').value}};
 const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const job=await r.json(); if(!r.ok){$('out').textContent=JSON.stringify(job,null,1);return;}
 const poll=async()=>{const j=await (await fetch(job.status_url)).json();
  $('out').textContent=`job ${j.id}: ${j.status} ${j.stage||''} ${j.progress||''} ${j.elapsed?j.elapsed+'s':''}\\n`+(j.error||'');
  if(j.status==='done'){$('out').textContent+='\\nvideo: '+(j.video_url||'')+(j.image_url?'\\nimage: '+j.image_url:'');
   if(j.video_url)$('media').innerHTML=`<video controls autoplay loop src="${j.video_url}"></video>`;
   else if(j.image_url)$('media').innerHTML=`<img src="${j.image_url}">`;}
  else if(j.status!=='failed')setTimeout(poll,3000);};
 poll();};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def prestart_comfy() -> None:
    """Bring ComfyUI up as soon as the API starts, so the first job and the ComfyUI web UI
    (http://127.0.0.1:8188) are ready without waiting for a request."""
    try:
        ensure_comfy()
    except BaseException as e:  # noqa: BLE001 - start_server_background may sys.exit(); keep the API alive
        print(f"could not pre-start ComfyUI ({e}); it will be retried on the first job", flush=True)


@app.on_event("startup")
def startup():
    load_jobs()
    threading.Thread(target=worker, daemon=True, name="comfy-worker").start()
    if COMFY_IS_LOCAL:
        threading.Thread(target=prestart_comfy, daemon=True, name="comfy-prestart").start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", 8000)))
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
