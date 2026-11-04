"""Server FastAPI per Visione A ("Occhi"):
- Stream video MJPEG con Set-of-Mark (`/stream/{source}`) e raw (`/stream/{source}/raw`)
- WebSocket con il JSON degli oggetti e telemetria (`/ws/objects`, `/ws/objects/{source}`)
- Endpoint REST per stato istantaneo, target del cane, verifica locale e bridge Grok Voice.
"""
from __future__ import annotations

import asyncio
from typing import Any

import cv2
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vision.grok.voice_tools import TOOLS as VOICE_TOOLS
from vision.grok.voice_tools import dispatch as voice_dispatch

from ..pipeline import EyePipeline
from ..verification.verifier import verify_action_local


class DogTargetRequest(BaseModel):
    target_id: int | None = None
    target_label: str | None = None


class ActionStartRequest(BaseModel):
    action: str = "pick"
    target_id: int
    destination_id: int | None = None


class VoiceToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] | str = {}


def encode_jpeg(frame_bgr, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def create_app(pipeline: EyePipeline | None = None) -> FastAPI:
    """Crea e configura l'applicazione FastAPI collegata alla `EyePipeline`."""
    if pipeline is None:
        pipeline = EyePipeline()
        pipeline.add_source("arm", mode="synthetic")
        pipeline.add_source("dog", mode="synthetic")

    app = FastAPI(
        title="Visione A - Occhi (Grok Hackathon)",
        version="0.1.0",
        description="Percezione locale: MJPEG Set-of-Mark, WebSocket JSON oggetti, calibrazione ArUco e telemetria cane.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.pipeline = pipeline
    app.state.vision_state = pipeline.state

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "sources": list(pipeline.nodes.keys()),
        }

    @app.get("/state/{source}")
    def get_source_state(source: str) -> dict:
        if source in pipeline.nodes and not pipeline._running:
            pipeline.step(source)
        payload = pipeline.state.get_payload(source)
        return payload.model_dump()

    @app.get("/frame/{source}.jpg")
    def get_marked_jpeg(source: str, raw: bool = Query(default=False)) -> Response:
        if source in pipeline.nodes and not pipeline._running:
            pipeline.step(source)
        img = pipeline.state.get_frame(source) if raw else pipeline.state.get_marked_frame(source)
        return Response(content=encode_jpeg(img), media_type="image/jpeg")

    async def _mjpeg_generator(source: str, raw: bool = False, fps: float = 20.0):
        delay = 1.0 / max(1.0, fps)
        while True:
            if source in pipeline.nodes and not pipeline._running:
                pipeline.step(source)
            img = pipeline.state.get_frame(source) if raw else pipeline.state.get_marked_frame(source)
            jpg = encode_jpeg(img)
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(delay)

    @app.get("/stream/{source}")
    async def stream_marked(source: str, fps: float = Query(default=20.0, ge=1.0, le=60.0)):
        """Stream MJPEG con riquadri e numeri Set-of-Mark per l'app web."""
        return StreamingResponse(
            _mjpeg_generator(source, raw=False, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/{source}/raw")
    async def stream_raw(source: str, fps: float = Query(default=20.0, ge=1.0, le=60.0)):
        """Stream MJPEG del frame originale pulito."""
        return StreamingResponse(
            _mjpeg_generator(source, raw=True, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/dog/target")
    def set_dog_target(req: DogTargetRequest) -> dict:
        node = pipeline.nodes.get("dog")
        if node and node.dog_navigator:
            node.dog_navigator.set_target(target_id=req.target_id, target_label=req.target_label)
        return {"ok": True, "target_id": req.target_id, "target_label": req.target_label}

    @app.post("/action/{source}/start")
    def start_action(source: str, req: ActionStartRequest) -> dict:
        if source in pipeline.nodes and not pipeline._running:
            pipeline.step(source)
        rec = pipeline.state.record_action_start(
            source=source,
            action=req.action,
            target_id=req.target_id,
            destination_id=req.destination_id,
        )
        return {
            "ok": True,
            "action": rec["action"],
            "target_id": rec["target_id"],
            "destination_id": rec["destination_id"],
            "objects_before_count": len(rec["objects_before"]),
        }

    @app.post("/action/{source}/verify_local")
    def verify_action_endpoint(source: str) -> dict:
        if source in pipeline.nodes and not pipeline._running:
            pipeline.step(source)
        last = pipeline.state.get_last_action(source)
        if last is None:
            return {"ok": False, "error": "Nessuna azione registrata per questa sorgente."}
        res = verify_action_local(
            objects_before=last["objects_before"],
            objects_after=pipeline.state.get_objects(source),
            action=last["action"],
            target_id=last["target_id"],
            destination_id=last.get("destination_id"),
        )
        return {"ok": True} | res.model_dump()

    @app.get("/voice/tools")
    def voice_tools() -> list[dict]:
        """Strumenti da mettere in session.tools di Grok Voice (così l'app non li copia a mano)."""
        return VOICE_TOOLS

    @app.post("/voice/dispatch")
    def dispatch_voice_tool(req: VoiceToolRequest) -> dict:
        """Esegue uno strumento di Grok Voice (`vision.grok.voice_tools.dispatch`) sullo stato live."""
        return voice_dispatch(req.name, req.arguments, pipeline.state)

    @app.websocket("/ws/objects")
    @app.websocket("/ws/objects/{source}")
    @app.websocket("/ws/telemetry")
    async def ws_objects(websocket: WebSocket, source: str = "arm", hz: float = 15.0):
        """WebSocket che invia il JSON continuo degli oggetti rilevati e telemetria."""
        await websocket.accept()
        delay = 1.0 / max(1.0, min(60.0, hz))
        try:
            while True:
                if source in pipeline.nodes and not pipeline._running:
                    pipeline.step(source)
                payload = pipeline.state.get_payload(source)
                await websocket.send_json(payload.model_dump())
                await asyncio.sleep(delay)
        except WebSocketDisconnect:
            return

    return app
