import uvicorn
import gradio as gr
from app import demo
from fastapi import FastAPI
from src.health.health_router import health_router
from src.recognize.recognize_router import recognize_router


app = FastAPI()
app.include_router(recognize_router)
app.include_router(health_router)
gr.mount_gradio_app(app, demo, path="/")
uvicorn.run(app, host="0.0.0.0", port=8000)
