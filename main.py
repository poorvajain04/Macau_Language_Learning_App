from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil

from ai import (
    speech_to_text,
    analyze_student,
    text_to_speech
)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/chat")
async def chat(
    audio: UploadFile = File(...),
    target_language: str = Form(...)
):

    # Save uploaded audio
    with open("input.webm", "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Convert speech to text
    user_text, detected_language = speech_to_text("input.webm")

    # Get AI response
    result = analyze_student(
        user_text,
        target_language
    )

    tutor_reply = result["tutor_reply"]

    # Convert to speech
    output_file = text_to_speech(
        tutor_reply,
        target_language
    )

    return FileResponse(
        output_file,
        media_type="audio/mpeg"
    )
