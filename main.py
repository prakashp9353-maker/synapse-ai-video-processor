from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os
import uuid
import uvicorn
import subprocess
import tempfile
from typing import Dict, Any

app = FastAPI(title="Synapse AI Video Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
results_db: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "🚀 Synapse AI Video Processor API is running!", "status": "active"}

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        video_path = tmp_file.name
    
    results_db[job_id] = {"status": "processing", "filename": file.filename}
    background_tasks.add_task(process_video, job_id, video_path)
    
    return JSONResponse({"job_id": job_id, "status": "processing", "message": "Video uploaded"})

def process_video(job_id: str, video_path: str):
    try:
        # Extract audio using ffmpeg
        audio_path = video_path.replace('.mp4', '.wav')
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1', audio_path, '-y']
        subprocess.run(cmd, check=True)
        
        # Transcribe using OpenAI Whisper API
        with open(audio_path, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            ).text
        
        # Generate summary using GPT
        summary = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational assistant that creates concise summaries."},
                {"role": "user", "content": f"Summarize this educational content: {transcript}"}
            ]
        ).choices[0].message.content
        
        # Generate quiz using GPT
        quiz_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Create multiple-choice quiz questions."},
                {"role": "user", "content": f"Create 3 quiz questions from: {transcript}"}
            ]
        ).choices[0].message.content
        
        results_db[job_id].update({
            "status": "completed",
            "summary": summary,
            "quiz_text": quiz_response,
            "transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript
        })
        
    except Exception as e:
        results_db[job_id].update({"status": "error", "error": str(e)})
    finally:
        # Cleanup
        for path in [video_path, video_path.replace('.mp4', '.wav')]:
            if os.path.exists(path):
                os.remove(path)

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in results_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(results_db[job_id])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
