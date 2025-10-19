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

app = FastAPI(
    title="Synapse AI Video Processor",
    description="AI-powered educational video summarizer and quiz generator",
    version="1.0.0"
)

# CORS middleware
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
    return {
        "message": "🚀 Synapse AI Video Processor API is running!",
        "endpoints": {
            "upload": "POST /upload-video/",
            "results": "GET /results/{job_id}",
            "health": "GET /health"
        },
        "status": "active"
    }

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video for processing"""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (max 25MB for free tier)
    max_size = 25 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 25MB")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        video_path = tmp_file.name
    
    # Store initial job status
    results_db[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "summary": "",
        "quiz": [],
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(process_video, job_id, video_path)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "message": "Video uploaded and processing started"
    })

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results"""
    if job_id not in results_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = results_db[job_id]
    
    if result["status"] == "processing":
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "message": "Still processing video..."
        })
    
    elif result["status"] == "completed":
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "filename": result["filename"],
            "summary": result["summary"],
            "quiz": result["quiz"],
            "message": "Processing completed successfully"
        })
    
    elif result["status"] == "error":
        return JSONResponse({
            "job_id": job_id,
            "status": "error",
            "error": result["error"],
            "message": "Processing failed"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Synapse AI Backend"}

def process_video(job_id: str, video_path: str):
    """Background task to process video using OpenAI APIs"""
    try:
        # Update status
        results_db[job_id]["status"] = "processing"
        
        # Step 1: Extract audio using ffmpeg
        print(f"[{job_id}] Extracting audio...")
        audio_path = video_path.replace('.mp4', '.wav')
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        
        # Step 2: Transcribe using OpenAI Whisper API
        print(f"[{job_id}] Transcribing audio...")
        with open(audio_path, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Step 3: Generate summary using GPT
        print(f"[{job_id}] Generating summary...")
        summary_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational expert that creates concise, informative summaries of educational content."},
                {"role": "user", "content": f"Create a clear, concise summary of this educational content:\n\n{transcript}"}
            ],
            max_tokens=300
        )
        summary = summary_response.choices[0].message.content
        
        # Step 4: Generate quiz using GPT
        print(f"[{job_id}] Generating quiz...")
        quiz_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational expert creating multiple-choice quiz questions. Format each question with Q:, A), B), C), D), and Correct: labels."},
                {"role": "user", "content": f"Create 3 multiple-choice quiz questions based on this educational content:\n\n{transcript}"}
            ],
            max_tokens=500
        )
        
        # Parse quiz questions
        quiz_text = quiz_response.choices[0].message.content
        quiz = parse_quiz_questions(quiz_text)
        
        # Store results
        results_db[job_id].update({
            "status": "completed",
            "summary": summary,
            "quiz": quiz,
            "transcript_length": len(transcript)
        })
        
        print(f"[{job_id}] Processing completed successfully!")
        
    except Exception as e:
        print(f"[{job_id}] Error: {str(e)}")
        results_db[job_id].update({
            "status": "error",
            "error": str(e)
        })
    
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")

def parse_quiz_questions(quiz_text):
    """Parse quiz questions from GPT response"""
    questions = []
    current_question = {}
    lines = quiz_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            if current_question:
                questions.append(current_question)
            current_question = {
                'question': line[2:].strip(),
                'options': {},
                'correct': ''
            }
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            option_key = line[0]
            option_text = line[2:].strip()
            current_question['options'][option_key] = option_text
        elif line.startswith('Correct:'):
            current_question['correct'] = line[8:].strip()
    
    if current_question:
        questions.append(current_question)
    
    # Ensure we have at least some questions
    if not questions:
        questions = [{
            'question': 'What was the main topic discussed?',
            'options': {'A': 'Technology', 'B': 'Science', 'C': 'Education', 'D': 'Business'},
            'correct': 'C'
        }]
    
    return questions

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
