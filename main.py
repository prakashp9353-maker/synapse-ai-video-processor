from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os
import uuid
import uvicorn
import subprocess
import tempfile
import yt_dlp
from typing import Dict, Any

app = FastAPI(
    title="Synapse AI Video Processor",
    description="AI-powered educational video summarizer and quiz generator",
    version="2.0.0"
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
        "version": "2.0.0",
        "features": ["File Upload", "YouTube URL Support"],
        "endpoints": {
            "upload": "POST /upload-video/",
            "youtube": "POST /process-youtube/",
            "results": "GET /results/{job_id}",
            "health": "GET /health"
        },
        "status": "active"
    }

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video file for processing"""
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    max_size = 25 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 25MB")
    
    job_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        video_path = tmp_file.name
    
    results_db[job_id] = {
        "status": "processing",
        "type": "file_upload",
        "filename": file.filename,
        "summary": "",
        "quiz": [],
        "error": None
    }
    
    background_tasks.add_task(process_video, job_id, video_path)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "message": "Video uploaded and processing started",
        "type": "file_upload"
    })

@app.post("/process-youtube/")
async def process_youtube(background_tasks: BackgroundTasks, youtube_url: str = Form(...)):
    """Process YouTube video from URL"""
    
    if not youtube_url or not (youtube_url.startswith('https://www.youtube.com/') or 
                              youtube_url.startswith('https://youtu.be/') or
                              youtube_url.startswith('https://youtube.com/')):
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL")
    
    job_id = str(uuid.uuid4())
    
    results_db[job_id] = {
        "status": "processing",
        "type": "youtube",
        "youtube_url": youtube_url,
        "summary": "",
        "quiz": [],
        "error": None
    }
    
    background_tasks.add_task(download_and_process_youtube, job_id, youtube_url)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "processing", 
        "message": "YouTube video download and processing started",
        "type": "youtube"
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
            "type": result.get("type", "unknown"),
            "message": "Still processing video..."
        })
    
    elif result["status"] == "completed":
        response_data = {
            "job_id": job_id,
            "status": "completed",
            "type": result.get("type", "unknown"),
            "summary": result["summary"],
            "quiz": result["quiz"],
            "message": "Processing completed successfully"
        }
        
        if result["type"] == "file_upload":
            response_data["filename"] = result["filename"]
        elif result["type"] == "youtube":
            response_data["youtube_url"] = result["youtube_url"]
            
        return JSONResponse(response_data)
    
    elif result["status"] == "error":
        return JSONResponse({
            "job_id": job_id,
            "status": "error",
            "type": result.get("type", "unknown"),
            "error": result["error"],
            "message": "Processing failed"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Synapse AI Backend", "version": "2.0.0"}

def download_and_process_youtube(job_id: str, youtube_url: str):
    """Download YouTube video and process it"""
    try:
        results_db[job_id]["status"] = "processing"
        
        # Step 1: Download YouTube video
        print(f"[{job_id}] Downloading YouTube video...")
        video_path = download_youtube_video(youtube_url, job_id)
        
        if not video_path:
            raise Exception("Failed to download YouTube video")
        
        # Step 2: Process the downloaded video
        process_video(job_id, video_path)
        
    except Exception as e:
        print(f"[{job_id}] YouTube processing error: {str(e)}")
        results_db[job_id].update({
            "status": "error",
            "error": str(e)
        })

def download_youtube_video(youtube_url: str, job_id: str) -> str:
    """Download YouTube video with robust error handling"""
    try:
        print(f"[{job_id}] Starting YouTube download: {youtube_url}")
        
        # Create temp directory for download
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, f"youtube_{job_id}_%(title)s.%(ext)s")
        
        # Simplified options that work reliably
        ydl_opts = {
            'format': 'best[height<=720]/best[height<=480]/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': True,
            'extract_flat': False,
            # Simulate a real browser
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
            },
        }
        
        downloaded_file = None
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Extract info first to check if video is accessible
                print(f"[{job_id}] Extracting video info...")
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info:
                    raise Exception("Could not get video information")
                
                # Check for common restrictions
                if info.get('age_limit', 0) > 0:
                    raise Exception("Age-restricted video - please try a different video")
                
                if info.get('availability') == 'needs_auth':
                    raise Exception("Video requires sign-in - please try a different video")
                
                print(f"[{job_id}] Video info extracted. Title: {info.get('title', 'Unknown')}")
                print(f"[{job_id}] Starting download...")
                
                # Now download the video
                ydl.download([youtube_url])
                
                # Find the actual downloaded file
                downloaded_file = ydl.prepare_filename(info)
                
                # If the file doesn't exist, look for it in temp dir
                if not os.path.exists(downloaded_file):
                    for filename in os.listdir(temp_dir):
                        if filename.startswith(f"youtube_{job_id}"):
                            downloaded_file = os.path.join(temp_dir, filename)
                            break
                
                if not downloaded_file or not os.path.exists(downloaded_file):
                    raise Exception("Downloaded file not found")
                
                print(f"[{job_id}] Successfully downloaded: {downloaded_file}")
                return downloaded_file
                
            except Exception as e:
                print(f"[{job_id}] Download attempt failed: {str(e)}")
                raise
                
    except Exception as e:
        print(f"[{job_id}] YouTube download error: {str(e)}")
        
        # Provide user-friendly error messages
        error_msg = str(e).lower()
        if any(word in error_msg for word in ['sign in', 'login', 'bot', 'auth']):
            raise Exception("This video requires sign-in verification. Please try a different YouTube video or use the file upload option.")
        elif 'age restrict' in error_msg:
            raise Exception("Age-restricted video. Please try a different educational video.")
        elif 'unavailable' in error_msg or 'private' in error_msg:
            raise Exception("Video is unavailable or private. Please try a different YouTube video.")
        else:
            raise Exception(f"YouTube download failed. Please try a different video or use file upload. Error: {str(e)}")
def get_youtube_video_info(youtube_url: str):
    """Get basic info about YouTube video before downloading"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'is_live': info.get('is_live', False),
            }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None
def process_video(job_id: str, video_path: str):
    """Process video (common function for both file upload and YouTube)"""
    try:
        results_db[job_id]["status"] = "processing"
        
        # Step 1: Extract audio using ffmpeg
        print(f"[{job_id}] Extracting audio...")
        audio_path = video_path.replace('.mp4', '.wav').replace('.webm', '.wav').replace('.mkv', '.wav')
        
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
            audio_path = video_path.replace('.mp4', '.wav').replace('.webm', '.wav').replace('.mkv', '.wav')
            if os.path.exists(audio_path):
                os.remove(audio_path)
            # Clean up YouTube download directory
            temp_dir = os.path.dirname(video_path)
            if "tmp" in temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
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
