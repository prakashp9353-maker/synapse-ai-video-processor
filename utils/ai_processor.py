import whisper
from transformers import pipeline
import openai
from pydub import AudioSegment
import tempfile
import os

class AIProcessor:
    def __init__(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        print("Loading summarization model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def extract_audio(self, video_path, audio_output_path="audio.wav"):
        try:
            audio = AudioSegment.from_file(video_path)
            audio.export(audio_output_path, format="wav")
            return audio_output_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def summarize_text(self, text, max_length=150, min_length=30):
        print("Generating summary...")
        
        if len(text.split()) > 1024:
            chunks = self._split_text(text)
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    
    def generate_quiz(self, transcript, num_questions=5):
        print("Generating quiz questions...")
        
        prompt = f"""
        Based on this educational content, generate {num_questions} multiple-choice questions.
        Format each question exactly as:
        Q: [question text]
        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]
        Correct: [correct letter]
        
        Content: {transcript[:3000]}
        
        Questions:
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational expert creating quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return self._parse_quiz_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self._generate_fallback_quiz(transcript)
    
    def _split_text(self, text, chunk_size=1000):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _parse_quiz_response(self, response_text):
        questions = []
        current_question = {}
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_question:
                    questions.append(current_question)
                current_question = {'question': line[2:].strip(), 'options': {}, 'correct': ''}
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                option_key = line[0]
                option_text = line[2:].strip()
                current_question['options'][option_key] = option_text
            elif line.startswith('Correct:'):
                current_question['correct'] = line[8:].strip()
        
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _generate_fallback_quiz(self, transcript):
        sentences = transcript.split('.')
        important_sentences = [s for s in sentences if len(s.split()) > 5][:5]
        
        questions = []
        for i, sentence in enumerate(important_sentences):
            if len(sentence.split()) > 3:
                words = sentence.split()
                question = f"What is the main topic of: {' '.join(words[:5])}..."
                questions.append({
                    'question': question,
                    'options': {'A': 'Technology', 'B': 'Science', 'C': 'Education', 'D': 'Business'},
                    'correct': 'C'
                })
        
        return questions
