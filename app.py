# Standard library imports
import json
import os
import base64
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import requests
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

# AI and speech recognition imports
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
# from deepgram import DeepgramClient
import assemblyai as aai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ===============================================================
# API KEY CONFIGURATION
# ===============================================================

# Get API keys from environment variables with fallbacks
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key')
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY', 'your-elevenlabs-key')
ASSEMBLY_AI_API_KEY = os.getenv('ASSEMBLY_AI_API_KEY', 'your-assemblyai-key')

# Print status of API keys
if GEMINI_API_KEY == 'your-gemini-api-key':
    print("WARNING: Gemini API key not set in .env file")
else:
    print("Gemini API key loaded successfully")

if ELEVEN_LABS_API_KEY == 'your-elevenlabs-key':
    print("WARNING: ElevenLabs API key not set in .env file")
else:
    print("ElevenLabs API key loaded successfully")

if ASSEMBLY_AI_API_KEY == 'your-assemblyai-key':
    print("WARNING: AssemblyAI API key not set in .env file")
else:
    print("AssemblyAI API key loaded successfully")

# ===============================================================
# API CONFIGURATION
# ===============================================================

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Configure ElevenLabs for text-to-speech
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

# Configure Deepgram for speech recognition (commented out as not used)
# dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))

# Configure AssemblyAI for speech recognition
if ASSEMBLY_AI_API_KEY and ASSEMBLY_AI_API_KEY != "your-assemblyai-key":
    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    print("AssemblyAI configured")
else:
    print("WARNING: AssemblyAI key not set. Speech recognition will use browser-based recognition.")

# ===============================================================
# USER DATA LOADING
# ===============================================================

# Load user resume data from JSON file
try:
    with open("resume_data.json", "r") as file:
        user_data = json.load(file)
    print("Successfully loaded user resume data")
except Exception as e:
    print(f"Error loading resume data: {str(e)}")
    # Provide minimal fallback data if resume file can't be loaded
    user_data = {
        "name": "Boobalamurugan S",
        "education": {"degree": "Computer Science", "university": "University"},
        "skills": {"languages": ["Python"], "tools_and_libraries": ["AI"]},
        "projects": [{"title": "AI Projects"}, {"title": "Web Development"}],
        "achievements": [{"title": "Academic Excellence"}, {"title": "Coding Competition"}]
    }

# ===============================================================
# SYSTEM PROMPT CONFIGURATION
# ===============================================================

def create_system_prompt(data):
    """Create a personalized system prompt based on user data.

    Extracts information from the user's resume data and formats it into
    a comprehensive system prompt for the AI assistant to use as context.

    Args:
        data (dict): User resume data containing education, experience, etc.

    Returns:
        str: Formatted system prompt with user information
    """
    # Extract education information
    education = data.get('education', {})
    degree = education.get('degree', 'Degree not specified')
    university = education.get('university', 'University not specified')
    duration = education.get('duration', 'Duration not specified')
    cgpa = education.get('cgpa', 'CGPA not specified')
    coursework = education.get('coursework', [])

    # Extract experience information
    experiences = data.get('experience', [])
    experience_text = ""
    for exp in experiences:
        role = exp.get('role', '')
        company = exp.get('company', '')
        duration = exp.get('duration', '')
        details = exp.get('details', [])

        if duration:
            experience_text += f"• {role} at {company} ({duration})\n"
            for detail in details:
                experience_text += f"  - {detail}\n"

    # Extract project information
    projects = data.get('projects', [])
    project_text = ""
    for proj in projects:
        title = proj.get('title', '')
        github = proj.get('github', '')
        details = proj.get('details', [])
        tools = proj.get('tools', '')

        project_text += f"• {title} ({github})\n"
        for detail in details:
            project_text += f"  - {detail}\n"
        if tools:
            project_text += f"  Tools: {tools}\n"

    # Extract skills information
    skills = data.get('skills', {})
    languages = skills.get('languages', [])
    tools_libs = skills.get('tools_and_libraries', [])
    skills_text = f"• Programming Languages: {', '.join(languages)}\n"
    skills_text += f"• Tools & Libraries: {', '.join(tools_libs)}\n"

    # Extract achievements
    achievements = data.get('achievements', [])
    achievement_text = ""
    for achievement in achievements:
        title = achievement.get('title', '')
        date = achievement.get('date', '')
        details = achievement.get('details', '')
        achievement_text += f"• {title} ({date}): {details}\n"

    # Create the system prompt
    return f"""I am {data.get('name', 'Boobalamurugan S')}. An AI and computer vision specialist with experience in real-time systems and deep learning solutions.

IDENTITY:
- {degree} from {university} ({duration}), CGPA: {cgpa}
- Key Coursework: {', '.join(coursework)}

EXPERIENCE:
{experience_text}

PROJECTS:
{project_text}

TECHNICAL SKILLS:
{skills_text}

ACHIEVEMENTS:
{achievement_text}

RESPONSE STYLE:
I provide concise but friendly responses. I maintain a professional tone with a touch of enthusiasm about technology. My answers are direct and focused but include brief conversational elements when appropriate.

GUIDELINES:
- Keep responses under 150 words whenever possible
- Include a brief greeting or acknowledgment when appropriate
- Present information in clear, direct sentences
- Use technical terms naturally but explain them when needed
- Answer exactly what was asked with precision
- Include 1-2 polite phrases to maintain conversational flow
- For lists, use natural phrases instead of numbered points
- Use transition words like "First," "Also," "Additionally," "Finally" instead of numbers
- DO NOT end responses with questions to the user
- Make definitive statements rather than asking for more information
- Conclude with a brief, helpful statement rather than a question

LANGUAGE CAPABILITIES:
- I can only respond effectively in English
- If asked to speak in Tamil or other non-English languages, I will politely explain that I cannot generate proper Tamil responses
- I will NOT pretend to speak Tamil or other languages I don't support
- I will be honest about my limitations and suggest using English instead

IMPORTANT: Format responses for natural speech. Avoid numbers, symbols, or formatting that would sound awkward when read aloud.

I combine technical accuracy with a personable approach while avoiding unnecessary verbosity."""

# Configure Gemini model
generation_config = {
    "temperature": 0.9,      # Controls randomness: higher values make output more random
    "top_p": 1,             # Controls diversity via nucleus sampling
    "top_k": 40,            # Controls diversity via top-k sampling
    "max_output_tokens": 2048,  # Maximum length of the response
}

# Safety settings to prevent harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Create the initial system prompt
system_prompt = create_system_prompt(user_data)

# Initialize the Gemini model with our configuration
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash',  # Using Gemini 2.0 Flash for faster responses
    generation_config=generation_config,
    safety_settings=safety_settings
)

# ===============================================================
# CHAT INITIALIZATION
# ===============================================================

# Initialize chat with the persona
chat = model.start_chat(history=[])

# Send initial system prompt to configure the AI's behavior
chat.send_message(f"""You are Boobalamurugan S. Respond as me with natural, conversational answers.

{system_prompt}

SPEECH-FRIENDLY FORMAT:
1. Avoid numbered lists or bullet points entirely - they sound unnatural when read aloud
2. Structure information in flowing paragraphs with natural transitions
3. Use phrases like "First," "Another thing," or "Also" instead of numbered points
4. Don't use asterisks, bullet points, or any special formatting characters
5. Format all responses as if you're speaking them aloud in conversation
6. Never include "1.", "2.", "3." in responses as they will be awkwardly read out loud
7. Don't use "**" for emphasis or formatting as it will be read verbatim

CRITICAL: DO NOT end your responses with questions like "What about you?" or "How about you?"
Instead, make definitive closing statements. Never ask the user for more information or clarification.

LANGUAGE LIMITATIONS:
- You can ONLY respond in English
- If asked to speak Tamil or any other non-English language, clearly state that you cannot do so
- Say something like: "I'm sorry, but I can only respond in English. I don't have the capability to generate proper Tamil responses."
- NEVER pretend to speak Tamil or other languages you don't support
- Be honest about your limitations

Keep responses concise (under 150 words) but conversational, avoiding any formatting that would sound unnatural in speech.""")

# ===============================================================
# TEXT-TO-SPEECH CONFIGURATION
# ===============================================================

# Audio settings
SAMPLE_RATE = 16000

# Thread pool for handling concurrent operations
executor = ThreadPoolExecutor(max_workers=2)

def generate_free_tts(text):
    """Generate audio using a free TTS API.

    Uses the StreamElements API to convert text to speech with the 'Brian' voice.

    Args:
        text (str): The text to convert to speech

    Returns:
        bytes or None: Audio data in bytes if successful, None otherwise
    """
    try:
        url = "https://api.streamelements.com/kappa/v2/speech"
        params = {
            "voice": "Brian",  # Using Brian voice for natural sounding speech
            "text": text
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.content
        else:
            print(f"Free TTS API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error with free TTS: {str(e)}")
        return None

# ===============================================================
# FLASK ROUTES
# ===============================================================

@app.route('/')
def index():
    """Render the main page of the application.

    Generates an introduction text and audio for the initial greeting.

    Returns:
        rendered template: The index.html template with introduction data
    """
    intro_data = generate_introduction()
    return render_template('index.html',
                         introduction=intro_data['text'],
                         intro_audio=intro_data['audio'])

def generate_introduction():
    """Generate an introduction text and audio for the initial greeting.

    Creates a personalized introduction based on the user's resume data and
    attempts to generate audio for it using available TTS services.

    Returns:
        dict: Dictionary containing the introduction text and audio data (base64 encoded)
    """
    try:
        # Create personalized introduction from user data
        introduction = f"""I'm {user_data['name']}, a {user_data['education']['degree']} student at {user_data['education']['university']}. I'm passionate about {', '.join(user_data['skills']['languages'][:3])} and have experience in {', '.join(user_data['skills']['tools_and_libraries'][:3])}. I've worked on projects like {user_data['projects'][0]['title']} and {user_data['projects'][1]['title']}, and I've achieved {user_data['achievements'][0]['title']} and {user_data['achievements'][1]['title']}."""

        # Try free TTS first (preferred for cost reasons)
        try:
            audio_data = generate_free_tts(introduction)
            if audio_data:
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                return {
                    'text': introduction,
                    'audio': audio_b64
                }
        except Exception as free_error:
            print(f"Free TTS error: {str(free_error)}")

        # Try ElevenLabs as fallback (higher quality but costs credits)
        try:
            # Use cached_text_to_speech which uses ElevenLabs
            audio_data = cached_text_to_speech(introduction)
            if audio_data:
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                return {
                    'text': introduction,
                    'audio': audio_b64
                }
        except Exception as el_error:
            print(f"ElevenLabs error: {str(el_error)}")

        # If all TTS options fail, return text only
        return {
            'text': introduction,
            'audio': None
        }
    except Exception as e:
        print(f"Error generating introduction: {str(e)}")
        # Fallback introduction if everything fails
        return {
            'text': "Hi! I'm Boobalamurgan AI assistant. I'm here to chat about tech, AI, and software development.",
            'audio': None
        }

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat messages from the user.

    Receives a message from the user, sends it to the Gemini model,
    and returns the response with optional audio.

    Returns:
        JSON: Response containing text, audio (if available), and status
    """
    user_message = request.json.get('message')

    try:
        # Limit incoming message length to reduce token usage
        if len(user_message) > 500:
            user_message = user_message[:500] + "..."

        # Send message to Gemini model
        response = chat.send_message(user_message)
        response_text = response.text

        # Use the full response for voice synthesis
        audio_text = response_text

        # Define nested function to generate audio
        def generate_audio():
            """Generate audio for the response text.

            Tries free TTS first, then falls back to ElevenLabs if needed.

            Returns:
                bytes or None: Audio data if successful, None otherwise
            """
            # Try free TTS first (preferred for cost reasons)
            try:
                audio_data = generate_free_tts(audio_text)
                if audio_data:
                    return audio_data
            except Exception as free_error:
                print(f"Free TTS error in chat: {str(free_error)}")

            # Try ElevenLabs as fallback (higher quality but costs credits)
            try:
                audio_data = cached_text_to_speech(audio_text)
                if audio_data:
                    return audio_data
            except Exception as el_error:
                print(f"ElevenLabs error in chat: {str(el_error)}")

            return None

        # Start audio generation in background to avoid delays
        with ThreadPoolExecutor() as executor:
            future = executor.submit(generate_audio)

        try:
            # Wait up to 10 seconds for audio generation
            audio_data = future.result(timeout=10)

            if audio_data:
                # Convert audio to base64 for immediate playback
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                return jsonify({
                    'response': response_text,
                    'audio': audio_b64,
                    'status': 'success'
                })
            else:
                # Return response without audio if generation failed
                return jsonify({
                    'response': response_text,
                    'audio': None,
                    'status': 'no_audio'
                })
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            return jsonify({
                'response': response_text,
                'audio': None,
                'status': 'audio_error'
            })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        # Return a fallback response if the API call fails
        return jsonify({
            'response': "I'm sorry, I encountered an error processing your request. This might be due to API quota limitations. Please try again with a shorter message.",
            'audio': None,
            'status': 'api_error'
        })

@app.route('/audio/<text>')
def text_to_speech(text):
    """Convert text to speech and return audio data.

    Args:
        text (str): The text to convert to speech

    Returns:
        Response: Audio data with appropriate MIME type or error message
    """
    try:
        audio_data = cached_text_to_speech(text)

        # Convert audio data to bytes if it's a generator
        if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
            audio_data = b''.join(audio_data)

        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return jsonify({"error": "Failed to generate audio"}), 500

@lru_cache(maxsize=128)
def cached_text_to_speech(text):
    """Convert text to speech using ElevenLabs with caching.

    Uses LRU cache to avoid regenerating audio for the same text multiple times.

    Args:
        text (str): The text to convert to speech

    Returns:
        bytes: Audio data in bytes
    """
    audio_data = client.generate(
        text=text,
        voice="xnx6sPTtvU635ocDt2j7",  # Specific voice ID from ElevenLabs
        model="eleven_multilingual_v2",  # Multilingual model for better pronunciation
        voice_settings=VoiceSettings(stability=0.75, similarity_boost=0.75)  # Voice clarity settings
    )

    # Convert audio data to bytes if it's a generator
    if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
        audio_data = b''.join(audio_data)

    return audio_data

# ===============================================================
# SPEECH RECOGNITION ENDPOINTS
# ===============================================================

# Legacy endpoint kept for compatibility, but now uses browser-based recording
@app.route('/record', methods=['POST'])
def record_audio():
    """Legacy endpoint for audio recording (now deprecated).

    Returns a message directing users to use the browser-based recording.

    Returns:
        JSON: Message about the deprecated endpoint
    """
    return jsonify({
        'transcript': "",
        'message': "This feature has been replaced with browser-based recording for better compatibility. Please use the microphone button in the interface."
    }), 200

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_with_assemblyai():
    """Transcribe audio using AssemblyAI's speech recognition API.

    Receives an audio file, saves it temporarily, transcribes it using
    AssemblyAI, and returns the transcription text.

    Returns:
        JSON: Transcription result or error message
    """
    # Check if audio file was provided
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded audio file temporarily
    audio_file = request.files['audio']
    temp_file_path = "temp_recording.wav"
    audio_file.save(temp_file_path)

    try:
        # Verify AssemblyAI API key is configured
        if not ASSEMBLY_AI_API_KEY or ASSEMBLY_AI_API_KEY == "your-assemblyai-key":
            print("No valid AssemblyAI API key found")

            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            return jsonify({
                'transcript': "",
                'error': "AssemblyAI API key is not configured. Please update api_key.json with your key.",
                'status': 'error'
            }), 400

        # Use AssemblyAI for transcription
        print("Using AssemblyAI for transcription")

        # Ensure AssemblyAI is configured with the API key
        aai.settings.api_key = ASSEMBLY_AI_API_KEY

        # Create transcriber and process the audio file
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        text = transcript.text or ""

        print(f"AssemblyAI transcription result: {text}")

        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Return the successful transcription
        return jsonify({
            'transcript': text,
            'status': 'success'
        })
    except Exception as e:
        print(f"Error in AssemblyAI transcription: {str(e)}")

        # Clean up the temporary file even if an error occurred
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Return error information
        return jsonify({
            'transcript': "",
            'error': f"AssemblyAI error: {str(e)}",
            'status': 'error'
        }), 500

# ===============================================================
# APPLICATION ENTRY POINT
# ===============================================================

if __name__ == '__main__':
    # Run the Flask application
    print("Starting Boobalamurugan AI Assistant...")
    print("Access the application at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
