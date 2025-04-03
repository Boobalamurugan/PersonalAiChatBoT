# Boobalamurugan's AI Assistant

A personalized AI chatbot that simulates conversations with Boobalamurugan using natural language processing and voice capabilities. Discuss computer vision, deep learning, and AI development with an AI version of Boobalamurugan anytime.

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/Boobalamurugan/PersonalAiChatBoT.git)

## Features

- **Advanced Conversational AI**: Powered by Google's Gemini 2.0 Flash model for natural, context-aware responses with deep technical understanding
- **Seamless Voice Interaction**: 
  - Text-to-Speech using StreamElements API with ElevenLabs fallback
  - Speech-to-Text powered by AssemblyAI for accurate transcription
  - Real-time voice feedback and visual indicators
- **Technical Domain Expertise**: 
  - Computer Vision & YOLOv8
  - AWS Cloud Services
  - Deep Learning & Neural Networks
  - Real-time Systems
- **Modern UI/UX**: 
  - Responsive design with Tailwind CSS
  - Smooth animations and transitions
  - Visual feedback for all interactions
  - Mobile-friendly interface
- **Robust Architecture**:
  - Fallback mechanisms for API service disruptions
  - Graceful error handling
  - Configurable API settings
  - Scalable Flask backend

## Prerequisites

- Python 3.8+
- Flask web framework
- Internet connection for API access
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Git (for cloning repository)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Boobalamurugan/PersonalAiChatBoT.git
   cd PersonalAiChatBoT
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

5. Access the interface:
   ```
   http://127.0.0.1:5000/
   ```

## Required API Keys

Register for free/paid API keys from:
- [Google Gemini API](https://makersuite.google.com/app/apikey)
- [AssemblyAI](https://www.assemblyai.com/dashboard/signup)
- [ElevenLabs](https://elevenlabs.io/sign-up) (optional)

## Voice Features

- **Intelligent Text-to-Speech**:
  - Primary: StreamElements API for fast, reliable synthesis
  - Backup: ElevenLabs for premium quality when available
- **Advanced Speech Recognition**:
  - High-accuracy transcription with AssemblyAI
  - Real-time voice activity detection
  - Automatic punctuation and formatting
- **Intuitive Controls**:
  - One-click recording toggle
  - Visual feedback during recording
  - Automatic silence detection

## Technology Stack

### Backend
- Flask (Python 3.8+)
- Google Gemini 2.0 Flash
- AssemblyAI Speech Recognition
- StreamElements & ElevenLabs TTS

### Frontend
- HTML5 & CSS3
- Tailwind CSS
- Modern JavaScript
- Font Awesome Icons
- Animate.css

## Project Structure

```
PersonalAiChatBoT/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   └── js/
│       └── script.js     # Frontend logic
├── resume_data.json      # AI personality data
├── .env                  # API credentials
├── .env.example          # API template
└── requirements.txt      # Python dependencies
```

## Troubleshooting Guide

### Microphone Issues
- Enable microphone permissions in browser settings
- Refresh page after granting permissions
- Try using Chrome or Firefox for best compatibility

### Audio Problems
- Verify API keys are correctly configured in `.env`
- Check browser audio output settings
- Ensure stable internet connection

### API Limitations
- Monitor API usage quotas
- Implement request rate limiting
- Check server logs for error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - feel free to use and modify for your own projects.

## Author

Boobalamurugan S  
[GitHub](https://github.com/Boobalamurugan) | [LinkedIn](https://linkedin.com/boobalamurugan)
