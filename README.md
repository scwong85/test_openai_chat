# 🎤📝 OpenAI Speech/Text-to-Speech Web App

This application allows users to interact with OpenAI's GPT-4o model using either speech or text input. The responses are provided both as text and synthesized speech.

## Features

- **Speech Input:** Hold the microphone button to speak.
- **Text Input:** Type your message and click "Send".
- **Real-time Responses:** Receive responses as both text and audio.

## Technologies Used

- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Backend:** FastAPI (Python)
- **APIs:** OpenAI's Whisper (Speech-to-Text), GPT-4o (Text Generation), and TTS (Text-to-Speech)

## Deployment

### Render (Backend) & Netlify (Frontend)

- Deploy the backend using the "Deploy to Render" button above.
- Deploy the frontend by uploading the `frontend/` folder to Netlify.

## Setup Instructions

1. Clone the repository.
2. Add your OpenAI API key to the environment variables as `OPENAI_API_KEY`.
3. Update the `backendUrl` in `frontend/app.js` with your deployed backend URL.
4. Run the application.

## License

This project is licensed under the MIT License.
