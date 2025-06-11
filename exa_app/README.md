# Minecraft RAG Assistant

A desktop application for Minecraft players to ask questions about the game. The app uses a local RAG (Retrieval-Augmented Generation) server to provide accurate answers based on Minecraft knowledge.

## Features

- **Supabase Authentication**: Secure login and signup functionality
- **Game Selection**: Choose Minecraft (expandable to other games in the future)
- **Dual Interaction Modes**: Chat and Voice interfaces
- **RAG-powered Responses**: Accurate answers using local LLM and vector search
- **Customizable Settings**: Dark/Light theme and window transparency options

## Requirements

- Node.js (v14+)
- Python (v3.8+) with pip
- CUDA-compatible GPU recommended for optimal performance

## Installation

1. Clone the repository
2. Install Node.js dependencies:
   ```
   npm install
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Development Mode

1. Start the RAG server:
   ```
   npm run start-server
   ```
2. In a separate terminal, start the Electron app:
   ```
   npm start
   ```
   
Or use the provided `start.bat` script to launch both simultaneously:
```
start.bat
```

### Building for Distribution

Build for your current platform:
```
npm run build
```

Platform-specific builds:
```
npm run build:win
npm run build:mac
npm run build:linux
```

## Project Structure

- `main.js` - Electron main process
- `preload.js` - Secure context bridge for renderer processes
- `login.html` - Authentication screen
- `index.html` - Game selection screen
- `mode-select.html` - Mode selection screen
- `chat-mode.html` - Chat interface
- `voice-mode.html` - Voice interface
- `settings.html` - Settings screen
- `src/` - Source code
  - `supabaseClient.js` - Supabase authentication
  - `localStorage.js` - Settings and state management
  - `navigation.js` - Navigation utilities
  - `apiClient.js` - RAG API communication
  - `errorHandler.js` - Error handling and logging
  - `styles.css` - Global styles
  - `rag_server.py` - Python RAG server

## Configuration

### Supabase Setup

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Enable Email/Password authentication
3. Copy your Supabase URL and anon key
4. Update the values in `src/supabaseClient.js`

### RAG Server Configuration

The RAG server uses the following models by default:
- Embedding: `jhgan/ko-sbert-nli`
- Reranker: `Dongjin-kr/ko-reranker`
- LLM: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`

To use different models, modify the constants in `src/rag_server.py`.

## License

MIT
