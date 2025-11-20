# Optimus Rhyme - Docker Hub Based Multi-Container AI Workspace

This repository sets up a GPU-enabled multi-container application using Docker Compose, with images hosted on Docker Hub.

## Overview

The setup includes 4 containers:
- **ai-workspace**: Main container with JupyterLab, VS Code Server, TensorBoard, and node-nexus debugger.
- **autovibe**: AI automation API bot (Node.js/Electron).
- **lightning-buffer**: FastAPI orchestration buffer.
- **telegram-bot**: Telegram Ollama chatbot.
- **ollama**: Shared Ollama service for AI models.

## Prerequisites

- Docker and Docker Compose installed.
- NVIDIA Docker support for GPU acceleration.
- Docker Hub account (for pushing images).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/luckyduckcode/optimus-rhyme.git
cd optimus-rhyme
```

### 2. Login to Docker Hub
```bash
docker login
```

### 3. Push Images to Docker Hub
The images have been built locally. To push them to Docker Hub:
```bash
docker push luckyduckcode/optimus-rhyme:ai-workspace
docker push luckyduckcode/optimus-rhyme:autovibe
docker push luckyduckcode/optimus-rhyme:buffer
docker push luckyduckcode/optimus-rhyme:telegram
```

### 4. Configure Environment Variables
Edit the `.env` files:
- `.env.autovibe`
- `.env.buffer`
- `.env.telegram`

### 5. Pull Ollama Models (Optional)
```bash
docker compose run --rm ollama ollama pull qwen2.5:7b
docker compose run --rm ollama ollama pull deepseek-coder:6.7b
```

### 6. Start the Services
```bash
docker compose up -d
```

### 7. Access the Services
- JupyterLab: http://localhost:8888
- VS Code Server: http://localhost:9000
- TensorBoard: http://localhost:6006
- Autovibe API: http://localhost:3000
- Buffer API: http://localhost:8000

## Workflow

1. Telegram bot receives messages.
2. Routes through lightning-buffer for orchestration.
3. Buffer calls autovibe API for code generation via Ollama.
4. Results stored in shared `/workspace/automations`.
5. Debug and edit in main ai-workspace.

## Development

To rebuild images locally:
```bash
docker build -f Dockerfile.main -t luckyduckcode/optimus-rhyme:ai-workspace .
docker build -f Dockerfile.autovibe -t luckyduckcode/optimus-rhyme:autovibe .
docker build -f Dockerfile.buffer -t luckyduckcode/optimus-rhyme:buffer .
docker build -f Dockerfile.telegram -t luckyduckcode/optimus-rhyme:telegram .
```

## Troubleshooting

- Ensure NVIDIA runtime is configured.
- Check logs: `docker compose logs <service>`
- For GPU issues, verify `nvidia-docker` installation.