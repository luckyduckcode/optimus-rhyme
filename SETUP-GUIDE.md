# Optimus Rhyme - Complete Setup Guide

## ✅ Current Status

### GPU Configuration
- **GPU Detected**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **GPU Access**: ✅ Working in Docker containers
- **Models**: 
  - Chat: qwen2.5:7b
  - Automation: deepseek-coder:6.7b

### Running Containers
1. ✅ `ollama-gpu` - Ollama AI service (port 11434)
2. ✅ `lightning-buffer` - FastAPI orchestration (port 8000)
3. ✅ `autovibe-ide` - Code generation UI (port 3000)
4. ✅ `ai-workspace` - Jupyter/VS Code/TensorBoard
5. ⚠️  `telegram-chatbot` - Needs ngrok tunnel

---

## 🤖 Telegram Bot Setup (N8thanbot)

### Configuration
- **Bot Token**: Configured ✅
- **Ngrok Domain**: `cityless-proactively-fermina.ngrok-free.dev`
- **Auth Token**: Configured ✅

### Step 1: Start ngrok Tunnel

You need to start ngrok in a separate terminal. Run ONE of these:

**Option A: Using the batch file I created**
```powershell
cd C:\Users\tenna\Desktop\O_R
.\start-ngrok.bat
```

**Option B: Direct ngrok command**
```powershell
ngrok http --domain=cityless-proactively-fermina.ngrok-free.dev 8443 --authtoken 35VP6U6HYAlHftHUSZStqcjAAHv_5Vq3s6SytvcAgj5Yrka5v
```

### Step 2: Verify Telegram Bot
Once ngrok is running, the bot should connect automatically. Check logs:
```powershell
docker logs -f telegram-chatbot
```

You should see: `✅ Webhook set successfully`

### Step 3: Test the Bot
1. Open Telegram
2. Search for your bot: `@N8thanbot`
3. Send `/start`

---

## 🔑 Getting Your Telegram User ID (For Access Control)

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your numeric ID (e.g., `123456789`)
3. Add it to `.env.telegram`:
   ```bash
   ADMIN_IDS=YOUR_ID_HERE
   ```
4. Restart the bot:
   ```powershell
   docker-compose restart telegram-bot
   ```

---

## 📊 Access Your Services

| Service | URL | Description |
|---------|-----|-------------|
| **Autovibe IDE** | http://localhost:3000 | AI code generation interface |
| **Lightning Buffer** | http://localhost:8000 | FastAPI orchestration API |
| **Jupyter Lab** | http://localhost:8888 | Python notebooks |
| **VS Code** | http://localhost:9000 | Browser-based VS Code |
| **TensorBoard** | http://localhost:6006 | ML visualizations |

---

## 🛠️ Useful Commands

### Check Container Status
```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Check GPU Usage
```powershell
docker exec ollama-gpu nvidia-smi
```

### View Logs
```powershell
# Telegram bot
docker logs -f telegram-chatbot

# Autovibe IDE
docker logs -f autovibe-ide

# Buffer API
docker logs -f lightning-buffer
```

### Restart Services
```powershell
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart telegram-bot
```

### Stop All Services
```powershell
docker-compose down
```

### Start All Services
```powershell
docker-compose up -d
```

---

## 🐛 Troubleshooting

### Telegram Bot Not Responding
1. **Check ngrok is running**: You should see the tunnel URL in ngrok's terminal
2. **Check logs**: `docker logs telegram-chatbot`
3. **Verify webhook**: The log should show `Starting with webhooks using ngrok host: cityless-proactively-fermina.ngrok-free.dev`

### GPU Not Being Used
```powershell
# Check if GPU is accessible
docker exec ollama-gpu nvidia-smi

# Should show your RTX 3060 with memory usage
```

### Container Won't Start
```powershell
# Check what went wrong
docker logs [container-name]

# Rebuild and restart
docker-compose build [service-name]
docker-compose up -d [service-name]
```

---

## 📝 Next Steps

1. ✅ Start ngrok tunnel (see Step 1 above)
2. ⏳ Get your Telegram User ID from @userinfobot
3. ⏳ Add your ID to `.env.telegram` for access control
4. ⏳ Restart telegram-bot after adding your ID
5. ⏳ Test the bot by sending `/start` on Telegram

---

## 🎯 What Each Service Does

- **ollama-gpu**: Runs AI models (qwen2.5:7b, deepseek-coder:6.7b) with GPU acceleration
- **lightning-buffer**: FastAPI service that orchestrates between services
- **autovibe-ide**: Web UI for AI-powered code generation
- **telegram-chatbot**: Telegram interface to interact with AI models
- **ai-workspace**: Development environment with Jupyter, VS Code, TensorBoard

---

*Configuration saved: 2025-11-21 01:50 CST*
