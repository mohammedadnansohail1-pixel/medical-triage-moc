# Getting Started Guide

A complete beginner's guide to running the Medical Triage AI system.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [System Requirements](#system-requirements)
3. [Option A: Easy Setup with Docker](#option-a-easy-setup-with-docker-recommended)
4. [Option B: Manual Setup without Docker](#option-b-manual-setup-without-docker)
5. [Using the Application](#using-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Stopping the System](#stopping-the-system)

---

## What This System Does

Medical Triage AI is an intelligent symptom checker that:
- 🩺 Asks about your symptoms through natural conversation
- 🚨 Detects emergencies and advises calling 911
- ❤️ Routes heart-related symptoms to a cardiology specialist
- 🔬 Analyzes skin conditions with image upload support
- 📋 Provides preliminary triage recommendations

**⚠️ Important:** This is for informational purposes only. Always consult a real healthcare professional for medical advice.

---

## System Requirements

### Minimum Hardware
- **RAM:** 16 GB (8 GB minimum, will be slower)
- **Storage:** 30 GB free space
- **GPU:** NVIDIA GPU with 8GB VRAM (optional, but recommended)
- **CPU:** Any modern processor (4+ cores recommended)

### Supported Operating Systems
- ✅ Ubuntu 20.04 / 22.04 / 24.04
- ✅ Windows 10/11 with WSL2
- ✅ macOS 12+ (Apple Silicon or Intel)

---

## Option A: Easy Setup with Docker (Recommended)

This is the easiest way to run the system. Docker handles everything automatically.

### Step 1: Install Docker

**On Ubuntu/Debian:**
```bash
# Update system
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add your user to docker group (so you don't need sudo)
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker --version
```

**On Windows:**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Run the installer
3. Restart your computer
4. Open Docker Desktop and wait for it to start

**On macOS:**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Drag to Applications folder
3. Open Docker Desktop and wait for it to start

### Step 2: Install NVIDIA Docker Support (GPU users only)

Skip this if you don't have an NVIDIA GPU.
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU is detected
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Step 3: Clone the Repository
```bash
# Install git if needed
sudo apt install git  # Ubuntu
# or: brew install git  # macOS

# Clone the project
git clone https://github.com/mohammedadnansohail1-pixel/medical-triage-moc.git

# Enter the project folder
cd medical-triage-moc
```

### Step 4: Start the System

**With GPU (faster, recommended):**
```bash
docker-compose up -d
```

**Without GPU (CPU only, slower):**
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### Step 5: Wait for Setup (First Time Only)

The first time you run, it will:
1. Download Docker images (~5 GB)
2. Download AI models (~5 GB)

This can take 10-30 minutes depending on your internet speed.

Check progress:
```bash
# See all containers
docker-compose ps

# Watch logs
docker-compose logs -f
```

Wait until you see all containers are "healthy":
```
NAME                      STATUS
medical-triage-api        Up (healthy)
medical-triage-ollama     Up (healthy)
medical-triage-frontend   Up
```

### Step 6: Open the Application

Open your web browser and go to:

👉 **http://localhost:3000**

You should see the Medical Triage AI chat interface!

---

## Option B: Manual Setup without Docker

If you can't use Docker, follow these steps.

### Step 1: Install Python 3.11+

**On Ubuntu:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**On macOS:**
```bash
brew install python@3.11
```

**On Windows:**
1. Download from https://www.python.org/downloads/
2. Run installer, CHECK "Add Python to PATH"
3. Open Command Prompt and verify: `python --version`

### Step 2: Install Node.js 18+

**On Ubuntu:**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
```

**On macOS:**
```bash
brew install node
```

**On Windows:**
1. Download from https://nodejs.org/
2. Run installer
3. Verify: `node --version`

### Step 3: Install Ollama (AI Model Server)

**On Ubuntu/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**On Windows:**
1. Download from https://ollama.com/download
2. Run the installer

### Step 4: Start Ollama and Download Model
```bash
# Start Ollama (runs in background)
ollama serve &

# Wait a few seconds, then download the AI model (4.7 GB)
ollama pull llama3.1:8b

# Verify it's downloaded
ollama list
```

You should see:
```
NAME            SIZE
llama3.1:8b     4.7 GB
```

### Step 5: Clone the Repository
```bash
git clone https://github.com/mohammedadnansohail1-pixel/medical-triage-moc.git
cd medical-triage-moc
```

### Step 6: Setup Backend
```bash
# Go to backend folder
cd backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR on Windows: venv\Scripts\activate

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Keep this terminal open. You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Setup Frontend (New Terminal)

Open a NEW terminal window:
```bash
# Go to frontend folder
cd medical-triage-moc/frontend

# Install dependencies
npm install

# Start development server
npm run dev -- --host
```

You should see:
```
VITE ready in 300ms
➜  Local:   http://localhost:5173/
```

### Step 8: Open the Application

Open your web browser and go to:

👉 **http://localhost:5173**

---

## Using the Application

### Basic Usage

1. **Open the app** in your browser
2. **Enter your age and sex** (optional, helps with diagnosis)
3. **Describe your symptoms** in the chat box
4. **Press Enter** or click the send button
5. **Answer follow-up questions** from the AI
6. **Receive triage recommendation**

### Example Conversations

**General symptom:**
```
You: I have a headache and feel tired
AI: I'd like to understand your symptoms better. How long have you had the headache?
You: Since yesterday morning
AI: Is the pain constant or does it come and go?
...
```

**Skin concern (with image):**
```
You: I have a rash on my arm
AI: To help assess your skin condition: How long have you had this?
You: [Click 📷 to upload a photo]
AI: [Analyzes image and provides recommendation]
```

**Emergency detection:**
```
You: I have severe chest pain and can't breathe
AI: ⚠️ MEDICAL EMERGENCY DETECTED
    Please call 911 immediately!
```

### Understanding the Interface

| Element | Meaning |
|---------|---------|
| 🏥 Header | App title and status |
| 🤖 supervisor | General symptom collection |
| 🤖 cardiology | Heart-related specialist |
| 🤖 dermatology | Skin-related specialist |
| 🤖 emergency | Emergency detected |
| ⚠️ Risk Badge | Current risk level |
| 🔄 Reset | Start new conversation |
| 📷 Button | Upload image |

### Risk Levels

| Level | Color | Meaning |
|-------|-------|---------|
| Routine | 🟢 Green | Can wait, self-care possible |
| Elevated | 🟡 Yellow | See doctor soon |
| Urgent | 🟠 Orange | See doctor today |
| Emergency | 🔴 Red | Call 911 immediately |

---

## Troubleshooting

### "Cannot connect to backend"

**Check if backend is running:**
```bash
curl http://localhost:8000/api/v1/health
```

Should return: `{"status":"healthy"}`

If not:
- Docker: `docker-compose logs api`
- Manual: Check terminal running uvicorn

### "Ollama connection refused"

**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

If not:
```bash
# Start Ollama
ollama serve &

# Or on Docker
docker-compose restart ollama
```

### "Model not found"

**Download the model:**
```bash
ollama pull llama3.1:8b
```

### Slow responses

- **CPU mode:** Responses take 5-30 seconds (normal)
- **GPU mode:** Responses take 1-3 seconds
- **First request:** Always slower (model loading)

### Out of memory

- Close other applications
- Use smaller model: `ollama pull llama3.2:3b`
- Add swap space (Linux):
```bash
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
```

### Port already in use
```bash
# Find what's using the port
lsof -i :8000  # or :3000, :5173

# Kill it
kill -9 <PID>
```

### Docker issues
```bash
# Restart everything
docker-compose down
docker-compose up -d

# Reset completely (removes all data)
docker-compose down -v
docker-compose up -d
```

---

## Stopping the System

### Docker
```bash
# Stop (keeps data)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### Manual Setup

1. Press `Ctrl+C` in the backend terminal
2. Press `Ctrl+C` in the frontend terminal
3. Stop Ollama: `pkill ollama`

---

## Getting Help

- **Issues:** https://github.com/mohammedadnansohail1-pixel/medical-triage-moc/issues
- **API Docs:** http://localhost:8000/docs (when running)

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start with Docker (GPU) |
| `docker-compose -f docker-compose.cpu.yml up -d` | Start with Docker (CPU) |
| `docker-compose down` | Stop Docker |
| `docker-compose logs -f` | View Docker logs |
| `ollama serve` | Start Ollama manually |
| `ollama pull llama3.1:8b` | Download AI model |

| URL | Service |
|-----|---------|
| http://localhost:3000 | Frontend (Docker) |
| http://localhost:5173 | Frontend (Dev) |
| http://localhost:8000 | Backend API |
| http://localhost:8000/docs | API Documentation |
| http://localhost:11434 | Ollama |

---

*Last Updated: January 2026*
