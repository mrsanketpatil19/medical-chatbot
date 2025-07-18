# Medical Chatbot - AI-Powered Patient Assistant

A comprehensive medical patient dashboard with conversational AI, featuring local privacy-focused Ollama integration and cloud deployment capabilities.

## 🚀 Features

### 🤖 **Advanced Conversational AI**
- **Natural Language Responses**: Human-like conversations with medical professionals
- **Follow-up Question Handling**: Context-aware responses with conversation memory
- **Dual AI Support**: 
  - **Local**: Ollama (Llama 3.1 8B) for privacy and cost savings
  - **Cloud**: OpenAI GPT-3.5 Turbo for deployment
- **Session-based Context**: Remembers conversation history per patient

### 📊 **Patient Data Management**
- **Interactive Dashboards**: Real-time patient data visualization
- **AI-Powered Similarity Search**: Find similar patients based on medical conditions
- **Provider Analytics**: Filter and analyze data by healthcare providers
- **Medical Condition Tracking**: Monitor treatments, medications, and outcomes
- **Data Export**: Export filtered data for analysis

### 🔒 **Privacy & Security**
- **Local AI Option**: Complete data privacy with Ollama integration
- **No External Dependencies**: Run entirely offline for sensitive data
- **Session Management**: Secure conversation context handling
- **Environment Detection**: Automatic fallback between local and cloud AI

### 🎨 **Modern Interface**
- **Dark Theme**: Professional medical interface design
- **Responsive Design**: Works seamlessly on all devices
- **Real-time Chat**: Instant AI responses with typing indicators
- **New Conversation**: Reset chat context for fresh discussions

## 🛠 Technology Stack

- **Backend**: Flask (Python), Session Management
- **AI/ML**: 
  - **Local**: Ollama (Llama 3.1 8B) via LangChain
  - **Cloud**: OpenAI GPT-3.5 Turbo
  - **ML**: scikit-learn for patient similarity
  - **Data**: pandas/numpy for processing
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Chart.js for interactive charts
- **Infrastructure**: Render deployment, GitHub integration

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git
- **For Local AI**: [Ollama](https://ollama.ai/) installation
- **For Cloud Deployment**: OpenAI API key

## 🔧 Installation

### Local Development (with Ollama - Recommended)

1. **Install Ollama** (macOS)
   ```bash
   brew install ollama
   ```

2. **Download Llama 3.1 model**
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Start Ollama service**
   ```bash
   brew services start ollama
   ```

4. **Clone and setup the project**
   ```bash
   git clone https://github.com/mrsanketpatil19/medical-chatbot.git
   cd medical-chatbot
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the dashboard**
   Open `http://localhost:5001`

### Cloud Deployment Setup

1. **Set environment variables**
   ```bash
   # Required for cloud deployment
   export OPENAI_API_KEY=your_openai_api_key
   export SECRET_KEY=your_secret_key_for_sessions
   ```

2. **The app automatically detects the environment:**
   - **Local**: Uses Ollama if available on localhost:11434
   - **Cloud**: Falls back to OpenAI API for deployment

## 🚀 Deployment on Render + Local Ollama

### 🎯 Deployment Strategy
Since Render can't run Ollama directly, we'll use a tunnel to connect your deployed app to your local Ollama instance.

### Prerequisites
- GitHub account
- Render account (free at [render.com](https://render.com))
- Local Ollama running with Llama 3.1 8B
- ngrok for tunneling (or Tailscale for production)

### 🔧 Option A: Quick Deploy with ngrok

#### Step 1: Set Up Local Ollama Tunnel

1. **Make sure Ollama is running:**
   ```bash
   brew services start ollama
   ollama run llama3.1:8b
   ```

2. **Install ngrok (if not installed):**
   ```bash
   brew install ngrok
   ```

3. **Run our tunnel setup script:**
   ```bash
   ./setup_tunnel.sh
   ```
   
   This will:
   - Check Ollama status
   - Start ngrok tunnel
   - Give you a public URL like `https://abc123.ngrok-free.app`

#### Step 2: Deploy to Render

1. **Connect to Render:**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the `medical-chatbot` repository

2. **Configure the deployment:**
   ```
   Name: medical-chatbot-ollama
   Environment: Docker
   Dockerfile Path: ./Dockerfile
   ```

3. **Set environment variables in Render:**
   ```
   OLLAMA_BASE_URL = https://your-ngrok-url.ngrok-free.app
   SECRET_KEY = random_secret_key_for_sessions
   OPENAI_API_KEY = your_backup_openai_key (optional)
   ```

4. **Deploy**: Click "Create Web Service"

### 🔒 Option B: Production Deploy with Tailscale

For longer-term production use:

1. **Install Tailscale:**
   ```bash
   brew install tailscale
   tailscale up
   ```

2. **Get your Tailscale machine URL:**
   ```bash
   tailscale status
   ```
   
   Your machine will have a URL like: `https://machine-name.tail-scale-domain.ts.net`

3. **Use Tailscale URL in Render:**
   ```
   OLLAMA_BASE_URL = https://your-machine.ts.net:11434
   ```

### 🧪 Testing Your Deployment

1. **Test local tunnel:**
   ```bash
   curl https://your-ngrok-url.ngrok-free.app/api/tags
   ```

2. **Test deployed app:**
   - Visit `https://your-app.onrender.com`
   - Enter patient ID: `patient-0001`
   - Ask: "Provide summary"

Your app will be available at: `https://your-app-name.onrender.com`

## 💡 Usage Examples

### Chat Interface
1. Enter patient ID (e.g., `patient-0001`)
2. Ask questions like:
   - "Provide summary"
   - "What are his current medications?"
   - "What was his last blood pressure reading?"
   - "How long has he been on Lisinopril?"

### Follow-up Conversations
The AI remembers context within the same session:
```
User: "Provide summary"
AI: "Mark Johnson is a 53-year-old male with hypertension..."

User: "What was his blood pressure again?"
AI: "Mark's blood pressure reading was 131/87..."

User: "How long has he been on medication?"
AI: "Mark's been taking Lisinopril for about a week now..."
```

## 🔒 Privacy Features

### Local Mode (Ollama)
- ✅ Complete data privacy
- ✅ No internet required for AI
- ✅ Zero API costs
- ✅ HIPAA-friendly offline operation

### Cloud Mode (OpenAI)
- ⚠️ Data sent to OpenAI servers
- ⚠️ API costs apply
- ✅ Better for deployment/sharing
- ✅ No local setup required

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/mrsanketpatil19/medical-chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mrsanketpatil19/medical-chatbot/discussions)

## 🌟 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI capabilities
- [LangChain](https://langchain.com/) for AI framework
- [OpenAI](https://openai.com/) for cloud AI services
- [Render](https://render.com/) for deployment platform 