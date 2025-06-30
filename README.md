# Windsurf Patient Dashboard

A Flask-based patient dashboard application with AI-powered patient similarity search and medical data analysis.

## Features

- Patient data visualization and management
- AI-powered patient similarity search
- Medical condition tracking
- Medication management
- Provider-patient relationship management
- Interactive chat interface for medical queries

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (create a `.env` file):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   ```
5. Run the application:
   ```bash
   python app.py
   ```
6. Open http://localhost:5001 in your browser

## Deployment on Render

### Prerequisites
- A GitHub account
- A Render account (free at render.com)

### Steps to Deploy

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Sign up for Render:**
   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

3. **Create a new Web Service:**
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository you just created

4. **Configure the service:**
   - **Name:** `windsurf-patient-dashboard` (or any name you prefer)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free (or choose a paid plan if needed)

5. **Add Environment Variables:**
   - Click on "Environment" tab
   - Add the following variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `SECRET_KEY`: A random secret key for Flask sessions

6. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - Wait for the build to complete (usually 2-5 minutes)

7. **Access your application:**
   - Once deployed, you'll get a URL like `https://your-app-name.onrender.com`
   - Your application will be live and accessible to anyone!

### Important Notes

- **Free Tier Limitations:** Render's free tier has some limitations:
  - Services spin down after 15 minutes of inactivity
  - Limited bandwidth and build minutes
  - For production use, consider upgrading to a paid plan

- **Environment Variables:** Make sure to set your `OPENAI_API_KEY` in Render's environment variables section

- **File Uploads:** If you need to handle file uploads, consider using cloud storage services like AWS S3 or Google Cloud Storage

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── render.yaml           # Render deployment configuration
├── csv_files/            # Patient data CSV files
├── static/               # CSS, JS, and other static files
├── templates/            # HTML templates
└── README.md            # This file
```

## Support

If you encounter any issues during deployment, check:
1. Render's build logs for error messages
2. Environment variables are properly set
3. All dependencies are listed in `requirements.txt`
4. The start command is correct (`gunicorn app:app`) 