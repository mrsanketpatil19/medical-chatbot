from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
import ast
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-for-development-only')

# Load environment variables
load_dotenv()

# Initialize CSV paths
csv_paths = {
    "Patient": "csv_files/Patient.csv",
    "Observation": "csv_files/Observation.csv",
    "MedicationRequest": "csv_files/MedicationRequest.csv",
    "DocumentReference": "csv_files/DocumentReference.csv",
    "Condition": "csv_files/Condition.csv",
    "Encounter": "csv_files/Encounter.csv",
    "Dashboard": "csv_files/ProviderPatientDashboard.csv"
}

def format_date(date_str):
    """Format date string to a more readable format."""
    try:
        # Handle different date formats
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d")
                return dt.strftime("%B %d, %Y")
            except ValueError:
                continue
        return date_str
    except:
        return date_str

def generate_patient_summary(patient_id):
    """Generate a summary of patient data from all CSV files."""
    try:
        print(f"\n=== Generating summary for patient: {patient_id} ===")
        
        # Define which columns to include and their display names
        columns_to_include = {
            'Patient': [
                ('Name', 'Patient Name'),
                ('Gender', 'Gender'),
                ('BirthDate', 'Date of Birth')
            ],
            'Encounter': [
                ('Start', 'Visit Date'),
                ('Status', 'Status'),
                ('Class', 'Visit Type')
            ],
            'Observation': [
                ('ObservationDate', 'Date'),
                ('Systolic', 'Systolic BP'),
                ('Diastolic', 'Diastolic BP'),
                ('Code', 'Test')
            ],
            'MedicationRequest': [
                ('Medication', 'Medication'),
                ('Dosage', 'Dosage'),
                ('MedicationDate', 'Prescribed On')
            ],
            'Condition': [
                ('ConditionCode', 'Condition'),
                ('ClinicalStatus', 'Status'),
                ('OnsetDate', 'First Diagnosed')
            ]
        }
        
        # Dictionary to store all patient data
        patient_data = {
            'demographics': {},
            'conditions': [],
            'medications': [],
            'visits': [],
            'observations': []
        }
        
        # First pass: collect all data
        for section, path in csv_paths.items():
            try:
                df = pd.read_csv(path)
                print(f"\nProcessing {section}...")
                
                if 'PatientID' in df.columns or 'patient_id' in df.columns:
                    patient_col = 'PatientID' if 'PatientID' in df.columns else 'patient_id'
                    records = df[df[patient_col].astype(str).str.lower() == str(patient_id).lower()]
                    
                    if not records.empty:
                        print(f"Found {len(records)} records in {section}")
                        
                        # Process each record
                        for _, row in records.iterrows():
                            if section == 'Patient':
                                # Handle demographics
                                for col, display_name in columns_to_include.get(section, []):
                                    if col in row:
                                        val = row[col]
                                        if 'Date' in col and pd.notna(val):
                                            val = format_date(str(val))
                                        patient_data['demographics'][display_name] = val
                                        
                            elif section == 'Condition':
                                # Handle conditions
                                condition = {}
                                for col, display_name in columns_to_include.get(section, []):
                                    if col in row:
                                        val = row[col]
                                        if 'Date' in col and pd.notna(val):
                                            val = format_date(str(val))
                                        condition[display_name] = val
                                if condition:
                                    patient_data['conditions'].append(condition)
                                    
                            elif section == 'MedicationRequest':
                                # Handle medications
                                med = {}
                                for col, display_name in columns_to_include.get(section, []):
                                    if col in row:
                                        val = row[col]
                                        if 'Date' in col and pd.notna(val):
                                            val = format_date(str(val))
                                        med[display_name] = val
                                if med:
                                    patient_data['medications'].append(med)
                                    
                            elif section == 'Encounter':
                                # Handle visits
                                visit = {}
                                for col, display_name in columns_to_include.get(section, []):
                                    if col in row:
                                        val = row[col]
                                        if 'Date' in col and pd.notna(val):
                                            val = format_date(str(val))
                                        visit[display_name] = val
                                if visit:
                                    patient_data['visits'].append(visit)
                                    
                            elif section == 'Observation':
                                # Handle observations
                                obs = {}
                                for col, display_name in columns_to_include.get(section, []):
                                    if col in row and pd.notna(row[col]):
                                        obs[display_name] = row[col]
                                if obs:
                                    patient_data['observations'].append(obs)
                    
            except Exception as e:
                print(f"Error processing {section}: {str(e)}")
                continue
        
        # Second pass: format the summary
        summary = []
        
        # Add patient header
        if patient_data['demographics']:
            summary.append("=== PATIENT DEMOGRAPHICS ===")
            for key, value in patient_data['demographics'].items():
                summary.append(f"{key}: {value}")
        
        # Add conditions
        if patient_data['conditions']:
            summary.append("\n=== MEDICAL CONDITIONS ===")
            for condition in patient_data['conditions']:
                cond_text = [f"- {condition.get('Condition', 'N/A')}"]
                if 'Status' in condition:
                    cond_text.append(f"  Status: {condition['Status']}")
                if 'First Diagnosed' in condition:
                    cond_text.append(f"  First Diagnosed: {condition['First Diagnosed']}")
                summary.append("\n".join(cond_text))
        
        # Add medications
        if patient_data['medications']:
            summary.append("\n=== CURRENT MEDICATIONS ===")
            for med in patient_data['medications']:
                med_text = [f"- {med.get('Medication', 'N/A')}"]
                if 'Dosage' in med:
                    med_text.append(f"  Dosage: {med['Dosage']}")
                if 'Prescribed On' in med:
                    med_text.append(f"  Prescribed: {med['Prescribed On']}")
                summary.append("\n".join(med_text))
        
        # Add visits
        if patient_data['visits']:
            summary.append("\n=== RECENT VISITS ===")
            for visit in patient_data['visits'][-3:]:  # Show only last 3 visits
                visit_text = []
                if 'Visit Date' in visit:
                    visit_text.append(f"- Date: {visit['Visit Date']}")
                if 'Visit Type' in visit:
                    visit_text.append(f"  Type: {visit['Visit Type']}")
                if 'Status' in visit:
                    visit_text.append(f"  Status: {visit['Status']}")
                if visit_text:
                    summary.append("\n".join(visit_text))
        
        # Add observations (blood pressure)
        if patient_data['observations']:
            bp_readings = [obs for obs in patient_data['observations'] 
                          if 'Systolic BP' in obs and 'Diastolic BP' in obs]
            if bp_readings:
                latest_bp = bp_readings[-1]  # Get most recent
                summary.append("\n=== VITAL SIGNS ===")
                summary.append(f"Blood Pressure: {latest_bp.get('Systolic BP', 'N/A')}/{latest_bp.get('Diastolic BP', 'N/A')}")
                if 'Date' in latest_bp:
                    summary.append(f"  (Taken on: {latest_bp['Date']})")
        
        if not any(patient_data.values()):
            return "No data found for this patient ID. Please check the ID and try again."
            
        return "\n".join(summary)
        
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        print(error_msg)
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return error_msg

def setup_llm_chain():
    """Initialize the LLM chain for processing questions."""
    try:
        # Get API key for fallback to OpenAI in cloud deployment
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Create a clean environment without proxy settings
        clean_env = {k: v for k, v in os.environ.items() 
                    if not k.lower().endswith('_proxy') and k != 'proxies'}
        
        # Save current environment and update with clean one
        old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(clean_env)
        
        try:
            # Initialize the language model - Auto-detect environment
            # Try Ollama first (for local development), fallback to OpenAI (for cloud deployment)
            
            try:
                                 # Try Ollama (local or remote via tunnel)
                 import requests
                 ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                 response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
                 if response.status_code == 200:
                     # Ollama is available - use model
                     llm = Ollama(
                         model="llama3.1:8b",
                         temperature=0.7,
                         base_url=ollama_base_url
                     )
                     print(f"Using Ollama model at: {ollama_base_url}")
                 else:
                     raise Exception("Ollama not available")
            except:
                # Fallback to OpenAI for cloud deployment
                if not api_key:
                    raise ValueError("OpenAI API key required for cloud deployment. Set OPENAI_API_KEY environment variable.")
                
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",  # Using 3.5-turbo for cost efficiency
                    temperature=0.7,
                    openai_api_key=api_key,
                    request_timeout=30
                )
                print("Using OpenAI API (cloud deployment)")
            
            # Define the prompt template
            prompt_template = """
            You are a friendly, knowledgeable medical assistant helping healthcare professionals understand patient information. 
            Speak naturally and conversationally, as if you're having a professional discussion with a colleague.
            
            Based on the patient data below, please answer the user's question in a warm, human-like manner.
            - Use natural language and speak as if you're talking to a friend
            - Be thorough but conversational
            - If you notice important patterns or concerns, mention them naturally
            - If you don't have specific information, say so honestly
            - For follow-up questions, reference previous context when helpful
            - Include relevant details like dates, values, and trends when discussing medical information
            
            Patient Information:
            {patient_summary}
            
            Question: {question}
            
            Please respond in a natural, conversational way:"""
            
            # Return the llm, prompt_template, and model type for proper invocation
            model_type = "ollama" if hasattr(llm, 'base_url') and "localhost" in str(llm.base_url) else "openai"
            return {"llm": llm, "prompt_template": prompt_template, "model_type": model_type}
            
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM (Ollama): {str(e)}. Make sure Ollama is running with: brew services start ollama")
            
    except Exception as e:
        print(f"Error setting up LLM chain: {str(e)}")
        raise
    finally:
        # Restore original environment
        if 'old_env' in locals():
            os.environ.clear()
            os.environ.update(old_env)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation context for a specific patient or all patients"""
    data = request.json
    patient_id = data.get('patient_id') if data else None
    
    if 'conversation_context' in session:
        if patient_id:
            # Clear conversation for specific patient
            if patient_id in session['conversation_context']:
                del session['conversation_context'][patient_id]
                session.modified = True
                return jsonify({'message': f'Conversation cleared for patient {patient_id}'})
            else:
                return jsonify({'message': f'No conversation found for patient {patient_id}'})
        else:
            # Clear all conversations
            session['conversation_context'] = {}
            session.modified = True
            return jsonify({'message': 'All conversations cleared'})
    
    return jsonify({'message': 'No conversations to clear'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    patient_id = data.get('patient_id')
    question = data.get('question')
    
    if not patient_id or not question:
        return jsonify({'error': 'Patient ID and question are required'}), 400
    
    try:
        # Initialize conversation context if it doesn't exist
        if 'conversation_context' not in session:
            session['conversation_context'] = {}
        
        # Get or create conversation for this patient
        if patient_id not in session['conversation_context']:
            session['conversation_context'][patient_id] = {
                'messages': [],
                'summary_generated': False
            }
        
        patient_conversation = session['conversation_context'][patient_id]
        
        # Generate patient summary (only once per conversation or if explicitly requested)
        if not patient_conversation['summary_generated'] or 'summary' in question.lower():
            summary = generate_patient_summary(patient_id)
            patient_conversation['summary_generated'] = True
        else:
            # For follow-up questions, use a shorter context
            summary = f"Continue discussing patient {patient_id}. Previous conversation context available."
        
        # Add current question to conversation history
        patient_conversation['messages'].append({
            'type': 'question',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 exchanges to avoid context overflow
        if len(patient_conversation['messages']) > 10:
            patient_conversation['messages'] = patient_conversation['messages'][-10:]
        
        # Build conversation context for better follow-ups
        conversation_history = ""
        if len(patient_conversation['messages']) > 1:
            recent_messages = patient_conversation['messages'][-6:-1]  # Last 3 Q&A pairs (excluding current)
            conversation_history = "\n\nRecent conversation:\n"
            for msg in recent_messages:
                if msg['type'] == 'question':
                    conversation_history += f"Previous question: {msg['content']}\n"
                elif msg['type'] == 'answer':
                    conversation_history += f"Previous answer: {msg['content'][:200]}...\n"
        
        # Get LLM response
        try:
            llm_setup = setup_llm_chain()
            llm = llm_setup["llm"]
            prompt_template = llm_setup["prompt_template"]
            model_type = llm_setup["model_type"]
            
            # Format the prompt with patient data and conversation context
            full_context = summary + conversation_history
            formatted_prompt = prompt_template.format(
                patient_summary=full_context,
                question=question
            )
            
            # Get response based on model type
            if model_type == "ollama":
                response = llm(formatted_prompt)
            else:  # OpenAI
                response = llm.predict(formatted_prompt)
            
            # Add response to conversation history
            patient_conversation['messages'].append({
                'type': 'answer',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save session
            session.modified = True
            
            return jsonify({
                'response': response,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'patient_id': patient_id
            })
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in LLM processing: {error_msg}")
            import traceback
            traceback.print_exc()
            
            if "maximum context length" in error_msg.lower():
                return jsonify({
                    'error': 'The patient record is too large to process. Please ask more specific questions.',
                    'requires_new_chat': True
                }), 413
                
            return jsonify({
                'error': f"I'm having trouble processing your request. Please try asking in a different way.",
                'details': str(e)
            }), 500
            
    except Exception as e:
        print(f"Error in /ask endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'An unexpected error occurred while processing your request.',
            'details': str(e)
        }), 500

@app.route('/get_providers')
def get_providers():
    try:
        # Read the CSV file
        df = pd.read_csv('csv_files/ProviderPatientDashboard.csv')
        # Get unique providers with their IDs
        providers = df[['ProviderID', 'ProviderName']].drop_duplicates().to_dict('records')
        return jsonify(providers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_patients/<provider_id>')
def get_patients(provider_id):
    try:
        # Read the CSV file
        df = pd.read_csv('csv_files/ProviderPatientDashboard.csv')
        # Filter patients by provider
        patients = df[df['ProviderID'] == provider_id][
            ['PatientID', 'Name', 'VisitDate', 'Status_x']
        ].drop_duplicates()
        patients['VisitDate'] = pd.to_datetime(patients['VisitDate']).dt.strftime('%Y-%m-%d')
        return jsonify(patients.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_statuses')
def get_statuses():
    try:
        # Read the CSV file
        df = pd.read_csv('csv_files/ProviderPatientDashboard.csv')
        
        # Get unique status values and sort them
        statuses = sorted(df['Status_x'].dropna().unique().tolist())
        return jsonify(statuses)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_patient_data/<provider_id>')
def get_patient_data(provider_id):
    try:
        # Read the CSV file
        df = pd.read_csv('csv_files/ProviderPatientDashboard.csv')
        
        # Filter by provider
        filtered_df = df[df['ProviderID'] == provider_id]
        
        # Get patient ID filter if provided
        patient_id_filter = request.args.get('patient_id')
        if patient_id_filter:
            filtered_df = filtered_df[filtered_df['PatientID'].astype(str).str.contains(patient_id_filter, case=False, na=False)]
        
        # Get status filter if provided
        status_filter = request.args.get('status')
        if status_filter and status_filter.lower() != 'all':
            filtered_df = filtered_df[filtered_df['Status_x'].str.lower() == status_filter.lower()]
            
        # Get sort order
        sort_order = request.args.get('sort', 'asc')
        
        # Convert VisitDate to datetime for proper sorting
        filtered_df['VisitDate'] = pd.to_datetime(filtered_df['VisitDate'])
        filtered_df = filtered_df.sort_values('VisitDate', ascending=(sort_order == 'asc'))
        
        # Convert back to string for JSON serialization
        filtered_df['VisitDate'] = filtered_df['VisitDate'].dt.strftime('%Y-%m-%d')
        
        # Return only necessary columns
        result = filtered_df[['PatientID', 'Name', 'VisitDate', 'Status_x']].drop_duplicates().to_dict('records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_embedding(embedding_str):
    """Safely convert string representation of list to numpy array"""
    try:
        # Try to evaluate the string as a Python literal
        return np.array(ast.literal_eval(embedding_str))
    except (ValueError, SyntaxError):
        # If evaluation fails, try to handle common formatting issues
        cleaned = embedding_str.strip('[]').split(',')
        return np.array([float(x.strip()) for x in cleaned if x.strip()])

@app.route('/api/similar-patients/<patient_id>', methods=['GET'])
def find_similar_patients(patient_id):
    try:
        # Load the dashboard data
        df = pd.read_csv(csv_paths['Dashboard'])
        
        # Check if patient exists
        if patient_id not in df['PatientID'].values:
            return jsonify({
                'status': 'error',
                'message': f'Patient with ID {patient_id} not found.'
            }), 404
        
        # Process embeddings
        df['embedding_array'] = df['ClinicalFocusEmbedding'].apply(parse_embedding)
        embedding_matrix = np.vstack(df['embedding_array'].values)
        
        # Get the query patient's index and embedding
        patient_idx = df[df['PatientID'] == patient_id].index[0]
        query_embedding = embedding_matrix[patient_idx]
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embedding_matrix
        ).flatten()
        
        # Add similarity scores to dataframe
        df['similarity'] = similarities
        
        # Get the most recent record for each patient (based on VisitDate or similar timestamp)
        # First, ensure we have a datetime column - we'll use VisitDate if available, otherwise use index
        timestamp_col = 'VisitDate' if 'VisitDate' in df.columns else None
        
        # Group by PatientID and get the record with the highest similarity score
        similar_patients = (
            df[df['PatientID'] != patient_id]
            .sort_values('similarity', ascending=False)
            .drop_duplicates(subset=['PatientID'], keep='first')
            .head(5)
        )
        
        # Get the query patient's data
        query_patient = df[df['PatientID'] == patient_id].iloc[0].to_dict()
        
        # Prepare response data
        columns_to_drop = ['ClinicalFocusEmbedding', 'CombinedText', 'embedding_array', 'similarity']
        
        # Format query patient data
        query_patient_data = {
            k: query_patient[k] 
            for k in query_patient 
            if k not in columns_to_drop and not k.startswith('Unnamed')
        }
        
        # Format similar patients data
        similar_patients_data = []
        for _, patient in similar_patients.iterrows():
            patient_dict = {
                k: patient[k] 
                for k in patient.index 
                if k not in columns_to_drop and not str(k).startswith('Unnamed')
            }
            patient_dict['similarity_score'] = float(patient['similarity'])
            similar_patients_data.append(patient_dict)
        
        return jsonify({
            'status': 'success',
            'query_patient': query_patient_data,
            'similar_patients': similar_patients_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)