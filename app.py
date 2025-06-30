from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
import ast
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
        # First, ensure we have the API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Create a clean environment without proxy settings
        clean_env = {k: v for k, v in os.environ.items() 
                    if not k.lower().endswith('_proxy') and k != 'proxies'}
        
        # Save current environment and update with clean one
        old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(clean_env)
        
        try:
            # Initialize the language model
            llm = ChatOpenAI(
                model_name="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper responses
                temperature=0.7,
                openai_api_key=api_key,
                request_timeout=30
            )
            
            # Define the prompt template
            prompt_template = """
            You are a helpful medical assistant analyzing patient data. 
            Given the following patient summary, answer the user's question.
            If you don't know the answer, say you don't know, don't try to make up an answer.
            
            Patient Summary:
            {patient_summary}
            
            Question: {question}
            
            Answer in a clear and concise manner:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            return LLMChain(llm=llm, prompt=prompt)
            
        except Exception as e:
            raise ValueError(f"Failed to initialize ChatOpenAI: {str(e)}")
            
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

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    patient_id = data.get('patient_id')
    question = data.get('question')
    
    if not patient_id or not question:
        return jsonify({'error': 'Patient ID and question are required'}), 400
    
    try:
        # Generate patient summary
        summary = generate_patient_summary(patient_id)
        
        # Get LLM response
        try:
            llm_chain = setup_llm_chain()
            response = llm_chain.run(
                patient_summary=summary,
                question=question
            )
            
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