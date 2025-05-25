# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, extract_text_from_csv, extract_text_from_json, extract_text_from_xlsx
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue
from embeddings import embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from docx import Document
from PyPDF2 import PdfReader
import pandas as pd
import logging
import os, uuid
import openai
import json
from datetime import datetime

import subprocess
import socket

def is_backend_running(host="127.0.0.1", port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

# Start the backend if not already running
if not is_backend_running():
    subprocess.Popen(["flask", "run", "--port", "5000"])


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not QDRANT_URL or not OPENAI_API_KEY or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or OPENAI_API_KEY or QDRANT_API_KEY is not set in .env file")

# Initialize Qdrant client
qdrant_client = QdrantClient(QDRANT_URL, 
                             api_key=QDRANT_API_KEY)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200)

# Load CrossEncoder model for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    uploaded_filenames = []

    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)
            uploaded_filenames.append(filename)
                
    if not uploaded_filenames:
        return jsonify({'error': 'No valid files uploaded'}), 400

    return jsonify({'uploaded': uploaded_filenames}), 200

@app.route('/documents', methods=['GET'])
def get_documents():
    files = os.listdir(UPLOAD_FOLDER)
    documents = []
    for filename in files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        documents.append({
            'filename': filename,
            'description': 'Uploaded document',
            'size_kb': round(os.path.getsize(filepath) / 1024, 2)
        })
    return jsonify(documents)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'json', 'csv', 'xlsx', 'pdf', 'docx'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper: check if extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extractor functions
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def extract_text_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')  # make sure openpyxl is installed
    return df.to_string()

@app.route("/save_vector", methods=["POST"])
def save_vector():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist('files')
        descriptions = []
        i = 0
        while True:
            desc = request.form.get(f"descriptions_{i}")
            if desc is None:
                break
            descriptions.append(desc)
            i += 1

        if len(files) != len(descriptions):
            return jsonify({"error": "Mismatched files and descriptions"}), 400

        extractors = {
            '.pdf': extract_text_from_pdf,
            '.docx': extract_text_from_docx,
            '.txt': extract_text_from_txt,
            '.csv': extract_text_from_csv,
            '.json': extract_text_from_json,
            '.xlsx': extract_text_from_xlsx,
        }

        response_payload = []
        total_chunks = 0
        failed_files = []
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        for i, file in enumerate(files):
            file_path = None
            try:
                # Robust extension detection
                filename = secure_filename(file.filename)
                ext = os.path.splitext(filename)[1].lower()
                
                # Content-type fallback for all supported types
                if not ext and file.content_type:
                    content_type = str(file.content_type).lower()
                    if 'pdf' in content_type:
                        ext = '.pdf'
                    elif 'wordprocessingml' in content_type:  # DOCX
                        ext = '.docx'
                    elif 'spreadsheetml' in content_type:  # XLSX
                        ext = '.xlsx'
                    elif 'csv' in content_type:
                        ext = '.csv'
                    elif 'json' in content_type:
                        ext = '.json'
                    elif 'text/plain' in content_type: 
                        ext = '.txt'
                
                # Final validation
                if ext not in extractors:
                    raise ValueError(f"Unsupported file type: {ext or 'no extension'}")

                # Save temporarily
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                # Verify file was saved
                if not os.path.exists(file_path):
                    raise IOError("File failed to save")

                # Extract text
                text = extractors[ext](file_path)
                if not text.strip():
                    raise ValueError("Empty file content")

                # Process content
                combined_text = f"Description: {descriptions[i]}\nContent:\n{text}"
                chunks = text_splitter.split_text(combined_text)
                
                # Prepare vectors
                points = []
                for chunk in chunks:
                    embedding = embeddings.embed_query(chunk)
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "filename": file.filename,
                            "description": descriptions[i],
                            "upload_date": datetime.now().isoformat()
                        }
                    ))

                # Save to Qdrant
                if points:
                    qdrant_client.upsert(
                        collection_name="Document",
                        points=points,
                        wait=True
                    )
                    total_chunks += len(points)
                    response_payload.append({
                        "filename": file.filename,
                        "chunks_saved": len(points),
                        "status": "success"
                    })

            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                app.logger.error(f"Error processing {file.filename}: {str(e)}")
                
            finally:
                # Clean up temp file
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        app.logger.error(f"Error removing temp file {file_path}: {str(e)}")

        return jsonify({
            "message": "Batch upload completed",
            "successful_uploads": response_payload,
            "failed_uploads": failed_files,
            "total_chunks": total_chunks,
            "total_files_processed": len(files),
            "status": "partial_success" if failed_files else "complete_success"
        }), 200

    except Exception as e:
        app.logger.error(f"Error in save_vector: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500 
    
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract query text and target files
        query_text = data.get("query_text", "").strip()
        target_files = data.get("target_files", [])

        # Ensure target_files is always a list
        if isinstance(target_files, str):
            target_files = [target_files]

        # Validate required fields
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query_text)

        # Prepare filter for specific files if provided
        filter_condition = None
        if target_files:
            filter_condition = Filter(
                must=[FieldCondition(
                    key="filename",
                    match=MatchAny(any=target_files)
                )]
            )

        # Search Qdrant vector database
        search_results = qdrant_client.search(
            collection_name="Document",
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=20
        )

        if not search_results:
            return jsonify({"error": "No relevant results found"}), 404

        # Rerank results using CrossEncoder
        reranker_inputs = [
            (query_text, result.payload.get('text', '')) 
            for result in search_results
        ]
        reranker_scores = reranker.predict(reranker_inputs)

        # Update scores with reranker values
        for i, result in enumerate(search_results):
            result.score = reranker_scores[i]

        # Get top 10 results after reranking
        reranked_results = sorted(
            search_results, 
            key=lambda x: x.score, 
            reverse=True
        )[:10]

        # Prepare response payload
        response_payload = [
            {
                "text": r.payload.get("text", "").strip(),
                "filename": r.payload.get("filename", "Unknown"),
                "score": float(r.score)  # Convert numpy float to native float
            }
            for r in reranked_results
        ]

        # Generate final answer using GPT-4
        context = "\n\n".join(
            f"Document chunk {i+1} from {chunk['filename']} (relevance: {chunk['score']:.2f}):\n{chunk['text']}"
            for i, chunk in enumerate(response_payload)
        )

        prompt = (
            "You are an AI assistant answering questions based on provided documents.\n\n"
            f"User Question: {query_text}\n\n"
            "Relevant Context:\n"
            f"{context}\n\n"
            "Instructions:\n"
            "1. Provide a concise answer to the question\n"
            "2. Only use information from the provided context\n"
            "3. If unsure, say 'I couldn't find enough information to answer that question'\n"
            "4. Format your response clearly with proper paragraphs and bullet points when needed"
        )

        # Call OpenAI API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
        # openai.api_key = OPENAI_API_KEY

        # response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate answers based on document content."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            stream=False
        )

        #final_answer = response.choices[0].message['content'].strip()
        final_answer = completion.choices[0].message.content.strip()

        # Return final response
        return jsonify({
            "answer": final_answer,
            "source": response_payload,
            "status": "success"
        }), 200

    except openai.APIError as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        return jsonify({
            "error": "Failed to generate answer",
            "details": str(e)
        }), 503

    except Exception as e:
        app.logger.error(f"Error in /chat: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

    
#List all files
@app.route("/list_files", methods=["GET"])
def list_files():
    try:
        results, _ = qdrant_client.scroll(
            collection_name="Document",
            with_payload=["filename", "description"],
            limit=1000
        )

        unique_files = {}

        for point in results:
            payload = point.payload  # Accessing the payload from the Record object
            filename = payload.get("filename")
            if filename and filename not in unique_files:
                unique_files[filename] = {
                    "filename": filename,
                    "description": payload.get("description", "")
                }

        # Convert the dict to a list of file info
        file_list = list(unique_files.values())

        return jsonify({
            "final_answer": "Successfully listed files",
            "files": file_list,
            "status": "success"
        }), 200

    except Exception as e:
        app.logger.error(f"Error in /list_files: {str(e)}")
        return jsonify({
            "error": "Failed to list files",
            "details": str(e)
        }), 500

# Delete a file
@app.route("/delete_file", methods=["POST"])
def delete_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        # Delete all vectors associated with the given filename
        qdrant_client.delete(
            collection_name="Document",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
        )

        return jsonify({
            "message": f"All vectors for '{filename}' deleted successfully",
            "status": "success"
        }), 200

    except Exception as e:
        logging.error(f"Error in delete_file: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# Running the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
