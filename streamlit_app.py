import streamlit as st
import requests
import os
import time
from functools import wraps
import socket
from typing import List, Dict
import logging

import time
time.sleep(3)

logging.basicConfig(level=logging.INFO)
logging.info("Starting Streamlit app")

# Configuration
FLASK_BACKEND = "http://127.0.0.1:5000"  # Flask backend address
session = requests.Session()
retries = requests.adapters.Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))

# Helper Functions
def secure_filename(filename: str) -> str:
    """Sanitize filenames to prevent path traversal"""
    import re
    return re.sub(r'[^\w\s-]', '', filename.strip())

def validate_file_extension(filename: str) -> bool:
    """Check if filename has a valid extension"""
    valid_extensions = ['.pdf', '.docx', '.txt', '.csv', '.json', '.xlsx']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def handle_connection_errors(func):
    """Decorator to handle connection issues gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            st.error("Connection failed - trying alternative method...")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect(('127.0.0.1', 5000))
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Permanent connection failure: {str(e)}")
                st.stop()
    return wrapper

@handle_connection_errors
def get_documents() -> List[Dict]:
    """Fetch all documents from backend"""
    response = requests.get(f"{FLASK_BACKEND}/list_files", timeout=60)
    response.raise_for_status()
    return response.json().get("files", [])

@handle_connection_errors
def delete_document(filename: str) -> bool:
    """Delete a document by filename"""
    response = requests.post(
        f"{FLASK_BACKEND}/delete_file",
        json={"filename": filename},
        timeout=10
    )
    response.raise_for_status()
    return True

@handle_connection_errors
def upload_documents(files_data, payload):
    response = requests.post(
        f"{FLASK_BACKEND}/save_vector",
        files=files_data,
        data=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()

@handle_connection_errors
def chat_with_backend(query: str, selected_files: List[str]) -> Dict:
    """Send chat query to backend"""
    response = requests.post(
        f"{FLASK_BACKEND}/chat",
        json={
            "query_text": query,
            "target_files": selected_files
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()

# UI Components
def show_document_stats(documents: List[Dict]):
    """Display document statistics in sidebar"""
    if documents:
        st.sidebar.success(f"📚 Total Documents: {len(documents)}")
        file_types = {}
        total_size = 0
        
        for doc in documents:
            ext = os.path.splitext(doc['filename'])[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            if 'size_kb' in doc:
                total_size += doc['size_kb']
        
        st.sidebar.write("**File Types:**")
        for ext, count in file_types.items():
            st.sidebar.write(f"- {ext.upper()}: {count}")
        
        if total_size > 0:
            st.sidebar.write(f"**Total Size:** {total_size/1024:.2f} MB")

def document_management():
    """Document management page with upload/delete functionality"""
    st.title("Document Management")
    
    try:
        documents = get_documents()
        show_document_stats(documents)
    except Exception as e:
        st.error(f"Failed to load documents: {str(e)}")
        documents = []

    # Document list with delete functionality
    if documents:
        st.subheader("Your Documents")
        for doc in documents:
            with st.expander(f" {doc['filename']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Description:** {doc['description']}")
                    st.caption(f"Uploaded: {doc.get('upload_date', 'Unknown')}")
                    if 'size_kb' in doc:
                        st.caption(f"Size: {doc['size_kb']:.1f} KB")
                with col2:
                    if st.button("Delete", key=f"del_{doc['filename']}"):
                        try:
                            with st.spinner(f"Deleting {doc['filename']}..."):
                                if delete_document(doc['filename']):
                                    st.success(f"Deleted {doc['filename']}!")
                                    time.sleep(1)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {str(e)}")
    else:
        st.info("No documents uploaded yet.")

    # File upload section
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Select files (PDF, DOCX, TXT, CSV, JSON, XLSX)",
        type=['pdf', 'docx', 'txt', 'csv', 'json', 'xlsx'],
        accept_multiple_files=True,
        help="Files must have proper extensions (.pdf, .docx, etc.)"
    )
   
    invalid_files = []

    if uploaded_files:
        invalid_files = [file.name for file in uploaded_files 
                     if not any(file.name.lower().endswith(ext) 
                                for ext in [".pdf", ".docx", ".txt", ".csv", ".json", ".xlsx"])]
    
    if invalid_files:
        st.error(f"The following files have unsupported extensions: {', '.join(invalid_files)}")
        st.stop()
    
    if uploaded_files and not invalid_files:
        st.write(f"Ready to upload {len(uploaded_files)} valid file(s)")

    with st.form("upload_form"):
            descriptions = []
            for i, file in enumerate(uploaded_files):
                desc = st.text_input(
                    f"Description for {file.name}",
                    value=f"Document about {os.path.splitext(file.name)[0]}",
                    key=f"desc_{i}"
                )
                descriptions.append(desc)
            
            if st.form_submit_button("Upload Documents"):
                try:
                    files_data = []
                    payload = {}
                    for i, file in enumerate(uploaded_files):
                        files_data.append((
                            "files",(file.name, file, file.type or 'application/octet-stream')))
                        payload[f"descriptions_{i}"] = descriptions[i]
                    
                    with st.spinner("Uploading..."):
                        result = upload_documents(files_data, payload)
                    
                    if result.get('failed_uploads'):
                        st.error(f"{len(result['failed_uploads'])} files failed")
                        for fail in result['failed_uploads']:
                            st.error(f"{fail['filename']}: {fail['error']}")
                    
                    if result.get('successful_uploads'):
                        st.success(f"Successfully uploaded {len(result['successful_uploads'])} files!")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

def chat_with_documents():
    """Chat interface with document knowledge"""
    st.title("Chat with Your Documents")
    
    try:
        documents = get_documents()
        if documents:
            filenames = [doc['filename'] for doc in documents]
            selected_files = st.multiselect(
                "Select documents to query",
                filenames,
                default=filenames
            )
            
            question = st.text_area("Ask a question:", height=150, 
                                  placeholder="What information are you looking for?")
            
            if st.button("Get Answer") and question:
                with st.spinner("Analyzing documents..."):
                    response = chat_with_backend(question, selected_files)
                
                st.subheader("Answer")
                st.markdown(response.get('answer', 'No answer returned from backend.'))
                
                # if st.checkbox("Show source references"):
                #     st.subheader("Source Materials")
                #     for chunk in response.get['source', []]:
                #         st.write(f"**From {chunk['filename']}** (relevance: {chunk['score']:.2f})")
                #         st.markdown(f"> {chunk['text']}")
                #         st.divider()
        else:
            st.warning("No documents available. Upload documents first.")
    except Exception as e:
        st.error(f"Chat error: {str(e)}")

def main():
    """Main application layout"""
    st.set_page_config(
        page_title="Document AI Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Document Management", "Chat with Documents"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Document AI Assistant\n\n"
        "1. Upload documents in multiple formats\n"
        "2. Chat with your knowledge base\n"
        "3. Manage your documents"
    )

    if page == "Document Management":
        document_management()
    else:
        chat_with_documents()

if __name__ == "__main__":
    main()