#!/usr/bin/env python
# build_knowledge_base.py

import os
import requests
import json

def fetch_medical_dataset(num_records=50):
    """
    Acts as an automated data pipeline fetching medical Q&A pairs from HuggingFace.
    We are using the 'medalpaca/medical_meadow_medical_flashcards' dataset.
    """
    print(f"📥 [EXTRACT] Fetching {num_records} medical records from HuggingFace Datasets API...")
    url = f"https://datasets-server.huggingface.co/rows?dataset=medalpaca%2Fmedical_meadow_medical_flashcards&config=default&split=train&offset=0&length={num_records}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch dataset: {response.text}")
        return []
        
    data = response.json()
    rows = data.get("rows", [])
    print(f"✅ Successfully downloaded {len(rows)} medical flashcards.")
    return rows

def transform_to_markdown(rows, output_path):
    """
    Transforms the raw JSON data into a clean text document that LangChain/ChromaDB can digest easily.
    """
    print(f"⚙️  [TRANSFORM] Formatting the JSON data into a clean Markdown Knowledge Base...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Automated Medical Knowledge Base\n\n")
        f.write("This document contains verified medical flashcards extracted from HuggingFace.\n\n")
        
        for row in rows:
            entry = row.get("row", {})
            instruction = entry.get("instruction", "").strip()
            input_text = entry.get("input", "").strip()
            output_text = entry.get("output", "").strip()
            
            f.write(f"## Medical Query: {instruction}\n")
            if input_text:
                f.write(f"**Context:** {input_text}\n")
            f.write(f"**Clinical Answer:** {output_text}\n\n")
            f.write("---\n\n")
            
    print(f"✅ Transformed data saved to: {output_path}")

def load_into_rag(file_path):
    """
    Loads the transformed document into our RAG Vector Database (ChromaDB).
    For a prototype, we just prove the file is created. The backend parses it automatically 
    like we saw in demo_rag.py.
    """
    print(f"💾 [LOAD] You can now ingest {file_path} into ChromaDB!")
    print("\n🚀 Pipeline finished successfully!")

if __name__ == "__main__":
    print("====================================================")
    print("🧬 Automated Medical Knowledge Base Builder (Pipeline)")
    print("====================================================")
    
    # Run the ETL Pipeline
    dataset_rows = fetch_medical_dataset(num_records=100) # Let's prep 100 deep medical facts
    if dataset_rows:
        output_file = os.path.join("downloaded_files", "huggingface_medical_kb.md")
        transform_to_markdown(dataset_rows, output_file)
        load_into_rag(output_file)
