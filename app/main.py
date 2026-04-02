# app/main.py

import os
import requests
import uuid
import base64
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .ingestion import process_and_get_retriever

# --- Configuration & Models ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
DOWNLOAD_PATH = "./downloaded_files"
DB_BASE_PATH = "./db"

class HackRxRequest(BaseModel):
    documents: str = Field(...)
    questions: List[str] = Field(...)
    is_base64: Optional[bool] = Field(default=False)

class HackRxResponse(BaseModel):
    answers: List[str]

# --- App Lifespan & FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(DB_BASE_PATH, exist_ok=True)
    global llm_pro, llm_flash
    llm_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=API_KEY, temperature=0)
    llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY, temperature=0)
    print("Models loaded.")
    yield
    print("Server shutting down.")

app = FastAPI(title="HackRx 6.0 Submission API", lifespan=lifespan)

# --- Health Check Endpoint ---
@app.get("/hackrx/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

# --- Helper function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Hackathon API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["HackRx Submission"])
async def run_submission(request: HackRxRequest):
    document_data = request.documents
    questions = request.questions
    is_base64 = request.is_base64
    
    local_filename = os.path.join(DOWNLOAD_PATH, str(uuid.uuid4()) + ".pdf")
    
    try:
        if is_base64:
            # Handle base64-encoded file upload
            file_content = base64.b64decode(document_data)
            with open(local_filename, 'wb') as f:
                f.write(file_content)
        else:
            # Handle URL download
            response = requests.get(document_data)
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                f.write(response.content)
    except base64.binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {e}")

    document_id = os.path.splitext(os.path.basename(local_filename))[0]
    
    retriever, full_text = process_and_get_retriever(local_filename, document_id)
    if not retriever:
        raise HTTPException(status_code=500, detail="Failed to process document.")

    # --- ROUTER CHAIN ---
    router_template = """You are an expert at routing a user's question. Based on the question, determine if it is a 'Specific Fact' question or a 'General Context' question.
    - 'Specific Fact' questions ask for a precise number, date, name, or a waiting period for a named item (e.g., "What is the waiting period for cataracts?", "What is the limit for room rent?").
    - 'General Context' questions are broader and ask for summaries or conditions (e.g., "Does this policy cover maternity?", "Summarize the organ donor rules.").
    Return only the single word 'Specific Fact' or 'General Context'.
    Question: {question}
    Classification:"""
    router_prompt = ChatPromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm_flash | StrOutputParser()

    # --- FINAL PROMPT ---
    final_prompt_template = """You are a highly precise Q&A engine that answers questions based ONLY on the provided CONTEXT.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    **INSTRUCTIONS FOR YOUR ANSWER:**
    1.  **Strictly Adhere to Context:** Your answer MUST be based exclusively on the information within the provided CONTEXT.
    2.  **CRITICAL REASONING RULE:** To answer the question, you may need to synthesize scattered information. For questions about a 'waiting period' for a specific procedure, you MUST find where the procedure is listed and what waiting period category it falls under.
    3.  **Conciseness vs. Completeness Rule:** Your answer MUST be a single, concise paragraph. 
        - For **definitional questions** (e.g., "How is a 'Hospital' defined?"), you MUST be comprehensive and include all specific criteria listed in the context (like bed counts, staff, etc.).
        - For **all other questions**, you MUST be ruthlessly concise and include ONLY the information that directly answers the question. Do not include extra details.
    4.  **Format:** If the question is objective, you MUST begin your answer with "Yes," or "No,".
    5.  **Data Extraction:** You MUST extract precise numerical values and percentages from the context.
    6.  **Missing Information:** If the information is not in the context, state only: "This information is not available in the provided document."

   

    **ANSWER:**"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
    
    answers = []
    for question in questions:
        print(f"--- Answering question: {question} ---")
        try:
            route = router_chain.invoke({"question": question})
            print(f"Router choice: {route}")
            
            if "Specific Fact" in route:
                print("Using Path A: Full Context Search")
                context = full_text
                chain = final_prompt | llm_pro | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            else:
                print("Using Path B: Retrieval-Based Search")
                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | final_prompt
                    | llm_pro
                    | StrOutputParser()
                )
                answer = chain.invoke(question)
            
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            print(f"Error on question '{question}': {e}")
            
    return HackRxResponse(answers=answers)
