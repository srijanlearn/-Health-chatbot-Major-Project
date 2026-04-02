# CLAUDE.md

This file provides guidance to Claude AI when working with this document Q&A application.

## Project Context

This is a FastAPI + Streamlit application that uses LangChain and Google Gemini to answer questions about PDF documents. The system uses a **dual-path approach**:
- **Specific Fact Path**: Full document context for precise data extraction
- **General Context Path**: RAG with vector retrieval for summaries

## Frontend Improvement Suggestions

### 1. **Session State Management**
**Problem**: Currently, answers disappear when you modify inputs or refresh the page.

**Solution**: Use `st.session_state` to persist:
- Document URL history
- Question-answer pairs
- Processing history/cache

**Implementation**:
```python
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
```

### 2. **Document Upload Support**
**Problem**: Users must have documents hosted at a URL.

**Solution**: Add `st.file_uploader` to allow local PDF uploads, then:
- Save temporarily to `./downloaded_files/`
- Process the local file directly
- Modify backend to accept both URLs and file uploads (multipart/form-data)

**Backend Change Needed**: Add new endpoint `/hackrx/upload` that accepts `UploadFile` from FastAPI.

### 3. **Question Templates / Quick Actions**
**Problem**: Users need to type questions manually.

**Solution**: Add preset question templates with a dropdown or button group:
```python
templates = {
    "Waiting Period": "What is the waiting period for {condition}?",
    "Coverage Check": "Does this policy cover {item}?",
    "Limits": "What is the limit for {benefit}?",
    "Definitions": "How is {term} defined in this policy?"
}
```

Users select template → fill in variable → auto-populate question field.

### 4. **Progress Indicators & Streaming**
**Problem**: No feedback during long processing times (document download, embedding, LLM inference).

**Solution**: 
- Add progress bar that shows: Download → Processing → Generating Answers
- Use `st.progress()` and update during API call
- Consider WebSocket streaming from backend for real-time updates

### 5. **Answer Quality Indicators**
**Problem**: Users can't tell which path was used or confidence level.

**Solution**: Modify backend response to include metadata:
```python
{
    "answers": [
        {
            "question": "...",
            "answer": "...",
            "path_used": "Specific Fact",  # or "General Context"
            "confidence": 0.95,
            "sources": ["page 3", "page 7"]
        }
    ]
}
```

Display badges in frontend: 🎯 Specific Fact | 🔍 General Context

### 6. **Document Caching**
**Problem**: Re-processing the same document wastes time and API credits.

**Solution**: 
- Check if document_id already exists in `./db/`
- If yes, skip ingestion and reuse existing retriever
- Add "Clear Cache" button to force reprocessing
- Show cached documents in sidebar with metadata (processed date, # questions asked)

### 7. **Bulk Question Import**
**Problem**: Manually entering 10 questions is tedious.

**Solution**: Add textarea for bulk entry:
```python
bulk_input = st.text_area("Or paste multiple questions (one per line)")
if bulk_input:
    questions = [q.strip() for q in bulk_input.split('\n') if q.strip()]
```

### 8. **Export Results**
**Problem**: No way to save/share results.

**Solution**: Add export buttons:
- 📄 Export as PDF report
- 📊 Export as CSV (Question | Answer | Path Used)
- 📋 Copy to clipboard as markdown

**Implementation**:
```python
import io
from reportlab.pdfgen import canvas

def export_as_pdf(qa_pairs):
    buffer = io.BytesIO()
    # Generate PDF with reportlab
    return buffer

st.download_button("Export as PDF", data=pdf_buffer, file_name="results.pdf")
```

### 9. **Multi-Document Comparison**
**Problem**: Can't compare answers across multiple policy documents.

**Solution**: 
- Allow uploading multiple documents
- Ask same questions to all documents
- Display side-by-side comparison table

### 10. **Error Recovery & Retry**
**Problem**: Failed requests lose all entered data.

**Solution**: 
- Wrap API calls in retry logic with exponential backoff
- Cache the request payload before submission
- Add "Retry" button on error that reuses cached payload

### 11. **Document Preview**
**Problem**: Users can't verify if correct document was processed.

**Solution**: 
- Show first page thumbnail using `pdf2image`
- Display document metadata (pages, size, title)
- Add expandable "View Full Text" section

### 12. **Question History & Favorites**
**Problem**: Users repeat similar questions.

**Solution**: 
- Maintain question history in `st.session_state`
- Add "Recent Questions" dropdown
- Star/favorite frequently used questions
- Persist to local JSON file or browser localStorage via custom component

### 13. **Responsive Layout Improvements**
**Current Issue**: Fixed 2-column layout doesn't work well on narrow screens.

**Solution**: Use `st.tabs` instead of columns for mobile-friendly design:
```python
tab1, tab2 = st.tabs(["📝 Input", "💡 Results"])
with tab1:
    # Document and questions
with tab2:
    # Answers
```

### 14. **API Health Check**
**Problem**: No indication if backend is down before submitting.

**Solution**: Add startup health check:
```python
try:
    response = requests.get(f"{api_endpoint.replace('/hackrx/run', '')}/docs")
    st.sidebar.success("✅ Backend Online")
except:
    st.sidebar.error("❌ Backend Offline")
```

### 15. **Advanced Settings Panel**
**Problem**: No control over LLM parameters.

**Solution**: Add collapsible settings:
- Model selection (gemini-pro vs gemini-flash)
- Temperature control
- Max tokens
- Chunk sizes for retrieval
- Top-k retrieval results

## Code Style Guidelines

### Streamlit Best Practices
1. **Always use session state** for data persistence across reruns
2. **Use callbacks** for button clicks to avoid rerun issues
3. **Implement caching** with `@st.cache_data` for expensive operations
4. **Use columns/tabs** for better layout organization
5. **Add loading states** with `st.spinner()` for long operations

### Error Handling
```python
try:
    # API call
except requests.exceptions.Timeout:
    st.error("⏱️ Request timed out. Try again.")
except requests.exceptions.ConnectionError:
    st.error("🔌 Cannot connect to backend.")
except requests.exceptions.HTTPError as e:
    st.error(f"❌ Server error: {e.response.status_code}")
    if e.response.text:
        with st.expander("Error Details"):
            st.code(e.response.text)
```

### Component Organization
Structure the frontend as:
```
frontend.py
├── Config & Imports
├── Session State Initialization
├── Helper Functions
│   ├── api_call()
│   ├── format_answer()
│   └── export_results()
├── Sidebar (Settings)
├── Main Content
│   ├── Input Section
│   └── Results Section
└── Footer
```

## Backend Integration Notes

### When modifying the backend for frontend features:

1. **File uploads**: Change `documents: str` to `documents: Union[str, UploadFile]` in `HackRxRequest`
2. **Metadata in response**: Modify `HackRxResponse` to include path used and confidence scores
3. **Streaming support**: Consider implementing Server-Sent Events (SSE) for progress updates
4. **Caching logic**: Add Redis or simple file-based caching to avoid reprocessing

### Environment Variables to Add
```
# .env
GOOGLE_API_KEY=your_key
MAX_QUESTIONS_PER_REQUEST=10
CACHE_EXPIRY_HOURS=24
ENABLE_DOCUMENT_PREVIEW=true
```

## Testing Guidelines

When implementing frontend improvements:
1. Test with various document sizes (small: 5 pages, large: 100+ pages)
2. Test error scenarios (invalid URL, timeout, malformed PDF)
3. Verify mobile responsiveness
4. Test with slow network (throttling)
5. Ensure session state persists correctly across reruns

## Performance Optimization

1. **Lazy load** heavy libraries (pdf2image, reportlab) only when needed
2. **Debounce** text inputs to avoid excessive reruns
3. **Cache API responses** in session state
4. **Use st.spinner** instead of st.info for non-blocking updates
5. **Implement pagination** for question history (show last 20)

## Accessibility

- Add alt text for emojis
- Use semantic HTML in markdown
- Ensure sufficient color contrast
- Add keyboard shortcuts for common actions
- Provide text alternatives for visual indicators
