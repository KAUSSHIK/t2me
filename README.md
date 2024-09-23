# t2me
# Kausshik's Assistant Backend

This is the backend component of Kausshik's Assistant, a chatbot designed to help recruiters learn about me (Kausshik) based on his resume and LinkedIn profile.

## Technology Stack

- Python 3.10
- FastAPI
- OpenAI GPT-4o-mini
- FAISS for efficient similarity search
- Sentence Transformers for text embeddings

## Setup

1. Clone the repository:
   ```
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Prepare your data:
   Ensure you have the following files in your project directory:
   - `embeddings.npy`: NumPy file containing pre-computed embeddings
   - `sentences.txt`: Text file containing sentences corresponding to the embeddings
   - `faiss_index.idx`: FAISS index file for efficient similarity search

## Running the Backend

To run the backend server:

```
uvicorn backend:app --host 0.0.0.0 --port 8000
```

The server will start running on `http://localhost:8000`.

## API Endpoints

- POST `/chat`: Send a question about Kausshik to get an AI-generated response.

  Request body:
  ```json
  {
    "question": "What are Kausshik's key skills?"
  }
  ```

  Response:
  ```json
  {
    "answer": "Based on the information provided, Kausshik's key skills include..."
  }
  ```

## Deployment

This backend is designed to be deployed on Heroku. Follow these steps:

1. Create a new Heroku app.
2. Set the necessary environment variables in the Heroku dashboard.
3. Deploy the code to Heroku using Git.

For detailed deployment instructions, refer to the Heroku documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
