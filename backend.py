# # backend.py

# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# import openai
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# # Load environment variables or set your OpenAI API key here
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# # Load the model, embeddings, sentences, and FAISS index
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = np.load('embeddings.npy')
# with open('sentences.txt', 'r', encoding='utf-8') as f:
#     sentences = [line.strip() for line in f]
# index = faiss.read_index('faiss_index.idx')

# app = FastAPI()

# class Message(BaseModel):
#     question: str

# def retrieve_relevant_info(question, k=5):
#     question_embedding = model.encode([question])
#     distances, indices = index.search(np.array(question_embedding).astype('float32'), k)
#     retrieved_sentences = [sentences[idx] for idx in indices[0]]
#     return ' '.join(retrieved_sentences)

# def generate_response(question):
#     context = retrieve_relevant_info(question)
#     usr_prompt = f"""You are an AI assistant helping a recruiter learn about a candidate, Kausshik Manojkumar, based on their resume and LinkedIn profile. Your responses should be based ONLY on the information provided in the context. If the information needed to answer the question is not in the context, say "I don't have enough information to answer that question."

#     Rules:
#     1. Only use information explicitly stated in the context.
#     2. Do not make assumptions or infer information not present in the context.
#     3. If asked about skills, experiences, or qualifications not mentioned in the context, state that you don't have that information.
#     4. Be concise and direct in your responses.

#     Context: {context}

#     Question: {question}

#     Answer:"""

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         prompt=usr_prompt,
#         max_tokens=150,
#         temperature=0.7,
#     )
#     return response.choices[0].text.strip()

# @app.post("/chat")
# async def chat_endpoint(message: Message):
#     answer = generate_response(message.question)
#     return {"answer": answer}

# backend.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load environment variables or set your OpenAI API key here
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load the model, embeddings, sentences, and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('embeddings.npy')
with open('sentences.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f]
index = faiss.read_index('faiss_index.idx')

app = FastAPI()

context_alternate = "Kausshik Manojkumar ♂phone+1 515-715-7861 /envel⌢pekausshikmanojkumar@gmail.com /linkedinlinkedin.com/kausshikm /githubKausshik /gl⌢bekausshik.dev Education Iowa State University Expected: May 2025 Bachelor of Science in Computer Science, Minor in Mathematics with Honors GPA: 3.97 Stanford University Online January 2024 Machine Learning Specialization Skills Languages : Python, Java, C, C++, JavaScript, SQL, Verilog, Bash Frameworks : PyTorch, Flask, Spring Boot, TensorFlow, OpenCV, NumPy, Pandas, HTML5, CSS, React, ROS Databases : MySQL, AWS DynamoDB, AWS RDS Tools & Technologies : Git, AWS (S3, EC2), SVN, Docker Work Experience RTX, Collins Aerospace (FMS) May 2023 – Present Software Engineering Intern Cedar Rapids, IA •Contributed to modernizing a vast code base by implementing updated calculations to improve the accuracy of certain parameters in the Flight Management System by 15%. •Executed and refined high-level, low-level, and regression tests in an Agile environment, enhancing software reliability and ensuring strict adherence to specified requirements. •Identified and resolved 5 critical security vulnerabilities, improving system integrity score by 90%. Iowa State University May 2024 – Present Machine Learning Research Assistant, Dr. Yang Li Ames, IA •Utilized PyTorch to modify the Transformer architecture of a deep learning model to integrate spatio-temporal data and auxiliary data •Achieved 30% more accuracy than state of the art models in forecasting time series data •Engineered efficient data preprocessing pipelines for large-scale time series datasets, optimizing model performance Projects Stock Price Prediction Using Transformers |Python, PyTorch, Time Series Analysis May 2024 – July 2024 •Developed a custom Transformer architecture to predict S&P 500 stock prices using historical data, auxiliary information, and other financial metrics •Implemented data preprocessing pipelines to handle time series data and integrate multiple data sources •Achieved a 25% improvement in prediction accuracy (MAPE) compared to traditional ARIMA and LSTM models SwipeHire Android Application |Java, Spring Boot, MySQL, WebSockets January 2023 – May 2023 •Developed RESTful APIs for the backend functionality using Java, Spring Boot, and MySQL •Integrated WebSockets for real-time texting and swiping features, providing a comprehensive platform for efficient hiring management •Established CI/CD pipelines, to automate code deployment to ensure rapid iterations and updates •Modified existing Junit tests and Iincreased test coverage to 90%, resulting in a 40% reduction in post-release bugs BeGreen Sustainability Application |Flask, Python, AWS RDS, OpenAI API March 2024 – April 2024 •Developed the backend of a sustainability application using Flask, Python, and AWS RDS (MySQL) to promote eco-friendly living through gamification for the Yale Hackathon •Optimized database queries and API endpoints, resulting in a 35% reduction in response times •Implemented an ML model using OpenAI’s API, increasing user engagement by 45% Brain Tumor Classification |TensorFlow, Keras, CNN, Transfer Learning August 2023 – October 2023 •Processed and augmented a dataset of 3000+ MRI images, increasing model robustness •Achieved 97% accuracy on a test set of 500 images, outperforming baseline models by 30% •Validated the model’s efficiency using real medical data from multiple patients in India with an accuracy of 90% Extra-Curricular Activities National Talent Search Examination Scholar 2019 : Placed in the top 2000 out of 1.5 million students across India Delegate of India : Royal Commonwealth Society’s India-Malaysia Student Exchange on Sustainable Development Goals Contact kausshik@iastate.edu www.linkedin.com/in/kausshikm (LinkedIn) Top Skills Large Language Models (LLM) Web Development Decision Trees Certifications Unsupervised Learning, Recommenders, Reinforcement Learning Software Engineering Virtual Experience Program Machine Learning Specialization Problem Solving Java Certification Honors-Awards National Talent Search Exam (NTSE) Scholar Honors Society Liberal Arts and Sciences' Deans Academic Excellence Award General Award for Superior Academic Performance Shikshan BharatiKausshik Manojkumar BS Computer Science + Math @ Iowa State University | SWE @ RTX | Java, Python, ML, LLMs Ames, Iowa, United States Summary Rising senior majoring in Computer Science and minoring in Mathematics at Iowa State University. I am swift at learning new programming languages and new concepts in the field of Science. Actively looking for co-op and full-time opportunities for Fall 2024/ Spring 2025. https://www.github.com/KAUSSHIK Experience Collins Aerospace Software Engineer Intern May 2023 - Present (1 year 5 months) Cedar Rapids, Iowa, United States SWE intern in the Flight Management Systems Department (Avionics) Iowa State University - College of Liberal Arts and Sciences Student Technician August 2022 - Present (2 years 2 months) Ames, Iowa, United States Highly flexible and resourceful student worker who deals with hardware and software issues the department may face as a team. Main responsibilities include troubleshooting, managing tickets, OS Deployments, and System Configuration. Iowa State University Undergraduate Research Assistant December 2021 - May 2022 (6 months) Dr. Mayly Sanchez's Lab The NOvA experiment conducted by Fermilab has a wealth of neutrino interaction data. Worked on a team of two to analyze this data by making plots of rare neutrino interaction data. Ran commands on my computer's Mac's Terminal to log in to Fermilab NoVA servers (novagpvms), plotted and Page 1 of 2 analyzed neutrino interaction data. Used the CAFAna framework (a framework written in C++) to make and study the plots. Education Iowa State University Bachelor of Science - BS, Mathematics and Computer Science · (August 2021 - July 2025) Maharishi Vidya Mandir Senior Secondary School Class 11&12, Physics, Chemistry, Math, Computer Science, and English · (July 2019 - May 2021) Bharatiya Vidya Bhavan's Classes 1-10, Physics, Chemistry, Math, Social Science, English, and French · (January 2006 - June 2019) Page 2 of 2"

class Message(BaseModel):
    question: str

def retrieve_relevant_info(question, k=5):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype('float32'), k)
    retrieved_sentences = [sentences[idx] for idx in indices[0]]
    return ' '.join(retrieved_sentences)

def generate_response(question):
    #context = retrieve_relevant_info(question)
    context = context_alternate
    messages = [
        {"role": "system", "content": """You are an AI assistant helping a recruiter learn about a candidate, Kausshik Manojkumar, based on their resume and LinkedIn profile. Your responses should be based ONLY on the information provided in the context. If the information needed to answer the question is not in the context, say "I don't have enough information to answer that question."
        Rules:
        1. Only use information explicitly stated in the context.
        2. Do not make assumptions or infer information not present in the context.
        3. If asked about skills, experiences, or qualifications not mentioned in the context, state that you don't have that information.
        4. Be CONCISE and DIRECT in your responses. There is no need to be verbose."""},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"}
    )

@app.post("/chat")
async def chat_endpoint(message: Message):
    try:
        answer = generate_response(message.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))