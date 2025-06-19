import os
import json
import fitz
import requests
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from starlette.middleware.sessions import SessionMiddleware
from deep_translator import GoogleTranslator
from typing import List, Dict

def create_retriever():
    return None

def init_db():
    pass

def save_session(history: List[Dict]):
    pass

app = FastAPI()
retriever = create_retriever()
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")
templates = Jinja2Templates(directory="templates")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    streaming=True
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
PDF_VECTOR_DIR = "pdf_vector_db"

def load_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_chat(question, answer):
    history = load_history()
    history.append({"question": question, "answer": answer})
    with open("chat_history.json", "w") as f:
        json.dump(history, f, indent=2)

GeneralKnowledgeTool = Tool(
    name="General Knowledge Tool",
    func=lambda q: llm.invoke(q).content,
    description="Provide detailed answers to any general knowledge-related question"
)

ProgrammingTool = Tool(
    name="Programming Tool",
    func=lambda q: llm.invoke(f"You're a coding expert that gives detail codes and also answers about any programming languages in detail, their detailed differences, advantages, disadvantages over each other. Please write the code for: {q}").content,
    description="Provide answers in code for any programming-related question"
)

TechnologyTool = Tool(
    name="Technology Tool",
    func=lambda q: llm.invoke(q).content,
    description="Provide detailed answers to any technology-related question"
)

HorrorTool = Tool(
    name="Horror Tool",
    func=lambda q: llm.invoke(q).content,
    description="Provide detailed guidance on where to go and avoid for horror-related queries"
)

GreetingTool = Tool(
    name="Greeting Tool",
    func=lambda q: greeting_response(q),
    description="Provide a friendly and detailed response to greetings like 'hi' or 'thanks'"
)

def greeting_response(query):
    query = query.lower()
    response = ""
    if "hi" in query or "hello" in query:
        response += "Hello! It is great to hear from you. I am your AI assistant, here to help with any questions you might have. Whether you are looking for information, coding assistance, or just want to chat, I am ready to assist! "
    if "programming" in query or "general" in query or "tips" in query:
        response += "Feel free to ask me anything, from general knowledge to programming tips. "
    if "detail" in query or "short" in query:
        response += "If you need detailed answers, I will provide them with context and examples. For shorter responses, just say 'give short.' "
    if "thanks" in query or "topic" in query or "interact" in query:
        response += "Thank you for interacting with me! If you have a specific topic in mind, let me know, and I will dive into it with enthusiasm. "
    if "explore" in query or "question" in query or "help" in query:
        response += "My goal is to make your experience informative and enjoyable, so don't hesitate to explore various subjects. What's on your mind today?"
    if response.strip() == "":
        response = "I'm here to assist with anything you need. Just say 'hi' or ask your question directly!"
    return response

GeneralKnowledgeAgent = initialize_agent(
    tools=[GeneralKnowledgeTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

ProgrammingAgent = initialize_agent(
    tools=[ProgrammingTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

TechnologyAgent = initialize_agent(
    tools=[TechnologyTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

HorrorAgent = initialize_agent(
    tools=[HorrorTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

GreetingAgent = initialize_agent(
    tools=[GreetingTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

SUPPORTED_LANGUAGES = {
    "fr": "french",
    "es": "spanish",
    "ur": "urdu",
    "de": "german",
    "ar": "arabic",
    "zh-cn": "chinese (simplified)"
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": load_history(),
        "answer": ""
    })

@app.post("/upload_pdf", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("uploaded.pdf", "wb") as f:
            f.write(contents)

        text = ""
        with fitz.open("uploaded.pdf") as doc:
            for page in doc:
                text += page.get_text()

        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.create_documents([text])

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(PDF_VECTOR_DIR)

        request.session["pdf_uploaded"] = True
        request.session["just_uploaded_pdf"] = True

        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": "✅ PDF uploaded and processed successfully!",
            "history": load_history(),
            "logged_in": True
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": f"❌ Error uploading PDF: {e}",
            "history": load_history(),
            "logged_in": True
        })

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    q_lower = question.lower()
    history = load_history()

    if request.session.get("just_uploaded_pdf"):
        request.session["just_uploaded_pdf"] = False
        vectorstore = FAISS.load_local(PDF_VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = pdf_qa.run(question)
        note = "📄 PDF Answer:"
    else:
        programming_keywords = ["code", "python", "function", "loop", "compile", "variable", "error", "logic", "c#", "java", "program"]
        technology_keywords = ["ai", "technology", "cloud", "internet", "software", "hardware", "robotics", "iot", "server", "gadget"]
        horror_keywords = ["ghost", "haunted", "scary", "supernatural", "possession", "evil", "creepy", "spirit", "demon", "dark"]
        greeting_keywords = ["hi", "hello", "thanks", "thank you"]

        conversation = "\n".join([f"User: {h['question']}\nBot: {h['answer']}" for h in history])
        full_context = f"{conversation}\n\n{question}"

        if any(word in q_lower for word in programming_keywords):
            prompt = f"Provide a programming answer to: {full_context}."
            answer = ProgrammingAgent.run(prompt)
            note = "💻 Programming Answer:"
        elif any(word in q_lower for word in technology_keywords):
            prompt = f"Provide a technology answer to: {full_context}."
            answer = TechnologyAgent.run(prompt)
            note = "🧠 Technology Answer:"
        elif any(word in q_lower for word in horror_keywords):
            prompt = f"Provide a horror guidance answer to: {full_context}."
            answer = HorrorAgent.run(prompt)
            note = "👻 Horror Guidance Answer:"
        elif any(word in q_lower for word in greeting_keywords):
            prompt = f"Provide a greeting response to: {full_context}."
            answer = GreetingAgent.run(prompt)
            note = "😊 Greeting Response:"
        else:
            prompt = f"Provide a general knowledge answer to: {full_context}."
            answer = GeneralKnowledgeAgent.run(prompt)
            note = "🌐 General Knowledge Answer:"

    save_chat(question, answer)
    full_answer = f"{note}\n\n{answer}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "answer": full_answer,
        "history": load_history(),
        "logged_in": True
    })

@app.post("/clear", response_class=HTMLResponse)
async def clear(request: Request):
    with open("chat_history.json", "w") as f:
        json.dump([], f)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": [],
        "answer": "✅ Chat history cleared!"
    })

@app.post("/voice_chat")
async def voice_chat(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()

        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": file.content_type
            },
            data=audio_data
        )

        if response.status_code != 200:
            return JSONResponse({"error": "Transcription failed", "details": response.text}, status_code=500)

        result = response.json()
        transcription = result["results"]["channels"][0]["alternatives"][0]["transcript"]

        if not transcription:
            return JSONResponse({"error": "No transcription received"}, status_code=400)

        q_lower = transcription.lower()
        programming_keywords = ["code", "python", "function", "loop", "compile", "variable", "error", "logic", "c#", "java", "program"]
        technology_keywords = ["ai", "technology", "cloud", "internet", "software", "hardware", "robotics", "iot", "server", "gadget"]
        horror_keywords = ["ghost", "haunted", "scary", "supernatural", "possession", "evil", "creepy", "spirit", "demon", "dark"]
        greeting_keywords = ["hi", "hello", "thanks", "thank you"]

        history = load_history()
        conversation = "\n".join([f"User: {h['question']}\nBot: {h['answer']}" for h in history])
        full_context = f"{conversation}\n\n{transcription}"

        if any(word in q_lower for word in programming_keywords):
            answer = ProgrammingAgent.run(f"Provide a programming answer to: {full_context}")
        elif any(word in q_lower for word in technology_keywords):
            answer = TechnologyAgent.run(f"Provide a technology answer to: {full_context}")
        elif any(word in q_lower for word in horror_keywords):
            answer = HorrorAgent.run(f"Provide a horror guidance answer to: {full_context}")
        elif any(word in q_lower for word in greeting_keywords):
            answer = GreetingAgent.run(f"Provide a greeting response to: {full_context}")
        else:
            answer = GeneralKnowledgeAgent.run(f"Provide a general knowledge answer to: {full_context}")

        save_chat(transcription, answer)

        return JSONResponse({"question": transcription, "answer": answer})
    except Exception as e:
        return JSONResponse({"error": f"Voice processing error: {str(e)}"}, status_code=500)

@app.post("/new_chat", response_class=HTMLResponse)
async def new_chat(request: Request):
    history = load_history()
    if history:
        save_session(history)
    with open("chat_history.json", "w") as f:
        json.dump([], f)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": [],
        "answer": "✅ New chat started!"
    })

@app.post("/translate")
async def translate(text: str = Form(...), lang: str = Form(...)):
    try:
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower())
        if not lang_name:
            return JSONResponse({"error": f"Unsupported language code: {lang}"}, status_code=400)

        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        if not translated:
            return JSONResponse({"error": "Translation failed, no result returned"}, status_code=500)

        save_chat(f"Translate '{text}' to {lang_name}", translated)

        return JSONResponse({
            "question": f"Translate '{text}' to {lang_name}",
            "translated_text": translated
        })
    except Exception as e:
        return JSONResponse({"error": f"Translation error: {str(e)}"}, status_code=500)