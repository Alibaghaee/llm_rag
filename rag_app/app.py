from fastapi import FastAPI
from pydantic import BaseModel
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext # <--- اصلاح شده
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # <--- مسیر رایج برای 0.9.x

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from pathlib import Path
import logging
import sys

# تنظیم لاگ برای نمایش بهتر اطلاعات
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = FastAPI()

# --- پیکربندی اولیه ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"
DATA_DIR = Path(__file__).parent / "data"

# --- بارگذاری مدل‌ها و ساخت ایندکس (در زمان شروع برنامه) ---
embed_model = None
index = None
query_engine = None
documents = [] # مقداردهی اولیه

try:
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=DEVICE)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    # می‌توانید تصمیم بگیرید که برنامه بدون مدل embedding ادامه ندهد
    # raise SystemExit(f"Failed to load embedding model: {e}")

if embed_model:
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None,
    )

    print("Loading documents and building RAG index...")
    if not DATA_DIR.exists():
        print(f"Data directory '{DATA_DIR}' does not exist. Creating it.")
        DATA_DIR.mkdir(parents=True, exist_ok=True) # ایجاد پوشه اگر وجود ندارد

    if not any(DATA_DIR.iterdir()):
        print(f"Warning: Data directory '{DATA_DIR}' is empty.")
        print("Please add your Persian text files (e.g., .txt, .md).")
    else:
        try:
            documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
            if not documents:
                print(f"Warning: No documents found in '{DATA_DIR}'.")
            else:
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                query_engine = index.as_query_engine(
                    similarity_top_k=5
                )
                print("RAG index built successfully.")
        except Exception as e:
            print(f"Error loading documents or building index: {e}")


print("Loading generation model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME).to(DEVICE)
    model.eval()
    print("Generation model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading generation model/tokenizer: {e}")
    # raise SystemExit(f"Failed to load generation model: {e}")


class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(payload: QueryRequest):
    if query_engine is None or index is None or not documents:
        return {"answer": "متاسفانه، سیستم پرسش و پاسخ به درستی مقداردهی اولیه نشده است یا سندی برای جستجو وجود ندارد. لطفا فایل‌های متنی خود را در پوشه 'data' قرار دهید و برنامه را مجددا راه‌اندازی کنید."}

    print(f"Received query: {payload.question}")

    retrieved_nodes = query_engine.retrieve(payload.question)

    if not retrieved_nodes:
        return {"answer": "متنی مرتبط با سوال شما در اسناد موجود یافت نشد."}

    context_chunks = [node.get_content() for node in retrieved_nodes]
    context = "\n\n---\n\n".join(context_chunks)

    print(f"Retrieved context: \n{context[:500]}...")

    prompt = (
        "Answer the following question based ONLY on the provided context. "
        "If the information is not in the context, say 'I cannot find this information in the provided context'. "
        "Be precise and specific.\n\n"
        "Context:\n"
        "------------\n"
        f"{context}\n"
        "------------\n\n"
        "Question: " + payload.question + "\n"
        "Answer:"
    )
    print(f"Constructed prompt for generation model: \n{prompt}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            temperature=0.3,
            top_p=0.9,
            no_repeat_ngram_size=2
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "پاسخ:" in answer:
        answer = answer.split("پاسخ:")[-1].strip()
    elif payload.question in answer:
        answer = answer.split(payload.question)[-1].strip()
        if answer.startswith(":") or answer.startswith("؟"):
            answer = answer[1:].strip()

    print(f"Generated answer: {answer}")
    return {"answer": answer}

# برای اجرای محلی با uvicorn (اختیاری)
# if __name__ == "__main__":
#     # اطمینان از ایجاد پوشه data
#     if not DATA_DIR.exists():
#         DATA_DIR.mkdir(parents=True, exist_ok=True)
#         print(f"Created data directory: {DATA_DIR}")
#         print("Please add your Persian text files to this directory.")
#
#     # بررسی اینکه مدل‌ها بارگذاری شده‌اند
#     if not embed_model or not tokenizer or not model:
#         print("Error: One or more models failed to load. Exiting.")
#     elif query_engine is None and any(DATA_DIR.iterdir()):
#         print("Warning: Query engine could not be initialized, likely due to issues with document loading or indexing.")
#     else:
#         import uvicorn
#         uvicorn.run(app, host="0.0.0.0", port=8000)