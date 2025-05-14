from fastapi import FastAPI
from pydantic import BaseModel
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
import torch
import os
from pathlib import Path
import logging
import sys
import multiprocessing

# تنظیم لاگ برای نمایش بهتر اطلاعات
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# تنظیمات بهینه‌سازی CPU
num_physical_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_physical_cores)  # استفاده از تمام هسته‌های CPU
os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)
os.environ['MKL_NUM_THREADS'] = str(num_physical_cores)
print(f"Using {num_physical_cores} CPU threads")

app = FastAPI()

# --- پیکربندی اولیه ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# تنظیمات مدل و بهینه‌سازی
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "bigscience/bloomz-560m"  # تغییر به مدل BLOOMZ که instruction-tuned است
DATA_DIR = Path(__file__).parent / "data"

# تنظیمات بهینه‌سازی برای inference
BATCH_SIZE = 1
NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)

# --- بارگذاری مدل‌ها و ساخت ایندکس (در زمان شروع برنامه) ---
embed_model = None
index = None
query_engine = None
documents = []

try:
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=DEVICE)
except Exception as e:
    print(f"Error loading embedding model: {e}")

if embed_model:
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None,
    )

    print("Loading documents and building RAG index...")
    if not DATA_DIR.exists():
        print(f"Data directory '{DATA_DIR}' does not exist. Creating it.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not any(DATA_DIR.iterdir()):
        print(f"Warning: Data directory '{DATA_DIR}' is empty.")
        print("Please add your text files (e.g., .txt, .md).")
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

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(payload: QueryRequest):
    if query_engine is None or index is None or not documents:
        return {"answer": "متاسفانه، سیستم پرسش و پاسخ به درستی مقداردهی اولیه نشده است."}

    print(f"Received query: {payload.question}")
    
    # تشخیص زبان سوال - بهبود تشخیص فارسی
    try:
        # اضافه کردن برخی کاراکترهای فارسی برای تشخیص بهتر
        has_persian = any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF' for c in payload.question)
        is_persian = has_persian or detect(payload.question) == 'fa'
    except:
        # اگر تشخیص زبان با خطا مواجه شد، بر اساس وجود کاراکترهای فارسی تصمیم می‌گیریم
        is_persian = any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF' for c in payload.question)
    
    print(f"Detected language: {'Persian' if is_persian else 'English'}")

    retrieved_nodes = query_engine.retrieve(payload.question)

    if not retrieved_nodes:
        return {"answer": "متنی مرتبط با سوال شما در اسناد موجود یافت نشد." if is_persian else "No relevant information found in the context."}

    context_chunks = [node.get_content() for node in retrieved_nodes]
    context = "\n\n---\n\n".join(context_chunks)

    print(f"Retrieved context: \n{context[:500]}...")

    # بهبود prompt برای پاسخ‌های فارسی با استفاده از فرمت مناسب برای مدل instruction-tuned
    if is_persian:
        prompt = (
            "با توجه به متن زیر، به سوال پاسخ دهید. پاسخ باید دقیق و فقط بر اساس اطلاعات متن باشد. "
            "اگر اطلاعات در متن نیست، بگویید 'اطلاعات مورد نظر در متن یافت نشد'.\n\n"
            "متن:\n"
            "------------\n"
            f"{context}\n"
            "------------\n\n"
            "سوال: " + payload.question + "\n"
            "پاسخ:"
        )
    else:
        prompt = (
            "Based on the following text, answer the question. "
            "The answer must be precise and based only on the information in the text. "
            "If the information is not in the text, say 'Information not found in the text'.\n\n"
            "Text:\n"
            "------------\n"
            f"{context}\n"
            "------------\n\n"
            "Question: " + payload.question + "\n"
            "Answer:"
        )

    print(f"Constructed prompt for generation model: \n{prompt}")

    # بهینه‌سازی تنظیمات generation برای مدل instruction-tuned
    generation_config = {
        'max_new_tokens': 200,
        'num_beams': 3,
        'temperature': 0.3,  # کاهش temperature برای پاسخ‌های دقیق‌تر
        'top_p': 0.9,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1.0,
        'use_cache': True,
        'do_sample': False,  # غیرفعال کردن sampling برای پاسخ‌های قطعی‌تر
        'num_return_sequences': 1,
    }

    with torch.no_grad():
        model.eval()
        with torch.inference_mode():
            outputs = model.generate(
                **tokenizer(prompt, return_tensors='pt').to(DEVICE),
                **generation_config
            )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw generated answer: {answer}")  # اضافه کردن لاگ برای عیب‌یابی

    # بهبود پردازش پاسخ
    if is_persian:
        # حذف همه عبارات احتمالی قبل از پاسخ اصلی
        possible_prefixes = ["پاسخ به فارسی:", "پاسخ:", "Answer in Persian:", "Answer:", "Persian answer:", "Response in Persian:"]
        for prefix in possible_prefixes:
            if prefix in answer:
                answer = answer.split(prefix)[-1].strip()
                break
    else:
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

    # حذف سوال از پاسخ اگر تکرار شده باشد
    if payload.question in answer:
        answer = answer.split(payload.question)[-1].strip()
    
    # حذف کاراکترهای اضافی از ابتدای پاسخ
    answer = answer.lstrip(":؟.,!?")
    answer = answer.strip()

    print(f"Final processed answer: {answer}")
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