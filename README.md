# 📚 LLM-RAG Persian - فارسی بدون نیاز به OpenAI

پروژه‌ای سبک برای پاسخ‌گویی به پرسش‌های کاربران بر اساس اسناد شخصی، بدون نیاز به کلید API از OpenAI، با پشتیبانی از زبان فارسی.  
از معماری RAG (Retrieval-Augmented Generation) به‌صورت ساده و بدون مدل‌های بزرگ استفاده شده است.

## 🧠 تکنولوژی‌های استفاده‌شده

- [FastAPI](https://fastapi.tiangolo.com/) - برای ساخت REST API
- [LlamaIndex](https://github.com/jerryjliu/llama_index) - برای ایندکس‌گذاری و بازیابی
- [HuggingFace Transformers](https://huggingface.co/transformers/) - برای مدل پرسش‌پاسخ (Extractive QA)
- [SentenceTransformers](https://www.sbert.net/) - برای تولید بردار متون
- مدل‌ها:
  - 🔍 `all-MiniLM-L6-v2` (برای Embedding)
  - ❓ `distilbert-base-cased-distilled-squad` (برای پرسش‌پاسخ)

## 📦 نصب و اجرا

### 1. کلون کردن پروژه

```bash
git clone https://github.com/username/llm-rag-persian.git
cd llm-rag-persian
