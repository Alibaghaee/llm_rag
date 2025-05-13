FROM python:3.10-slim
# یا python:3.11-slim برای نسخه‌های جدیدتر

WORKDIR /rag_app

# نصب نیازمندی‌های سیستم (اگر کتابخانه‌ای نیاز به کامپایل داشت)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .

# توصیه: مشخص کردن نسخه torch و torchvision متناسب با CUDA اگر از GPU استفاده می‌کنید
# مثال برای CUDA 11.8:
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# سپس بقیه نیازمندی‌ها:
RUN pip install --no-cache-dir --default-timeout=100 \
    -r requirements.txt

COPY . .

# اطمینان از وجود پوشه data (اگرچه volume mount آن را مدیریت می‌کند)
RUN mkdir -p /rag_app/data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]