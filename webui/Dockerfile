FROM atinoda/text-generation-webui:latest



RUN apt-get update && apt-get install -y python3-yaml && \
    pip install --no-cache-dir pyyaml ruamel.yaml && \
    rm -rf /var/lib/apt/lists/*

# نصب PyYAML و ruamel.yaml برای رفع خطای ModuleNotFoundError: yaml
RUN pip install --no-cache-dir pyyaml ruamel.yaml