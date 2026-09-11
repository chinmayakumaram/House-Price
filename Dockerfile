FROM python:3.11-slim

WORKDIR /workspace

# System deps (kept minimal; tensorflow needs libgomp on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Default command runs the Streamlit prediction app.
# Override with e.g. `docker run <image> python src/train_ml.py` to run training instead.
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
