FROM python:3.12-slim

WORKDIR /app

# Install dependencies, including psycopg2 requirements for Neon PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory for secrets
RUN mkdir -p /root/.streamlit

# The app and tests will be mounted as a volume in docker-compose,
# but we set the default command to run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
