FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install streamlit requests Pillow

# Copy source code
COPY streamlit_app/ streamlit_app/
COPY src/ src/

# Expose port
EXPOSE 8501

# Entrypoint
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
