FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY music_genre_cnn.h5 .
COPY scaler.joblib .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]