FROM condaforge/miniforge3:latest

RUN conda install python=3.9 -y

WORKDIR /app

COPY main.py .
COPY requirements.txt .

RUN python -m pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
