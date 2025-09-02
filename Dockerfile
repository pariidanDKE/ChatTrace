# start from base image
FROM python:3.11-slim
# setup working direcory ( i.e a folder inside my container where
WORKDIR /app
# upgrade pip
RUN pip install --upgrade pip
# copy special linux dependency list and install them
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

### copy rest of the app ( excluding the dockerignore files)
COPY . .
### run the app
CMD ["python", "app.py"]
