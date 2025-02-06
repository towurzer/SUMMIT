FROM python:3.12.8-bookworm

WORKDIR /usr/src/app

# Install pip packages
COPY requirements.txt ./

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over files
COPY . .
COPY config_docker.json config.json

CMD ["python", "./src/main.py"]