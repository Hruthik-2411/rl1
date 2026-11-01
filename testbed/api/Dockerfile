FROM python:3.11-slim

# stress-ng for CPU faults, iproute2 for tc/netem network faults, procps for pgrep/pkill
RUN apt-get update \
 && apt-get install -y --no-install-recommends stress-ng iproute2 procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 5000
CMD ["python", "app.py"]
