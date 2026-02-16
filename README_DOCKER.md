# Docker Deployment

## Build

```bash
docker build -t novel-writer:latest .
```

## Run

### Data Pipeline

```bash
docker run -it \
  -v $(pwd)/data:/app/data \
  novel-writer:latest \
  novel-writer pipeline --clean
```

### API Server

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  novel-writer:latest \
  python -m novel_writer.api
```

### Dashboard

```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  novel-writer:latest \
  streamlit run novel_writer/dashboard.py --server.port=8501
```

## Docker Compose

Run all services:

```bash
docker-compose up -d
```

Stop all services:

```bash
docker-compose down
```

## GPU Support

For GPU acceleration, use nvidia-docker:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# ... rest of Dockerfile
```

Then run:

```bash
docker run --gpus all -it novel-writer:latest ...
```
