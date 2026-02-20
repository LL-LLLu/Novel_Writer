# Multi-Agent Story Generator v2

8 specialized AI agents collaborate to write high-quality fiction through a multi-round debate architecture.

## Architecture

| Agent | Role | Model |
|-------|------|-------|
| Outline Architect | Generate/parse story outlines | Gemini |
| Style Analyzer | Extract writing rules from guidance | Gemini |
| Mastermind | Plan chapters scene-by-scene | Gemini |
| Memory Agent | Track story bible across chapters | Gemini |
| Writer | Generate prose | Qwen 3.5 Plus |
| Continuity Critic | Review plot/character consistency | Gemini |
| Style Critic | Review style/guidance compliance | Gemini |
| Judge | Synthesize feedback, approve/revise | Gemini |

**Flow per chapter:** Mastermind plans → Writer drafts → Critics review (parallel) → Judge decides → Writer revises (up to 3 rounds)

## Setup

```bash
pip install -r requirements.txt
```

You need two API keys:
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)
- **Qwen (DashScope) API Key** from [Alibaba Cloud](https://dashscope.console.aliyun.com/)

## Run

```bash
python -m novel_writer_v2 --port 7860
```

Open `http://localhost:7860` in your browser.

## Usage

1. **Setup tab**: Enter API keys, select models, configure debate rounds
2. **Story Workshop tab**: Write a premise, generate outline, generate/upload guidance, then generate chapters
3. **Story Bible tab**: View/initialize characters, plot threads, timeline
4. **Full Story tab**: Read combined chapters, export as .txt
5. **Agent Log tab**: Monitor agent activity and debate rounds

## Supported Languages

- English
- Chinese (中文)
- Auto-detect
