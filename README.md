# Economic News Bot

A self-hosted Telegram bot that aggregates and posts economic news, market updates, and calendar events to a Telegram channel.

## Features

- Monitors specified Telegram groups for relevant economic news and reposts them
- Pulls and posts from curated RSS feeds (e.g. Investing.com)
- Tracks economic calendar events
- Uses sentence embeddings to detect and skip duplicate posts
- Runs continuously as a background service (systemd)

## Tech Stack

- Python 3
- [Telethon](https://docs.telethon.dev/) — Telegram client API
- [sentence-transformers](https://www.sbert.net/) — semantic deduplication
- feedparser — RSS parsing
- python-dotenv — environment variable management

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/jisayo/economic-news-bot.git
   cd economic-news-bot
