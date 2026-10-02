import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from dateutil import parser as date_parser
from sentence_transformers import SentenceTransformer, util
from telethon import TelegramClient, events

import os
from dotenv import load_dotenv
load_dotenv()

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')

source_groups = ['TradingNewsIO']
target_group = 'economicsnews26'

rss_feeds = [
    'https://www.investing.com/rss/news_1.rss',
    'https://www.forexlive.com/feed/news',
    'https://www.fxstreet.com/rss/news',
]

RSS_CHECK_INTERVAL = 300
DUPLICATE_WINDOW_HOURS = 24
SIMILARITY_THRESHOLD = 0.85

CALENDAR_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
CALENDAR_REFRESH_INTERVAL = 3600
ALERT_CHECK_INTERVAL = 60
ALERT_LEAD_MINUTES = 60
ALERT_WINDOW_MINUTES = 2
ALLOWED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY'}
ALLOWED_IMPACTS = {'High', 'Medium'}

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

client = TelegramClient('session_name', api_id, api_hash)

seen_history = []


def prune_old_history():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_WINDOW_HOURS)
    global seen_history
    seen_history = [item for item in seen_history if item['timestamp'] > cutoff]


def is_duplicate(text):
    if not seen_history:
        return False
    embedding = model.encode(text, convert_to_tensor=True)
    for item in seen_history:
        similarity = util.cos_sim(embedding, item['embedding']).item()
        if similarity >= SIMILARITY_THRESHOLD:
            return True
    return False


def remember(text):
    embedding = model.encode(text, convert_to_tensor=True)
    seen_history.append({
        'embedding': embedding,
        'timestamp': datetime.now(timezone.utc),
        'text': text,
    })


@client.on(events.NewMessage(chats=source_groups))
async def handler(event):
    text = event.message.message or ''
    if not text.strip():
        await client.send_message(target_group, event.message)
        return

    prune_old_history()
    if is_duplicate(text):
        print(f"[Telegram] Skipped duplicate: {text[:60]}...")
        return

    await client.send_message(target_group, event.message)
    remember(text)
    print(f"[Telegram] Posted: {text[:60]}...")


posted_links = set()
POSTED_LINKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'posted_links.json')


def load_posted_links():
    global posted_links
    if os.path.exists(POSTED_LINKS_FILE):
        try:
            with open(POSTED_LINKS_FILE, 'r') as f:
                data = json.load(f)
                posted_links = set(data)
                print(f"Loaded {len(posted_links)} previously posted links.")
        except Exception as e:
            print(f"Could not load posted links file: {e}")
            posted_links = set()
    else:
        posted_links = set()


def save_posted_links():
    try:
        links_list = list(posted_links)[-2000:]
        with open(POSTED_LINKS_FILE, 'w') as f:
            json.dump(links_list, f)
    except Exception as e:
        print(f"Could not save posted links file: {e}")


load_posted_links()


async def check_rss_feeds():
    while True:
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    link = entry.get('link')
                    title = entry.get('title', '').strip()

                    if not link or link in posted_links:
                        continue
                    if not title:
                        continue

                    posted_links.add(link)
                    save_posted_links()

                    prune_old_history()
                    if is_duplicate(title):
                        print(f"[RSS] Skipped duplicate: {title[:60]}...")
                        continue

                    message_text = f"{title}\n{link}"
                    await client.send_message(target_group, message_text)
                    remember(title)
                    print(f"[RSS] Posted: {title[:60]}...")

            except Exception as e:
                print(f"[RSS] Error checking {feed_url}: {e}")

        await asyncio.sleep(RSS_CHECK_INTERVAL)


calendar_events = []
alerted_event_ids = set()
last_summary_date = None
last_summary_msg_id = None
ALERTED_EVENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerted_events.json')


def load_alerted_events():
    global alerted_event_ids
    if os.path.exists(ALERTED_EVENTS_FILE):
        try:
            with open(ALERTED_EVENTS_FILE, 'r') as f:
                alerted_event_ids = set(json.load(f))
        except Exception as e:
            print(f"Could not load alerted events file: {e}")
            alerted_event_ids = set()
    else:
        alerted_event_ids = set()


def save_alerted_events():
    try:
        ids_list = list(alerted_event_ids)[-2000:]
        with open(ALERTED_EVENTS_FILE, 'w') as f:
            json.dump(ids_list, f)
    except Exception as e:
        print(f"Could not save alerted events file: {e}")


load_alerted_events()


def event_id(evt):
    return f"{evt.get('country')}|{evt.get('title')}|{evt.get('date')}"


async def refresh_calendar():
    global calendar_events
    while True:
        try:
            response = requests.get(CALENDAR_URL, timeout=15)
            response.raise_for_status()
            raw_events = response.json()

            filtered = []
            for evt in raw_events:
                if evt.get('country') not in ALLOWED_CURRENCIES:
                    continue
                if evt.get('impact') not in ALLOWED_IMPACTS:
                    continue
                try:
                    evt['_parsed_date'] = date_parser.parse(evt.get('date'))
                except Exception:
                    continue
                filtered.append(evt)

            calendar_events = filtered
            print(f"[Calendar] Loaded {len(calendar_events)} relevant events.")

        except Exception as e:
            print(f"[Calendar] Error fetching calendar: {e}")

        await asyncio.sleep(CALENDAR_REFRESH_INTERVAL)


async def check_calendar_alerts():
    while True:
        now = datetime.now(timezone.utc)

        for evt in calendar_events:
            evt_time = evt.get('_parsed_date')
            if not evt_time:
                continue

            minutes_until = (evt_time - now).total_seconds() / 60
            eid = event_id(evt)

            if eid in alerted_event_ids:
                continue

            if (ALERT_LEAD_MINUTES - ALERT_WINDOW_MINUTES) <= minutes_until <= (ALERT_LEAD_MINUTES + ALERT_WINDOW_MINUTES):
                impact_emoji = "🔴" if evt.get('impact') == 'High' else "🟠"
                message = (
                    f"{impact_emoji} {evt.get('country')} - {evt.get('title')}\n"
                    f"⏰ In ~1 hour ({evt_time.strftime('%H:%M UTC')})\n"
                    f"Forecast: {evt.get('forecast') or 'N/A'} | Previous: {evt.get('previous') or 'N/A'}"
                )
                try:
                    await client.send_message(target_group, message)
                    alerted_event_ids.add(eid)
                    save_alerted_events()
                    print(f"[Calendar] Alerted: {evt.get('country')} {evt.get('title')}")
                except Exception as e:
                    print(f"[Calendar] Error sending alert: {e}")

        await asyncio.sleep(ALERT_CHECK_INTERVAL)


async def post_daily_summary():
    global last_summary_date, last_summary_msg_id
    while True:
        now = datetime.now(timezone.utc)
        today = now.date()

        if last_summary_date != today and now.hour == 0:
            todays_events = [
                evt for evt in calendar_events
                if evt.get('_parsed_date') and evt['_parsed_date'].date() == today
            ]
            todays_events.sort(key=lambda e: e['_parsed_date'])

            if todays_events:
                lines = [f"📅 Today's Economic Calendar ({today.strftime('%A, %B %d')})\n"]
                for evt in todays_events:
                    impact_emoji = "🔴" if evt.get('impact') == 'High' else "🟠"
                    time_str = evt['_parsed_date'].strftime('%H:%M UTC')
                    lines.append(f"{impact_emoji} {time_str} - {evt.get('country')} {evt.get('title')}")
                summary_message = "\n".join(lines)
            else:
                summary_message = f"📅 Today's Economic Calendar ({today.strftime('%A, %B %d')})\n\nNo major USD/EUR/GBP/JPY events scheduled today."

            try:
                if last_summary_msg_id:
                    try:
                        await client.unpin_message(target_group, last_summary_msg_id)
                    except Exception as e:
                        print(f"[Calendar] Error unpinning previous summary: {e}")

                sent_msg = await client.send_message(target_group, summary_message)
                await client.pin_message(target_group, sent_msg, notify=False)
                last_summary_msg_id = sent_msg.id
                last_summary_date = today
                print(f"[Calendar] Posted and pinned daily summary for {today}.")

            except Exception as e:
                print(f"[Calendar] Error posting daily summary: {e}")

        await asyncio.sleep(60)


async def main():
    await client.start()
    print("Bot is running... watching Telegram groups, RSS feeds, and economic calendar")
    await asyncio.gather(
        check_rss_feeds(),
        refresh_calendar(),
        check_calendar_alerts(),
        post_daily_summary(),
    )


with client:
    client.loop.run_until_complete(main())
