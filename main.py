#!/usr/bin/env python3
"""
main.py — Yaarn Daily Nigerian News
Scrapes → Groq filters → Groq scripts → Kokoro TTS → MoviePy builds → Instagram + Telegram
"""

import os, json, re, hashlib, textwrap, time, random, math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import feedparser
import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from groq import Groq
from dotenv import load_dotenv
from moviepy import (
    VideoClip, ImageClip, AudioFileClip, concatenate_videoclips,
    CompositeAudioClip, concatenate_audioclips
)

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
KOKORO_API_URL   = os.getenv("KOKORO_API_URL")
KOKORO_API_KEY   = os.getenv("KOKORO_API_KEY")
IG_ACCOUNT_ID    = os.getenv("INSTAGRAM_ACCOUNT_ID", "17841422988712010")
FB_PAGE_TOKEN    = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN")
GITHUB_REPO      = os.getenv("GITHUB_REPO", "")   # e.g. "Techie-John/yaarn-news"

GROQ_MODEL    = "llama-3.3-70b-versatile"
VIDEO_W       = 1080
VIDEO_H       = 1920
FPS           = 24
MAX_STORIES   = 15
HOURS_BACK    = 18
SIM_THRESH    = 0.55
OUTPUT_DIR    = Path("output")
MEMORY_FILE   = Path("story_memory.json")
BG_MUSIC_FILE = "audio.mp3"
BG_MUSIC_VOL  = 0.07

BLACK = (8, 8, 8)
WHITE = (255, 255, 255)
GREY  = (150, 150, 150)

RSS_SOURCES = [
    {"name": "Punch",           "url": "https://punchng.com/feed/",                "priority": 1},
    {"name": "Channels TV",     "url": "https://www.channelstv.com/feed/",         "priority": 2},
    {"name": "Premium Times",   "url": "https://www.premiumtimesng.com/feed",      "priority": 3},
    {"name": "Vanguard",        "url": "https://www.vanguardngr.com/feed/",        "priority": 4},
    {"name": "The Guardian NG", "url": "https://guardian.ng/feed/",                "priority": 5},
    {"name": "Sahara Reporters","url": "https://saharareporters.com/rss.xml",      "priority": 6},
    {"name": "Daily Trust",     "url": "https://dailytrust.com/feed",              "priority": 7},
]

NIGERIA_KEYWORDS = {
    "naira", "fuel", "petrol", "electricity", "tariff", "inflation", "price",
    "cbn", "bank", "loan", "tax", "budget", "economy", "forex", "dollar",
    "market", "subsidy", "power", "nnpc", "dangote", "killed", "attack",
    "bandits", "kidnap", "abduct", "troops", "military", "police", "bomb",
    "explosion", "crisis", "conflict", "shooting", "dead", "victims",
    "rescue", "terror", "boko haram", "ipob", "strike", "protest",
    "shutdown", "resign", "impeach", "arrest", "court", "sentence",
    "convicted", "efcc", "icpc", "suspended", "sacked", "fired",
    "tinubu", "minister", "governor", "senate", "inec", "election",
    "asuu", "nlc", "hospital", "flood", "fire", "crash", "accident",
    "disease", "epidemic", "hunger", "food", "water", "road", "bridge",
    "school", "university", "lagos", "abuja", "kano", "ibadan", "rivers",
    "breaking", "just in", "urgent", "emergency", "update", "nigeria",
    "nigerian", "federal", "state", "government",
}

# ── Nigerian states: checked first so location-specific news gets the right image ──
NIGERIAN_STATES = {
    "delta":    ["Delta State Nigeria", "Asaba Nigeria"],
    "lagos":    ["Lagos Nigeria skyline", "Lagos Island Nigeria"],
    "abuja":    ["Abuja Nigeria", "Aso Rock Abuja"],
    "kano":     ["Kano Nigeria", "Emir of Kano palace"],
    "rivers":   ["Port Harcourt Nigeria", "Rivers State Nigeria"],
    "anambra":  ["Anambra State Nigeria", "Awka Nigeria"],
    "enugu":    ["Enugu Nigeria"],
    "imo":      ["Imo State Nigeria", "Owerri Nigeria"],
    "ogun":     ["Ogun State Nigeria", "Abeokuta Nigeria"],
    "oyo":      ["Ibadan Nigeria", "Oyo State Nigeria"],
    "kaduna":   ["Kaduna Nigeria"],
    "katsina":  ["Katsina Nigeria"],
    "zamfara":  ["Zamfara State Nigeria"],
    "sokoto":   ["Sokoto Nigeria"],
    "borno":    ["Borno State Nigeria", "Maiduguri Nigeria"],
    "adamawa":  ["Adamawa State Nigeria", "Yola Nigeria"],
    "benue":    ["Benue State Nigeria", "Makurdi Nigeria"],
    "plateau":  ["Jos Plateau Nigeria", "Plateau State Nigeria"],
    "niger":    ["Niger State Nigeria", "Minna Nigeria"],
    "kwara":    ["Kwara State Nigeria", "Ilorin Nigeria"],
    "osun":     ["Osun State Nigeria", "Osogbo Nigeria"],
    "ekiti":    ["Ekiti State Nigeria"],
    "ondo":     ["Ondo State Nigeria", "Akure Nigeria"],
    "edo":      ["Benin City Nigeria", "Edo State Nigeria"],
    "cross river": ["Cross River State Nigeria", "Calabar Nigeria"],
    "akwa ibom": ["Akwa Ibom State Nigeria", "Uyo Nigeria"],
    "bayelsa":  ["Bayelsa State Nigeria", "Yenagoa Nigeria"],
    "taraba":   ["Taraba State Nigeria"],
    "gombe":    ["Gombe Nigeria"],
    "yobe":     ["Yobe State Nigeria"],
    "bauchi":   ["Bauchi Nigeria"],
    "jigawa":   ["Jigawa State Nigeria"],
    "kebbi":    ["Kebbi State Nigeria"],
    "nasarawa": ["Nasarawa State Nigeria"],
    "ebonyi":   ["Ebonyi State Nigeria"],
    "abia":     ["Abia State Nigeria", "Umuahia Nigeria"],
}

# ── Topic → relevant Nigerian entity images ────────────────────────────
TOPIC_IMAGE_MAP = [
    (["naira", "cbn", "inflation", "forex", "dollar", "bank", "loan",
      "tax", "budget", "economy", "market", "subsidy"],
     ["Central Bank of Nigeria", "Nigerian naira banknote",
      "Central Bank of Nigeria headquarters"]),

    (["fuel", "petrol", "nnpc", "dangote", "electricity", "power", "tariff"],
     ["Dangote Refinery", "NNPC filling station Nigeria",
      "Nigeria fuel queue"]),

    (["bandits", "kidnap", "abduct", "troops", "military", "police",
      "bomb", "explosion", "attack", "killed", "shooting", "terror",
      "boko haram", "ipob", "rescue", "conflict", "soldier"],
     ["Nigerian Army", "Nigeria Police Force",
      "Nigerian Armed Forces"]),

    (["strike", "protest", "asuu", "nlc", "shutdown"],
     ["ASUU Nigeria", "Nigeria protest",
      "Nigerian Labour Congress"]),

    (["flood", "fire", "crash", "accident", "disaster"],
     ["Lagos flooding", "Nigeria flood disaster"]),

    (["disease", "epidemic", "hospital", "health", "lassa"],
     ["University of Lagos Teaching Hospital",
      "Nigeria health workers"]),

    (["tinubu", "minister", "governor", "senate", "inec",
      "election", "government", "federal", "presidency"],
     ["National Assembly Nigeria", "Aso Rock Nigeria",
      "Presidential Villa Abuja"]),

    (["efcc", "icpc", "court", "arrest", "convicted",
      "sentence", "suspended", "sacked", "fired", "impeach"],
     ["EFCC Nigeria", "Nigeria court"]),

    (["school", "university", "asuu", "education"],
     ["University of Ibadan", "University of Lagos campus"]),

    (["food", "hunger", "market", "farming", "agriculture"],
     ["Nigeria market food", "Balogun Market Lagos"]),
]

# ── Generic Nigeria fallbacks — never gradient ────────────────────────
GENERIC_NIGERIA_IMAGES = [
    "Flag of Nigeria",
    "Coat of arms of Nigeria",
    "Nigeria map",
    "Abuja Nigeria",
    "Lagos Nigeria skyline",
    "National Assembly Nigeria",
    "River Niger Nigeria",
    "Zuma Rock Nigeria",
]


# ══════════════════════════════════════════════════════════════════════
# STORY MEMORY
# ══════════════════════════════════════════════════════════════════════

def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return {}


def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def _story_key(title):
    return hashlib.md5(title.lower().strip().encode()).hexdigest()


def is_seen(title, memory):
    return _story_key(title) in memory


def mark_seen(title, memory):
    memory[_story_key(title)] = datetime.now(timezone.utc).isoformat()
    return memory


def cleanup_memory(memory):
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    return {k: v for k, v in memory.items()
            if datetime.fromisoformat(v) > cutoff}


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — SCRAPE
# ══════════════════════════════════════════════════════════════════════

def scrape_news():
    print("\n[1/6] Scraping Nigerian news feeds...")
    cutoff      = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    memory      = load_memory()
    all_stories = []

    for source in RSS_SOURCES:
        try:
            feed  = feedparser.parse(source["url"])
            count = 0
            for entry in feed.entries[:30]:
                published = _parse_date(entry)
                if published and published < cutoff:
                    continue
                title   = entry.get("title", "").strip()
                summary = _clean_html(entry.get("summary", ""))
                link    = entry.get("link", "")
                if not title:
                    continue
                if not _is_relevant(title + " " + summary):
                    continue
                if is_seen(title, memory):
                    continue
                all_stories.append({
                    "title":     title,
                    "link":      link,
                    "summary":   summary,
                    "published": published,
                    "source":    source["name"],
                    "priority":  source["priority"],
                    "image_url": _get_og_image(link),
                })
                count += 1
            print(f"  ✓ {source['name']}: {count} stories")
        except Exception as e:
            print(f"  ✗ {source['name']}: {e}")

    print(f"  Raw total  : {len(all_stories)} new stories")
    deduped = _deduplicate(all_stories)
    print(f"  After dedup: {len(deduped)} stories")
    return deduped, memory


def _is_relevant(text):
    return any(kw in text.lower() for kw in NIGERIA_KEYWORDS)


def _deduplicate(stories):
    stories.sort(key=lambda x: (
        x["priority"],
        -(x["published"].timestamp() if x["published"] else 0)
    ))
    seen_titles, source_counts, result = [], {}, []
    for s in stories:
        src = s["source"]
        if source_counts.get(src, 0) >= 5:
            continue
        title_lower = s["title"].lower()
        if any(SequenceMatcher(None, title_lower, seen).ratio() >= SIM_THRESH
               for seen in seen_titles):
            continue
        result.append(s)
        seen_titles.append(title_lower)
        source_counts[src] = source_counts.get(src, 0) + 1
        if len(result) >= 50:
            break
    return result


def _parse_date(entry):
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _get_og_image(url):
    if not url:
        return None
    try:
        r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        soup = BeautifulSoup(r.text, "html.parser")
        for prop in [("property", "og:image"), ("name", "twitter:image")]:
            tag = soup.find("meta", {prop[0]: prop[1]})
            if tag and tag.get("content"):
                return tag["content"]
    except Exception:
        pass
    return None


def _clean_html(raw):
    try:
        return BeautifulSoup(raw, "html.parser").get_text(" ").strip()
    except Exception:
        return raw.strip()


# ══════════════════════════════════════════════════════════════════════
# STAGE 2A — GROQ FILTER
# ══════════════════════════════════════════════════════════════════════

def filter_stories(stories):
    print("\n[2/6] Filtering stories for impact...")
    if not stories:
        return []
    stories_text = "\n".join(
        f"[{i+1}] {s['title']} ({s['source']})"
        for i, s in enumerate(stories)
    )
    prompt = f"""You are a Nigerian news editor at a big TV station.
Pick the {MAX_STORIES} stories that matter most to everyday Nigerians.

PICK stories about:
- Fuel price, electricity bills, naira rate — things people feel in their pocket
- Security — attacks, kidnapping, killings, military operations
- Strikes, protests, road closures — things that affect daily movement
- Natural disasters — floods, fires, building collapse
- Health emergencies — disease outbreaks, hospital issues
- Big government actions that directly affect ordinary people
- Breaking news everyone will be talking about today

DO NOT PICK:
- Routine government meetings and committee setups
- Award ceremonies and inaugurations with no real impact
- Press conferences that say nothing new
- Political back-and-forth with no direct effect on people

Stories:
{stories_text}

Return ONLY a JSON array of story numbers. Example: [1, 3, 5, 7, 8, 11, 14, 16]
Pick exactly {MAX_STORIES} or fewer if not enough qualify.
Return ONLY the JSON array. Nothing else."""

    client = Groq(api_key=GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    raw   = resp.choices[0].message.content.strip()
    match = re.search(r'\[[\d,\s]+\]', raw)
    if not match:
        return stories[:MAX_STORIES]
    indices  = json.loads(match.group(0))
    selected = [stories[i - 1] for i in indices if 1 <= i <= len(stories)]
    print(f"  ✓ {len(selected)} high-impact stories selected")
    return selected


# ══════════════════════════════════════════════════════════════════════
# STAGE 2B — GROQ SCRIPT
# ══════════════════════════════════════════════════════════════════════

def generate_script(stories):
    print("\n[3/6] Writing script...")
    today        = datetime.now(timezone.utc).date()
    yesterday    = today - timedelta(days=1)
    display_date = f"{today.day} {today.strftime('%B %Y')}"

    def _timing_label(story):
        pub = story.get("published")
        if pub:
            pub_date = pub.astimezone(timezone.utc).date()
            if pub_date >= today:
                return "TODAY"
            elif pub_date >= yesterday:
                return "YESTERDAY"
        return "TODAY"

    stories_text = "\n\n".join(
        f"STORY {i+1} [WHEN: {_timing_label(s)}]\n"
        f"Headline: {s['title']}\n"
        f"Details: {s['summary'][:300]}"
        for i, s in enumerate(stories)
    )

    system_prompt = f"""You are the scriptwriter for Yaarn — Nigeria's daily news channel.
Write a strictly neutral, factual news script. No opinions. No bias.

TODAY'S DATE: {display_date}

YOUR STYLE:
- Very simple English. Grade 4 level. Short words. Short sentences.
- Report ONLY facts — what happened, where, and what it means.
- 100% NEUTRAL. Do NOT condemn or praise any person, group, or government.
- Do NOT use emotional words: "shocking", "outrageous", "alarming", "sad", "unfortunately".
- Do NOT take sides on any political, ethnic, or religious issue.
- Clear. Direct. Factual.

MONEY RULE — VERY IMPORTANT:
- NEVER write currency as symbols or abbreviations like NGN, ₦, N, $, USD, bn, m, k.
- ALWAYS write money amounts fully in words.
- Examples:
  - "₦600 million" → "six hundred million naira"
  - "N1.2 billion" → "one point two billion naira"
  - "$50,000" → "fifty thousand dollars"
  - "₦10bn" → "ten billion naira"
  - "N500m" → "five hundred million naira"
  - "N50k" → "fifty thousand naira"

VARY YOUR LANGUAGE — VERY IMPORTANT:
- Do NOT start every story with "Today" or "This morning" or "Yesterday".
- Only ONE or TWO stories in the whole script should mention a time word at all.
- All other stories start straight with the fact itself — no time word.
  Good openers: "Fuel price has gone up in Lagos." / "A bridge in Kano has collapsed."
  "Security forces arrested three suspects in Abuja." / "The Senate passed a new bill."

STRICT RULES:
1. INTRO: Start exactly with: "This is what happened in Nigeria today, {display_date}."
   Then name the top 3 stories. One short sentence each.
2. Each story narration: 1 sentence. Use a 2nd sentence ONLY if essential.
3. Short display headline: max 5 words. No full stop.
4. OUTRO: Short. Tell people to follow Yaarn. Neutral tone.
5. NO URLs. NO "according to". NO "reportedly". NO "it was gathered".
6. Return ONLY valid JSON. No markdown.

JSON FORMAT:
{{
  "date": "{display_date}",
  "intro": "<intro narration>",
  "stories": [
    {{"headline": "<max 5 words>", "narration": "<1-2 sentences>"}}
  ],
  "outro": "<short follow CTA>"
}}"""

    client = Groq(api_key=GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Write the Yaarn script for these {len(stories)} stories:\n\n{stories_text}"},
        ],
        temperature=0.6,
        max_tokens=5000,
    )
    print(f"  Groq finish_reason: {resp.choices[0].finish_reason}")
    raw = resp.choices[0].message.content.strip()

    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$",       "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group(0)
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)

    try:
        script = json.loads(raw)
    except json.JSONDecodeError:
        print("  ⚠ JSON truncated — attempting repair...")
        raw    = _repair_truncated_json(raw)
        script = json.loads(raw)

    # Post-process: convert any remaining currency abbreviations to words
    script = _fix_currency_in_script(script)

    print(f"  ✓ Script ready — {len(script['stories'])} stories")
    return script


def _fix_currency_in_script(script):
    """Safety net: convert any leftover currency shorthand to spoken words."""
    def _fix(text):
        # ₦ / N / NGN amounts — handle bn, m, k suffixes
        def _replace_naira(m):
            amount_str = m.group(1).replace(",", "")
            suffix     = m.group(2).lower() if m.group(2) else ""
            try:
                amount = float(amount_str)
            except ValueError:
                return m.group(0)
            multipliers = {"bn": 1_000_000_000, "b": 1_000_000_000,
                           "m": 1_000_000,      "k": 1_000}
            if suffix in multipliers:
                amount *= multipliers[suffix]
            return _num_to_naira_words(int(amount))

        text = re.sub(
            r"(?:₦|NGN\s*|(?<!\w)N\s*)([\d,]+(?:\.\d+)?)\s*(bn|b|m|k)?",
            _replace_naira, text, flags=re.IGNORECASE
        )
        # $ amounts
        def _replace_dollar(m):
            amount_str = m.group(1).replace(",", "")
            suffix     = m.group(2).lower() if m.group(2) else ""
            try:
                amount = float(amount_str)
            except ValueError:
                return m.group(0)
            multipliers = {"bn": 1_000_000_000, "b": 1_000_000_000,
                           "m": 1_000_000,      "k": 1_000}
            if suffix in multipliers:
                amount *= multipliers[suffix]
            return _num_to_dollar_words(int(amount))

        text = re.sub(
            r"\$([\d,]+(?:\.\d+)?)\s*(bn|b|m|k)?",
            _replace_dollar, text, flags=re.IGNORECASE
        )
        return text

    script["intro"] = _fix(script["intro"])
    script["outro"] = _fix(script["outro"])
    for s in script["stories"]:
        s["narration"] = _fix(s["narration"])
    return script


def _num_to_words(n):
    """Convert integer n to English words (handles up to trillions)."""
    if n == 0:
        return "zero"
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]

    def _below_1000(num):
        if num == 0:
            return ""
        elif num < 20:
            return ones[num]
        elif num < 100:
            return tens[num // 10] + (" " + ones[num % 10] if num % 10 else "")
        else:
            return ones[num // 100] + " hundred" + (" " + _below_1000(num % 100) if num % 100 else "")

    parts = []
    scales = [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"),
              (1_000_000, "million"),           (1_000, "thousand")]
    for scale, name in scales:
        if n >= scale:
            parts.append(_below_1000(n // scale) + " " + name)
            n %= scale
    if n > 0:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _num_to_naira_words(n):
    return _num_to_words(n) + " naira"


def _num_to_dollar_words(n):
    return _num_to_words(n) + " dollars"


def _repair_truncated_json(s):
    if s.count('"') % 2 != 0:
        s += '"'
    stack, in_str, escape = [], False, False
    for ch in s:
        if escape:
            escape = False; continue
        if ch == '\\' and in_str:
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if not in_str:
            if ch in '{[':  stack.append(ch)
            elif ch in '}]' and stack: stack.pop()
    closers = {'{': '}', '[': ']'}
    for opener in reversed(stack):
        s += closers[opener]
    return s


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — AUDIO
# ══════════════════════════════════════════════════════════════════════

def generate_audio(script, output_dir):
    print("\n[4/6] Generating audio (Kokoro bm_george)...")
    segments = [("intro", script["intro"])]
    for i, s in enumerate(script["stories"]):
        segments.append((f"story_{i+1:02d}", s["narration"]))
    segments.append(("outro", script["outro"]))

    audio_files = {}
    for name, text in segments:
        response = requests.post(
            KOKORO_API_URL,
            json={"text": text, "voice": "bm_george",
                  "speed": 0.94, "api_key": KOKORO_API_KEY},
            timeout=60,
        )
        response.raise_for_status()
        path = output_dir / f"audio_{name}.wav"
        path.write_bytes(response.content)
        audio_files[name] = str(path)
        print(f"  ✓ {name}")
    return audio_files


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — VIDEO
# ══════════════════════════════════════════════════════════════════════

def build_video(script, stories, audio_files, output_dir):
    print("\n[5/6] Building video...")

    # Track which Wikipedia URLs have already been used this run
    # so no image repeats across slides.
    used_wiki_urls = set()

    images = [None] * len(stories)

    def _fetch_one(args):
        idx, s = args
        context = s["title"] + " " + s.get("summary", "")
        return idx, _fetch_image(s.get("image_url"), context, used_wiki_urls)

    # Fetch sequentially so used_wiki_urls tracking is thread-safe
    for idx, s in enumerate(stories):
        _, img = _fetch_one((idx, s))
        images[idx] = img
    print(f"  ✓ {len(images)} story images ready (no repeats, no blanks)")

    clips = []

    intro_audio = AudioFileClip(audio_files["intro"])
    intro_frame = _make_collage_intro(script["date"], images, used_wiki_urls)
    intro_clip  = _static_clip(intro_frame, intro_audio.duration).with_audio(intro_audio)
    clips.append(intro_clip)

    for i, (story_data, pil_image) in enumerate(zip(script["stories"], images)):
        key        = f"story_{i+1:02d}"
        audio      = AudioFileClip(audio_files[key])
        frame      = _make_story_frame(pil_image, story_data["headline"])
        story_clip = _zoom_clip(frame, audio.duration).with_audio(audio)
        clips.append(story_clip)
        print(f"  ✓ Story {i+1}/{len(stories)}: {story_data['headline'][:55]}")

    outro_audio = AudioFileClip(audio_files["outro"])
    outro_bg    = _pick_outro_image(images, used_wiki_urls)
    outro_frame = _make_outro_frame(outro_bg)
    outro_clip  = _static_clip(outro_frame, outro_audio.duration).with_audio(outro_audio)
    clips.append(outro_clip)

    final = concatenate_videoclips(clips, method="compose")
    final = _mix_background_music(final)

    out_path = str(output_dir / "video.mp4")
    final.write_videofile(out_path, fps=FPS, codec="libx264",
                          audio_codec="aac", logger=None)
    print(f"  ✓ Video saved → {out_path}")
    return out_path


def _mix_background_music(video_clip):
    if not os.path.exists(BG_MUSIC_FILE):
        return video_clip
    try:
        bg = AudioFileClip(BG_MUSIC_FILE)
        if bg.duration < video_clip.duration:
            loops = math.ceil(video_clip.duration / bg.duration)
            bg    = concatenate_audioclips([bg] * loops)
        bg    = bg.subclipped(0, video_clip.duration).with_volume_scaled(BG_MUSIC_VOL)
        mixed = CompositeAudioClip([video_clip.audio, bg])
        print(f"  ✓ Background music mixed at {int(BG_MUSIC_VOL*100)}% volume")
        return video_clip.with_audio(mixed)
    except Exception as e:
        print(f"  ⚠ Background music error: {e}")
        return video_clip


# ── Frame composers ───────────────────────────────────────────────────

def _make_collage_intro(date_str, images, used_wiki_urls):
    canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), BLACK)
    cols, rows  = 3, 4
    cell_w      = VIDEO_W // cols
    cell_h      = VIDEO_H // rows
    total_cells = cols * rows

    valid = [im for im in images if im is not None]
    pool  = list(valid)
    if len(pool) < total_cells:
        needed  = total_cells - len(pool)
        fillers = _get_generic_nigeria_images(needed, used_wiki_urls)
        pool.extend(fillers)
    if not pool:
        pool = [_gradient_fallback("Nigeria")]

    for idx in range(total_cells):
        img_src = pool[idx % len(pool)].convert("RGB")
        iw, ih  = img_src.size
        scale   = max(cell_w / iw, cell_h / ih)
        nw, nh  = int(iw * scale), int(ih * scale)
        thumb   = img_src.resize((nw, nh), Image.LANCZOS)
        cx, cy  = (nw - cell_w) // 2, (nh - cell_h) // 2
        thumb   = thumb.crop((cx, cy, cx + cell_w, cy + cell_h))
        canvas.paste(thumb, ((idx % cols) * cell_w, (idx // cols) * cell_h))

    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 150))
    canvas  = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw    = ImageDraw.Draw(canvas)
    draw.text((90, 520),  "YAARN",          font=_font(180), fill=WHITE)
    draw.rectangle([90, 750, VIDEO_W - 90, 760], fill=WHITE)
    draw.text((90, 786),  date_str.upper(), font=_font(48),  fill=GREY)
    return np.array(canvas)


def _draw_text_with_shadow(draw, xy, text, font, fill=WHITE, shadow_offset=5):
    x, y = xy
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def _make_story_frame(pil_image, headline):
    img = _fill_canvas(pil_image)
    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for y in range(VIDEO_H // 2, VIDEO_H):
        alpha = int(190 * ((y - VIDEO_H // 2) / (VIDEO_H // 2)))
        ov_draw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
    img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    bar_y = VIDEO_H - 560
    draw.rectangle([90, bar_y, 200, bar_y + 10], fill=WHITE)

    f_size = 100
    min_fs = 64
    wrap_w = 13
    while f_size >= min_fs:
        lines   = textwrap.wrap(headline.upper(), width=wrap_w)
        total_h = len(lines) * (f_size + 20)
        if bar_y + 40 + total_h < VIDEO_H - 60:
            break
        f_size -= 6

    y = bar_y + 40
    for line in textwrap.wrap(headline.upper(), width=wrap_w):
        _draw_text_with_shadow(draw, (90, y), line, font=_font(f_size))
        y += f_size + 20

    draw.text((VIDEO_W - 36, VIDEO_H - 44), "YAARN",
              font=_font(30), fill=(170, 170, 170), anchor="rs")
    return np.array(img)


def _pick_outro_image(images, used_wiki_urls):
    valid = [im for im in images if im is not None]
    if valid:
        return random.choice(valid)
    pool = _get_generic_nigeria_images(1, used_wiki_urls)
    return pool[0]


def _make_outro_frame(bg_image):
    img  = _fill_canvas(bg_image)
    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 165))
    img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_text_with_shadow(draw, (90, 620),  "FOLLOW", font=_font(80))
    _draw_text_with_shadow(draw, (90, 730),  "YAARN",  font=_font(180))
    draw.rectangle([90, 960, VIDEO_W - 90, 970], fill=WHITE)
    _draw_text_with_shadow(draw, (90, 996),  "Nigerian news, every day.", font=_font(52), fill=GREY)
    _draw_text_with_shadow(draw, (90, 1078), "@yaarn.ng",                 font=_font(52), fill=GREY)
    return np.array(img)


# ── MoviePy helpers ───────────────────────────────────────────────────

def _static_clip(frame_array, duration):
    return ImageClip(frame_array).with_duration(duration)


def _zoom_clip(frame_array, duration):
    img_pil = Image.fromarray(frame_array)
    w, h    = img_pil.size

    def make_frame(t):
        zoom   = 1.0 + 0.06 * (t / max(duration, 0.001))
        nw, nh = int(w * zoom), int(h * zoom)
        zoomed = img_pil.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        return np.array(zoomed.crop((left, top, left + w, top + h)))

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Image helpers ─────────────────────────────────────────────────────

def _fill_canvas(pil_image):
    src    = pil_image.convert("RGB")
    iw, ih = src.size
    scale_fill = max(VIDEO_W / iw, VIDEO_H / ih)
    bg = src.resize((int(iw * scale_fill), int(ih * scale_fill)), Image.LANCZOS)
    bx = (bg.width - VIDEO_W) // 2
    by = (bg.height - VIDEO_H) // 2
    bg = bg.crop((bx, by, bx + VIDEO_W, by + VIDEO_H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=40))
    bg = bg.point(lambda p: int(p * 0.35))

    scale_w   = VIDEO_W / iw
    scale_h   = (VIDEO_H * 0.60) / ih
    scale_fit = max(scale_w, scale_h)
    fw = int(iw * scale_fit)
    fh = int(ih * scale_fit)
    fg = src.resize((fw, fh), Image.LANCZOS)
    if fw > VIDEO_W:
        crop_x = (fw - VIDEO_W) // 2
        fg     = fg.crop((crop_x, 0, crop_x + VIDEO_W, fh))
        fw     = VIDEO_W
    canvas = bg.copy()
    canvas.paste(fg, ((VIDEO_W - fw) // 2, (VIDEO_H - fh) // 2))
    return canvas


def _state_search_terms(text):
    """Return image search terms if a Nigerian state name appears in the text."""
    text_lower = text.lower()
    for state, terms in NIGERIAN_STATES.items():
        if state in text_lower:
            shuffled = terms.copy()
            random.shuffle(shuffled)
            return shuffled
    return []


def _topic_search_terms(text):
    """Return image search terms matching the story topic."""
    text_lower = text.lower()
    for keywords, terms in TOPIC_IMAGE_MAP:
        if any(kw in text_lower for kw in keywords):
            shuffled = terms.copy()
            random.shuffle(shuffled)
            return shuffled
    return []


def _fetch_image(image_url, context_text, used_wiki_urls=None):
    """
    Always returns a usable PIL image — never None, never blank.
    Priority: OG image → Nigerian state map → topic entity → generic Nigeria pool → gradient
    No Wikipedia image is reused within the same video run (tracked via used_wiki_urls).
    """
    if used_wiki_urls is None:
        used_wiki_urls = set()

    # 1 — OG image from the article
    if image_url:
        try:
            r    = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            data = r.content
            img  = Image.open(BytesIO(data))
            img.load()
            if img.width >= 250 and img.height >= 150:
                return img.convert("RGB")
        except Exception:
            pass

    # 2 — Nigerian state-specific image (Delta State, Kano, etc.)
    for term in _state_search_terms(context_text):
        img, url = _wikimedia_image_single(term)
        if img and url not in used_wiki_urls:
            used_wiki_urls.add(url)
            return img

    # 3 — Topic-matched Nigeria entity
    for term in _topic_search_terms(context_text):
        img, url = _wikimedia_image_single(term)
        if img and url not in used_wiki_urls:
            used_wiki_urls.add(url)
            return img

    # 4 — Generic Nigeria pool
    for term in random.sample(GENERIC_NIGERIA_IMAGES, len(GENERIC_NIGERIA_IMAGES)):
        img, url = _wikimedia_image_single(term)
        if img and url not in used_wiki_urls:
            used_wiki_urls.add(url)
            return img

    # 5 — Gradient (should essentially never reach here)
    return _gradient_fallback(context_text)


@lru_cache(maxsize=128)
def _wikimedia_image_single(query):
    """
    Fetch one Wikipedia image for the given query.
    Returns (PIL.Image, source_url) or (None, None).
    Cached so repeated queries for the same term don't hit the network twice.
    """
    SKIP_EXTS = (".svg", ".ogg", ".ogv", ".pdf")
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search",
                "gsrsearch": query, "gsrlimit": 8,
                "prop": "pageimages", "piprop": "thumbnail",
                "pithumbsize": 1200, "format": "json",
            },
            headers={"User-Agent": "YaarnNewsBot/1.0"},
            timeout=10,
        )
        if not r.ok:
            return None, None
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source", "")
            if not thumb:
                continue
            if any(thumb.lower().split("?")[0].endswith(ext) for ext in SKIP_EXTS):
                continue
            try:
                img_resp = requests.get(thumb, timeout=10,
                                         headers={"User-Agent": "YaarnNewsBot/1.0"})
                if not img_resp.ok:
                    continue
                ct = img_resp.headers.get("Content-Type", "")
                if "svg" in ct or "text" in ct or "html" in ct:
                    continue
                img = Image.open(BytesIO(img_resp.content))
                img.load()
                if img.width >= 300 and img.height >= 200:
                    return img.convert("RGB"), thumb
            except Exception:
                continue
    except Exception:
        pass
    return None, None


def _get_generic_nigeria_images(count, used_wiki_urls):
    results = []
    terms   = GENERIC_NIGERIA_IMAGES.copy()
    random.shuffle(terms)
    for term in terms:
        if len(results) >= count:
            break
        img, url = _wikimedia_image_single(term)
        if img and url not in used_wiki_urls:
            used_wiki_urls.add(url)
            results.append(img)
    while len(results) < count:
        results.append(_gradient_fallback("Nigeria"))
    return results


def _gradient_fallback(query):
    digest = int(hashlib.md5(query.encode()).hexdigest()[:6], 16)
    r_val  = 80 + ((digest >> 16) & 0xFF) % 120
    g_val  = 80 + ((digest >> 8)  & 0xFF) % 120
    b_val  = 80 + (digest         & 0xFF) % 120
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H))
    draw = ImageDraw.Draw(img)
    for y in range(VIDEO_H):
        t = y / VIDEO_H
        draw.line([(0, y), (VIDEO_W, y)],
                  fill=(int(r_val*(1-t) + r_val//2*t),
                        int(g_val*(1-t) + g_val//2*t),
                        int(b_val*(1-t) + b_val//2*t)))
    draw.text((90, VIDEO_H // 2 - 30), query[:25].upper(),
              font=_font(52), fill=WHITE)
    return img


# ── Font — guaranteed scalable TTF, no silent fallback to tiny bitmap ─

_FONT_PATH_CACHE = None

def _ensure_font_file():
    global _FONT_PATH_CACHE
    if _FONT_PATH_CACHE:
        return _FONT_PATH_CACHE

    candidates = [
        # Linux (GitHub Actions)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        # Windows (local dev)
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
        # Bundled in repo (place any .ttf here)
        "assets/fonts/Bold.ttf",
        "/tmp/yaarn_font.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"  ✓ Font loaded: {path}")
            _FONT_PATH_CACHE = path
            return path

    # Download as last resort — correct path for Google Fonts
    urls = [
        "https://github.com/google/fonts/raw/main/apache/robotocondensed/static/RobotoCondensed-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/oswald/static/Oswald-Bold.ttf",
    ]
    for url in urls:
        try:
            print(f"  ⚠ No system font found — downloading from {url}...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            Path("/tmp/yaarn_font.ttf").write_bytes(r.content)
            _FONT_PATH_CACHE = "/tmp/yaarn_font.ttf"
            print("  ✓ Font downloaded successfully")
            return _FONT_PATH_CACHE
        except Exception as e:
            print(f"  ✗ Font download failed: {e}")

    print("  ✗ CRITICAL: No font available — text will be tiny!")
    return None


@lru_cache(maxsize=30)
def _font(size):
    path = _ensure_font_file()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception as e:
            print(f"  ✗ Font load error: {e}")
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — DELIVER
# ══════════════════════════════════════════════════════════════════════

def _build_caption(script):
    lines = [f"Yaarn — {script['date']}\n"]
    for i, s in enumerate(script["stories"], 1):
        lines.append(f"{i}. {s['headline']}")
    lines.append("\nFollow @yaarn.ng — Nigerian news, every day")
    return "\n".join(lines)


def _upload_to_github_release(video_path):
    """
    Upload video to a temporary GitHub Release and return a public download URL.
    The release is tagged 'temp-video-TIMESTAMP' and should be cleaned up later.
    Requires GITHUB_TOKEN and GITHUB_REPO env vars.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None, None

    timestamp  = datetime.now().strftime("%Y%m%d%H%M%S")
    tag        = f"temp-video-{timestamp}"
    api_base   = f"https://api.github.com/repos/{GITHUB_REPO}"
    headers    = {"Authorization": f"token {GITHUB_TOKEN}",
                  "Accept": "application/vnd.github+json"}

    # Create release
    try:
        r = requests.post(f"{api_base}/releases", headers=headers,
                          json={"tag_name": tag, "name": tag,
                                "draft": False, "prerelease": True},
                          timeout=30)
        r.raise_for_status()
        release    = r.json()
        release_id = release["id"]
        upload_url = release["upload_url"].split("{")[0]
    except Exception as e:
        print(f"  ✗ GitHub release create failed: {e}")
        return None, None

    # Upload asset
    try:
        file_size = os.path.getsize(video_path)
        with open(video_path, "rb") as f:
            r = requests.post(
                f"{upload_url}?name=video.mp4",
                headers={**headers, "Content-Type": "video/mp4",
                         "Content-Length": str(file_size)},
                data=f,
                timeout=600,
            )
        r.raise_for_status()
        asset     = r.json()
        video_url = asset["browser_download_url"]
        print(f"  ✓ GitHub Release upload: {video_url}")
        return video_url, (api_base, headers, release_id)
    except Exception as e:
        print(f"  ✗ GitHub Release upload failed: {e}")
        # Clean up the empty release
        try:
            requests.delete(f"{api_base}/releases/{release_id}", headers=headers, timeout=15)
        except Exception:
            pass
        return None, None


def _delete_github_release(cleanup_info):
    """Delete the temporary GitHub release after Instagram has processed it."""
    if not cleanup_info:
        return
    api_base, headers, release_id = cleanup_info
    try:
        requests.delete(f"{api_base}/releases/{release_id}",
                        headers=headers, timeout=15)
        print("  ✓ Temporary GitHub release deleted")
    except Exception as e:
        print(f"  ⚠ Could not delete GitHub release: {e}")


def upload_video_with_fallbacks(video_path):
    """
    Try upload hosts in order. Returns (url, cleanup_info).
    cleanup_info is non-None only for GitHub releases (needs deletion after use).
    """
    size_mb = os.path.getsize(video_path) / 1024 / 1024
    print(f"  Uploading video ({size_mb:.1f} MB) — trying hosts in order...")

    # 1) GitHub Releases — preferred on GitHub Actions, zero third-party dependency
    if GITHUB_TOKEN and GITHUB_REPO:
        print("  Trying GitHub Releases...")
        url, cleanup = _upload_to_github_release(video_path)
        if url:
            return url, cleanup

    # 2) catbox.moe — permanent, direct URL
    print("  Trying catbox.moe...")
    for attempt in range(2):
        try:
            with open(video_path, "rb") as f:
                resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": ("video.mp4", f, "video/mp4")},
                    timeout=360,
                )
            if resp.status_code == 200 and resp.text.strip().startswith("https://"):
                url = resp.text.strip()
                print(f"  ✓ catbox.moe: {url}")
                return url, None
            print(f"  ✗ catbox.moe failed (attempt {attempt+1}): {resp.text[:80]}")
        except Exception as e:
            print(f"  ✗ catbox.moe error (attempt {attempt+1}): {e}")
        if attempt == 0:
            time.sleep(5)

    # 3) litterbox.catbox.moe — temporary 72h
    print("  Trying litterbox.catbox.moe...")
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                data={"reqtype": "fileupload", "time": "72h"},
                files={"fileToUpload": ("video.mp4", f, "video/mp4")},
                timeout=360,
            )
        if resp.status_code == 200 and resp.text.strip().startswith("https://"):
            url = resp.text.strip()
            print(f"  ✓ litterbox: {url}")
            return url, None
        print(f"  ✗ litterbox failed: {resp.text[:80]}")
    except Exception as e:
        print(f"  ✗ litterbox error: {e}")

    # 4) tmpfiles.org — direct /dl/ URL
    print("  Trying tmpfiles.org...")
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": ("video.mp4", f, "video/mp4")},
                timeout=360,
            ).json()
        if resp.get("status") == "success":
            url = resp["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            print(f"  ✓ tmpfiles.org: {url}")
            return url, None
        print(f"  ✗ tmpfiles.org failed: {resp}")
    except Exception as e:
        print(f"  ✗ tmpfiles.org error: {e}")

    return None, None


def send_to_instagram(video_path, caption):
    print("  → Instagram (@yaarn.ng)...")
    video_url, cleanup_info = upload_video_with_fallbacks(video_path)
    if not video_url:
        print("  ✗ Skipping Instagram — all upload hosts failed")
        return False

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v25.0/{IG_ACCOUNT_ID}/media",
            data={"media_type": "REELS", "video_url": video_url,
                  "caption": caption, "access_token": FB_PAGE_TOKEN},
            timeout=60,
        ).json()

        if "id" not in resp:
            print(f"  ✗ Container error: {resp}")
            _delete_github_release(cleanup_info)
            return False
        container_id = resp["id"]
        print(f"  ✓ Container created — processing...")

        for attempt in range(30):
            time.sleep(10)
            status = requests.get(
                f"https://graph.facebook.com/v25.0/{container_id}",
                params={"fields": "status_code", "access_token": FB_PAGE_TOKEN},
                timeout=30,
            ).json().get("status_code", "IN_PROGRESS")
            print(f"  [{attempt+1}/30] {status}")
            if status == "FINISHED":
                break
            if status == "ERROR":
                print("  ✗ Instagram processing error")
                _delete_github_release(cleanup_info)
                return False

        pub = requests.post(
            f"https://graph.facebook.com/v25.0/{IG_ACCOUNT_ID}/media_publish",
            data={"creation_id": container_id, "access_token": FB_PAGE_TOKEN},
            timeout=60,
        ).json()

        _delete_github_release(cleanup_info)

        if "id" in pub:
            print(f"  ✓ Instagram posted (ID: {pub['id']})")
            return True
        print(f"  ✗ Publish error: {pub}")
        return False

    except Exception as e:
        _delete_github_release(cleanup_info)
        print(f"  ✗ Instagram exception: {e}")
        return False


def send_to_telegram(video_path, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  → Telegram skipped (secrets not set yet)")
        return
    print("  → Telegram...")
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption,
                      "supports_streaming": True, "parse_mode": "Markdown"},
                files={"video": f},
                timeout=600,
            )
        print("  ✓ Telegram posted" if resp.status_code == 200
              else f"  ✗ Telegram error: {resp.text}")
    except Exception as e:
        print(f"  ✗ Telegram exception: {e}")


def deliver(video_path, script):
    print("\n[6/6] Delivering...")
    caption = _build_caption(script)
    ig_ok   = send_to_instagram(video_path, caption)
    send_to_telegram(video_path, caption)
    print(f"\n  Instagram : {'✅ Posted' if ig_ok else '❌ Failed'}")
    print("  (Facebook: share from Instagram via cross-post toggle)")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir   = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    stories, memory = scrape_news()
    if not stories:
        print("No new Nigerian stories found. Exiting.")
        return

    filtered = filter_stories(stories)
    if not filtered:
        print("No high-impact stories after filtering. Exiting.")
        return

    script = generate_script(filtered)
    (run_dir / "script.json").write_text(json.dumps(script, indent=2))

    audio_files = generate_audio(script, run_dir)
    video_path  = build_video(script, filtered, audio_files, run_dir)

    deliver(video_path, script)
    (run_dir / "caption.txt").write_text(_build_caption(script), encoding="utf-8")

    for story in filtered:
        memory = mark_seen(story["title"], memory)
    memory = cleanup_memory(memory)
    save_memory(memory)

    print(f"\n✅ Done. Output → {run_dir}/")


if __name__ == "__main__":
    run()