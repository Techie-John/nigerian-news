#!/usr/bin/env python3
"""
main.py — Yaarn Daily Nigerian News
Scrapes → Groq filters → Groq scripts → Kokoro TTS → MoviePy builds → Facebook + Instagram + Telegram
"""

import os, json, re, hashlib, textwrap, time
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
    CompositeAudioClip
)


load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
KOKORO_API_URL   = os.getenv("KOKORO_API_URL")
KOKORO_API_KEY   = os.getenv("KOKORO_API_KEY")
FB_PAGE_ID       = os.getenv("FACEBOOK_PAGE_ID")
FB_PAGE_TOKEN    = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
IG_ACCOUNT_ID    = os.getenv("INSTAGRAM_ACCOUNT_ID")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")   # optional — add later
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")     # optional — add later

GROQ_MODEL   = "llama-3.3-70b-versatile"
VIDEO_W      = 1080
VIDEO_H      = 1920
FPS          = 24
MAX_STORIES  = 30
HOURS_BACK   = 18        # same-day news only
SIM_THRESH   = 0.55
OUTPUT_DIR   = Path("output")
MEMORY_FILE  = Path("story_memory.json")  # committed back to repo after each run

# Yaarn brand — black and white
BLACK = (8, 8, 8)
WHITE = (255, 255, 255)
GREY  = (150, 150, 150)

# Nigerian RSS sources
RSS_SOURCES = [
    {"name": "Punch",           "url": "https://punchng.com/feed/",                   "priority": 1},
    {"name": "Channels TV",     "url": "https://www.channelstv.com/feed/",            "priority": 2},
    {"name": "Premium Times",   "url": "https://www.premiumtimesng.com/feed",         "priority": 3},
    {"name": "Vanguard",        "url": "https://www.vanguardngr.com/feed/",           "priority": 4},
    {"name": "The Guardian NG", "url": "https://guardian.ng/feed/",                   "priority": 5},
    {"name": "Sahara Reporters","url": "https://saharareporters.com/rss.xml",         "priority": 6},
    {"name": "Daily Trust",     "url": "https://dailytrust.com/feed",                 "priority": 7},
]

# Broad pre-filter — Groq does the real impact filtering
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


# ══════════════════════════════════════════════════════════════════════
# STORY MEMORY — prevents repeating headlines across days
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
    """Remove entries older than 7 days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    return {
        k: v for k, v in memory.items()
        if datetime.fromisoformat(v) > cutoff
    }


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
    text_lower = text.lower()
    return any(kw in text_lower for kw in NIGERIA_KEYWORDS)


def _deduplicate(stories):
    stories.sort(key=lambda x: (
        x["priority"],
        -(x["published"].timestamp() if x["published"] else 0)
    ))
    seen_titles   = []
    source_counts = {}
    result        = []

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
        if len(result) >= 50:  # keep up to 50 for Groq to filter down to MAX_STORIES
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
        # Try og:image first, then twitter:image as fallback
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
# STAGE 2A — GROQ FILTER (picks most impactful stories)
# ══════════════════════════════════════════════════════════════════════

def filter_stories(stories):
    """Groq pass 1 — pick the most impactful stories for everyday Nigerians."""
    print("\n[2/6] Filtering stories for impact...")

    if not stories:
        return []

    stories_text = "\n".join(
        f"[{i+1}] {s['title']} ({s['source']})"
        for i, s in enumerate(stories)
    )

    prompt = f"""You are a Nigerian news editor at a big TV station.
Your job: pick the {MAX_STORIES} stories that matter most to everyday Nigerians.

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
- Stories only politicians care about

Stories to choose from:
{stories_text}

Return ONLY a JSON array of the story numbers you picked. Example: [1, 3, 5, 7, 8, 11, 14, 16]
Pick exactly {MAX_STORIES} or fewer if not enough qualify.
Return ONLY the JSON array. Nothing else."""

    client = Groq(api_key=GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )

    raw   = resp.choices[0].message.content.strip()
    match = re.search(r'\[[\d,\s]+\]', raw)
    if not match:
        print("  ⚠ Could not parse filter — using first 8 stories")
        return stories[:MAX_STORIES]

    indices  = json.loads(match.group(0))
    selected = [stories[i - 1] for i in indices if 1 <= i <= len(stories)]
    print(f"  ✓ {len(selected)} high-impact stories selected")
    return selected


# ══════════════════════════════════════════════════════════════════════
# STAGE 2B — GROQ SCRIPT (writes engaging narration)
# ══════════════════════════════════════════════════════════════════════

def generate_script(stories):
    """Groq pass 2 — write engaging Nigerian news narration."""
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
        f"Details: {s['summary'][:400]}"
        for i, s in enumerate(stories)
    )

    system_prompt = f"""You are the voice of Yaarn — Nigeria's daily news channel.
You report the facts clearly and without bias.

TODAY'S DATE: {display_date}

YOUR STYLE:
- Use very simple English. Grade 4-5 level. Short sentences. Easy words that everyone understands.
- Report facts only. Do NOT editorialize, express opinions, or use charged language.
- Do NOT use words like "shocking", "outrageous", "unbelievable", "people are angry", "this is serious".
- Stay calm and neutral. Like a professional newsreader, not a commentator.
- NO big grammar words. NO "furthermore". NO "it is imperative". NO "stakeholders".
- Sound direct and clear.

RULES:
1. INTRO: Start exactly with: "This is what happened in Nigeria today, {display_date}."
   Then briefly mention 2-3 stories coming up.
2. Each story narration: EXACTLY 1 to 2 SHORT sentences. No more.
   - One sentence states the core fact. One sentence adds one key detail if needed. That is all.
3. [WHEN: TODAY] → say "this morning", "earlier today"
   [WHEN: YESTERDAY] → say "yesterday", "last night"
4. Short display headline: max 6 words. Like a strong WhatsApp message. No full stop.
5. OUTRO: Brief and factual. Tell people to follow Yaarn for daily Nigerian news. Keep it very short.
6. NO URLs. NO "according to". NO "reportedly". NO "it was gathered". NO passive voice. NO opinions.
7. Return ONLY valid JSON. No markdown. No extra text.

JSON FORMAT:
{{
  "date": "{display_date}",
  "intro": "<intro narration>",
  "stories": [
    {{"headline": "<max 6 words>", "narration": "<2-3 sentences>"}}
  ],
  "outro": "<friendly CTA to follow Yaarn>"
}}"""

    client = Groq(api_key=GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Write the Yaarn script for these {len(stories)} Nigerian stories:\n\n{stories_text}"},
        ],
        temperature=0.75,
        max_tokens=8000,
    )

    print(f"  Groq finish_reason: {resp.choices[0].finish_reason}")
    raw = resp.choices[0].message.content.strip()

    # JSON repair
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

    print(f"  ✓ Script ready — {len(script['stories'])} stories")
    return script


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
# STAGE 3 — AUDIO (Kokoro via Modal — bm_george voice)
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
            json={
                "text":    text,
                "voice":   "bm_george",
                "speed":   0.94,
                "api_key": KOKORO_API_KEY,
            },
            timeout=60,
        )
        response.raise_for_status()
        path = output_dir / f"audio_{name}.wav"
        path.write_bytes(response.content)
        audio_files[name] = str(path)
        print(f"  ✓ {name}")

    return audio_files


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — VIDEO (MoviePy v2 + Pillow)
# ══════════════════════════════════════════════════════════════════════

def build_video(script, stories, audio_files, output_dir):
    print("\n[5/6] Building video...")

    # Prefetch all images in parallel
    images = [None] * len(stories)

    def _fetch_one(args):
        idx, s = args
        return idx, _fetch_image(s["image_url"], s["title"])

    with ThreadPoolExecutor(max_workers=6) as ex:
        for idx, img in ex.map(_fetch_one, enumerate(stories)):
            images[idx] = img
    print(f"  ✓ {sum(1 for im in images if im is not None)} images ready")

    clips = []

    # Intro clip
    intro_audio = AudioFileClip(audio_files["intro"])
    intro_frame = _make_collage_intro(script["date"], images)
    intro_clip  = _static_clip(intro_frame, intro_audio.duration).with_audio(intro_audio)
    clips.append(intro_clip)

    # Story clips
    for i, (story_data, pil_image) in enumerate(zip(script["stories"], images)):
        key        = f"story_{i+1:02d}"
        audio      = AudioFileClip(audio_files[key])
        frame      = _make_story_frame(pil_image, story_data["headline"])
        story_clip = _zoom_clip(frame, audio.duration).with_audio(audio)
        clips.append(story_clip)
        print(f"  ✓ Story {i+1}/{len(stories)}: {story_data['headline'][:55]}")

    # Outro clip
    outro_audio = AudioFileClip(audio_files["outro"])
    outro_frame = _make_outro_frame()
    outro_clip  = _static_clip(outro_frame, outro_audio.duration).with_audio(outro_audio)
    clips.append(outro_clip)

    # FIX: MoviePy v2 — no padding, no crossfades, no temp_audiofile param
    final    = concatenate_videoclips(clips, method="compose")

    # Background music — mix audio.mp3 at low volume if it exists
    bg_music_path = Path("audio.mp3")
    if bg_music_path.exists():
        try:
            bg = AudioFileClip(str(bg_music_path))
            # Loop or trim to match video duration
            total_dur = final.duration
            if bg.duration < total_dur:
                import math
                loops = math.ceil(total_dur / bg.duration)
                from moviepy import concatenate_audioclips
                bg = concatenate_audioclips([bg] * loops)
            bg = bg.subclipped(0, total_dur).with_volume_scaled(0.07)
            mixed = CompositeAudioClip([final.audio, bg])
            final = final.with_audio(mixed)
            print("  ✓ Background music mixed in (7% volume)")
        except Exception as e:
            print(f"  ⚠ Background music skipped: {e}")

    out_path = str(output_dir / "video.mp4")
    final.write_videofile(
        out_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    print(f"  ✓ Video saved → {out_path}")
    return out_path


# ── Frame composers ───────────────────────────────────────────────────

def _make_collage_intro(date_str, images):
    canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), BLACK)

    cols, rows = 3, 4
    cell_w = VIDEO_W // cols
    cell_h = VIDEO_H // rows

    # Guarantee every cell has an image — cycle through whatever we have,
    # and if images list is totally empty, fetch a Nigeria fallback image.
    valid = [im for im in images if im is not None]
    if not valid:
        fallback = _wikimedia_image("Nigeria")
        if fallback is None:
            fallback = _gradient_fallback("Nigeria")
        valid = [fallback]

    total_cells = cols * rows
    for idx in range(total_cells):
        img_src = valid[idx % len(valid)].convert("RGB")
        iw, ih  = img_src.size
        scale   = max(cell_w / iw, cell_h / ih)
        nw, nh  = int(iw * scale), int(ih * scale)
        thumb   = img_src.resize((nw, nh), Image.LANCZOS)
        cx      = (nw - cell_w) // 2
        cy      = (nh - cell_h) // 2
        thumb   = thumb.crop((cx, cy, cx + cell_w, cy + cell_h))
        canvas.paste(thumb, ((idx % cols) * cell_w, (idx // cols) * cell_h))

    # Semi-dark overlay so text pops — keep it moderate (not too heavy)
    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 140))
    canvas  = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw    = ImageDraw.Draw(canvas)

    # Brand — large and bold
    draw.text((90, 520),  "YAARN",       font=_font(680), fill=WHITE)
    draw.rectangle([90, 740, VIDEO_W - 90, 750], fill=WHITE)
    draw.text((90, 776),  date_str.upper(), font=_font(48),  fill=GREY)

    return np.array(canvas)


def _draw_text_with_shadow(draw, xy, text, font, fill=WHITE, shadow_offset=4, shadow_fill=(0,0,0,200)):
    """Draw text with a drop shadow for readability on any background."""
    x, y = xy
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_fill)
    draw.text((x, y), text, font=font, fill=fill)


def _make_story_frame(pil_image, headline):
    img  = _fill_canvas(pil_image)

    # Bottom gradient burn — makes text area always readable
    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    # Gradient from transparent at 50% to near-black at bottom
    for y in range(VIDEO_H // 2, VIDEO_H):
        alpha = int(220 * ((y - VIDEO_H // 2) / (VIDEO_H // 2)))
        ov_draw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
    img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # White accent bar
    bar_y = VIDEO_H - 560
    draw.rectangle([90, bar_y, 200, bar_y + 10], fill=WHITE)

    # Headline — auto-scale, starts at 100px
    f_size = 100
    while f_size >= 60:
        lines   = textwrap.wrap(headline.upper(), width=18)
        total_h = len(lines) * (f_size + 20)
        if bar_y + 40 + total_h < VIDEO_H - 60:
            break
        f_size -= 8

    y = bar_y + 40
    for line in textwrap.wrap(headline.upper(), width=18):
        _draw_text_with_shadow(draw, (90, y), line, font=_font(f_size), shadow_offset=5)
        y += f_size + 20

    # Watermark
    draw.text(
        (VIDEO_W - 36, VIDEO_H - 44), "YAARN",
        font=_font(30), fill=(160, 160, 160), anchor="rs"
    )
    return np.array(img)


def _make_outro_frame():
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H), BLACK)
    draw = ImageDraw.Draw(img)
    draw.text((90, 620),  "FOLLOW",                   font=_font(80),  fill=WHITE)
    draw.text((90, 730),  "YAARN",                    font=_font(180), fill=WHITE)
    draw.rectangle([90, 960, VIDEO_W - 90, 970],       fill=WHITE)
    draw.text((90, 996),  "Nigerian news, every day.", font=_font(52),  fill=GREY)
    draw.text((90, 1078), "@yaarn.ng",                 font=_font(52),  fill=GREY)
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
        left   = (nw - w) // 2
        top    = (nh - h) // 2
        return np.array(zoomed.crop((left, top, left + w, top + h)))

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Image helpers ─────────────────────────────────────────────────────

def _fill_canvas(pil_image):
    src    = pil_image.convert("RGB")
    iw, ih = src.size

    # Blurred background
    scale_fill = max(VIDEO_W / iw, VIDEO_H / ih)
    bg = src.resize((int(iw * scale_fill), int(ih * scale_fill)), Image.LANCZOS)
    bx = (bg.width - VIDEO_W) // 2
    by = (bg.height - VIDEO_H) // 2
    bg = bg.crop((bx, by, bx + VIDEO_W, by + VIDEO_H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=40))
    bg = bg.point(lambda p: int(p * 0.35))

    # Foreground — fill at least 60% of canvas height
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


def _fetch_image(image_url, fallback_query):
    # 1 — OG image from article
    if image_url:
        try:
            r    = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            data = r.content
            img  = Image.open(BytesIO(data))
            img.load()
            return img
        except Exception:
            pass

    # 2 — Wikipedia (tries full query, shorter queries, then "Nigeria" as last resort)
    wiki = _wikimedia_image(fallback_query)
    if wiki:
        return wiki

    # 3 — Colorful gradient — never blank, never black
    return _gradient_fallback(fallback_query)


def _wikimedia_image(query):
    """Try multiple query variations against Wikipedia. Skips SVG/non-raster results."""
    words = query.split()
    queries_to_try = list(dict.fromkeys(filter(None, [
        query,
        " ".join(words[:4]),
        " ".join(words[:2]),
        "Nigeria",
        "Lagos Nigeria",
    ])))

    SKIP_EXTS = (".svg", ".webp", ".ogg", ".ogv", ".pdf")

    for q in queries_to_try:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "generator": "search",
                    "gsrsearch": q, "gsrlimit": 8,
                    "prop": "pageimages", "piprop": "thumbnail",
                    "pithumbsize": 1200, "format": "json",
                },
                headers={"User-Agent": "YaarnNewsBot/1.0"},
                timeout=10,
            )
            if not r.ok:
                continue
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {}).get("source", "")
                if not thumb:
                    continue
                # Skip SVG and other non-raster formats Pillow can't handle
                thumb_lower = thumb.lower().split("?")[0]
                if any(thumb_lower.endswith(ext) for ext in SKIP_EXTS):
                    continue
                # Fetch the image and verify it's actually a raster image
                img_resp = requests.get(thumb, timeout=10,
                                        headers={"User-Agent": "YaarnNewsBot/1.0"})
                if not img_resp.ok:
                    continue
                ct = img_resp.headers.get("Content-Type", "")
                if "svg" in ct or "text" in ct or "html" in ct:
                    continue
                try:
                    img = Image.open(BytesIO(img_resp.content))
                    img.load()
                    return img.convert("RGB")
                except Exception:
                    continue
        except Exception:
            continue
    return None


def _gradient_fallback(query):
    digest = int(hashlib.md5(query.encode()).hexdigest()[:6], 16)
    # Use bright ranges (80-200) so it NEVER looks black or blank
    r_val = 80 + ((digest >> 16) & 0xFF) % 120
    g_val = 80 + ((digest >> 8)  & 0xFF) % 120
    b_val = 80 + (digest         & 0xFF) % 120
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H))
    draw = ImageDraw.Draw(img)
    for y in range(VIDEO_H):
        t = y / VIDEO_H
        draw.line([(0, y), (VIDEO_W, y)],
                  fill=(int(r_val*(1-t) + r_val//2*t),
                        int(g_val*(1-t) + g_val//2*t),
                        int(b_val*(1-t) + b_val//2*t)))
    # Label so it never looks empty
    label = query[:25].upper()
    draw.text((90, VIDEO_H // 2 - 30), label, font=_font(52), fill=WHITE)
    return img


@lru_cache(maxsize=20)
def _font(size):
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "assets/fonts/Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — DELIVER (Facebook + Instagram + Telegram)
# ══════════════════════════════════════════════════════════════════════

def _build_caption(script):
    lines = [f"🗞 *Yaarn — {script['date']}*\n"]
    for i, s in enumerate(script["stories"], 1):
        lines.append(f"{i}. {s['headline']}")
    lines.append("\n📲 Follow @yaarn.ng — Nigerian news, every day")
    return "\n".join(lines)


def _get_public_video_url(video_path):
    """Upload video to catbox.moe — reliable free public host for Instagram."""
    print("  Uploading to catbox.moe...")
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": ("video.mp4", f, "video/mp4")},
                timeout=300,
            )
        if resp.status_code == 200 and resp.text.strip().startswith("https://"):
            url = resp.text.strip()
            print(f"  ✓ Public URL: {url}")
            return url
        print(f"  ✗ catbox.moe failed: {resp.text[:120]}")
    except Exception as e:
        print(f"  ✗ catbox.moe error: {e}")
    return None


def send_to_facebook(video_path, caption):
    print("  → Facebook...")
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"https://graph.facebook.com/v25.0/{FB_PAGE_ID}/videos",
                data={
                    "description":  caption,
                    "access_token": FB_PAGE_TOKEN,
                    "published":    "true",
                },
                files={"source": f},
                timeout=600,
            )
        data = resp.json()
        if "id" in data:
            print(f"  ✓ Facebook posted (ID: {data['id']})")
            return True
        print(f"  ✗ Facebook error: {data}")
        return False
    except Exception as e:
        print(f"  ✗ Facebook exception: {e}")
        return False


def send_to_instagram_url(video_url, caption):
    print("  → Instagram (@yaarn.ng)...")
    if not video_url:
        print("  ✗ Skipping Instagram — no public URL")
        return False
    try:
        # Step 1: Create media container
        resp = requests.post(
            f"https://graph.facebook.com/v25.0/{IG_ACCOUNT_ID}/media",
            data={
                "media_type":   "REELS",
                "video_url":    video_url,
                "caption":      caption,
                "access_token": FB_PAGE_TOKEN,
            },
            timeout=60,
        )
        data = resp.json()
        if "id" not in data:
            print(f"  ✗ Container error: {data}")
            return False
        container_id = data["id"]
        print(f"  ✓ Container created — waiting for Instagram to process...")

        # Step 2: Poll until ready (max 5 minutes)
        for attempt in range(30):
            time.sleep(10)
            status = requests.get(
                f"https://graph.facebook.com/v25.0/{container_id}",
                params={"fields": "status_code", "access_token": FB_PAGE_TOKEN},
                timeout=30,
            ).json().get("status_code", "IN_PROGRESS")
            print(f"  Status: {status} ({attempt + 1}/30)")
            if status == "FINISHED":
                break
            if status == "ERROR":
                print("  ✗ Instagram processing error")
                return False

        # Step 3: Publish
        pub = requests.post(
            f"https://graph.facebook.com/v25.0/{IG_ACCOUNT_ID}/media_publish",
            data={"creation_id": container_id, "access_token": FB_PAGE_TOKEN},
            timeout=60,
        ).json()
        if "id" in pub:
            print(f"  ✓ Instagram posted (ID: {pub['id']})")
            return True
        print(f"  ✗ Publish error: {pub}")
        return False

    except Exception as e:
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
                data={
                    "chat_id":            TELEGRAM_CHAT_ID,
                    "caption":            caption,
                    "supports_streaming": True,
                    "parse_mode":         "Markdown",
                },
                files={"video": f},
                timeout=600,
            )
        if resp.status_code == 200:
            print("  ✓ Telegram posted")
        else:
            print(f"  ✗ Telegram error: {resp.text}")
    except Exception as e:
        print(f"  ✗ Telegram exception: {e}")


def deliver(video_path, script):
    print("\n[6/6] Delivering...")

    # Verify token before attempting any posts
    try:
        verify = requests.get(
            f"https://graph.facebook.com/v25.0/{FB_PAGE_ID}",
            params={"fields": "name", "access_token": FB_PAGE_TOKEN},
            timeout=15,
        ).json()
        if "error" in verify:
            print(f"  ✗ Token invalid: {verify['error']['message']}")
            print("  ⚠ Skipping FB + IG — fix token and re-run deliver manually")
            send_to_telegram(video_path, _build_caption(script))
            return
        print(f"  ✓ Token valid — page: {verify.get('name')}")
    except Exception as e:
        print(f"  ✗ Token check failed: {e}")

    caption = _build_caption(script)

    # Upload to catbox once — reuse URL for both FB and IG
    video_url = _get_public_video_url(video_path)

    # Post FB (direct file upload) and IG (URL-based) — FB doesn't need the public URL
    fb_ok = send_to_facebook(video_path, caption)
    ig_ok = send_to_instagram_url(video_url, caption) if video_url else False
    send_to_telegram(video_path, caption)

    print(f"\n  Facebook  : {'✅ Posted' if fb_ok else '❌ Failed'}")
    print(f"  Instagram : {'✅ Posted' if ig_ok else '❌ Failed'}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir   = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Scrape
    stories, memory = scrape_news()
    if not stories:
        print("No new Nigerian stories found. Exiting.")
        return

    # Stage 2a: Filter for impact
    filtered = filter_stories(stories)
    if not filtered:
        print("No high-impact stories after filtering. Exiting.")
        return

    # Stage 2b: Write script
    script = generate_script(filtered)
    (run_dir / "script.json").write_text(json.dumps(script, indent=2))

    # Stage 3: Audio
    audio_files = generate_audio(script, run_dir)

    # Stage 4: Video
    video_path = build_video(script, filtered, audio_files, run_dir)

    # Stage 5: Deliver
    deliver(video_path, script)
    (run_dir / "caption.txt").write_text(_build_caption(script), encoding="utf-8")

    # Update story memory (GitHub Actions workflow commits this back to repo)
    for story in filtered:
        memory = mark_seen(story["title"], memory)
    memory = cleanup_memory(memory)
    save_memory(memory)

    print(f"\n✅ Done. Output → {run_dir}/")


if __name__ == "__main__":
    run()