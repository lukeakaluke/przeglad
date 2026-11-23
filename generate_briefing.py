import html
import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import feedparser
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

# Zdefiniowane źródła RSS
FEEDS = [
    {"name": "Wyborcza (Kraj)", "url": "https://rss.gazeta.pl/pub/rss/gazetawyborcza_kraj.xml", "limit": 15},
    {"name": "Wyborcza (Świat)", "url": "https://rss.gazeta.pl/pub/rss/gazetawyborcza_swiat.xml", "limit": 15},
    {"name": "Rzeczpospolita", "url": "https://www.rp.pl/rss/1056", "limit": 20},
    {"name": "Puls Biznesu", "url": "https://www.pb.pl/rss/najnowsze.xml", "limit": 15},
    {"name": "DGP", "url": "https://rss.gazetaprawna.pl/GazetaPrawna", "limit": 15},
    {"name": "Onet", "url": "https://wiadomosci.onet.pl/.feed", "limit": 20},
    {"name": "WP", "url": "https://wiadomosci.wp.pl/rss.xml", "limit": 20},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MorningBriefingBot/1.0)"
}


class SourceLink(BaseModel):
    name: str
    url: str


class NewsItem(BaseModel):
    headline: str
    summary: str
    sources: list[SourceLink]


class DailyBriefing(BaseModel):
    one_liner: str
    polska: list[NewsItem]
    swiat: list[NewsItem]
    gospodarka_i_prawo: list[NewsItem]


def clean_html(raw_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_recent_articles(hours: int = 24) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles = []

    for feed_cfg in FEEDS:
        try:
            resp = requests.get(feed_cfg["url"], headers=HEADERS, timeout=15)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)

            count = 0
            for entry in parsed.entries:
                if count >= feed_cfg["limit"]:
                    break

                # Filtrowanie po dacie publikacji (jeśli dostępna)
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue

                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", ""))[:350]
                link = entry.get("link", "").strip()

                if title and link:
                    articles.append({
                        "source": feed_cfg["name"],
                        "title": title,
                        "summary": summary,
                        "url": link,
                    })
                    count += 1
        except Exception as e:
            print(f"[WARN] Błąd pobierania {feed_cfg['name']}: {e}")

    return articles


def generate_summary(articles: list[dict]) -> DailyBriefing:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    articles_text = "\n".join(
        f"- [{a['source']}] {a['title']} | {a['summary']} | URL: {a['url']}"
        for a in articles
    )

    prompt = f"""Jesteś doświadczonym redaktorem naczelnym przygotowującym zwięzły, merytoryczny poranny przegląd prasy.

Poniżej znajduje się lista surowych nagłówków z ostatnich 24 godzin z polskich portali i dzienników.

TWOJE ZADANIE I ZASADY SELEKCJI:
1. BEZLITOŚNIE ODFILTRUJ SZUM (szczególnie z portali WP i Onet): pomiń wypadki drogowe, lokalną kryminalistykę, alerty pogodowe w stylu clickbait, plotki, sport, lifestyle, drobne przepychanki słowne w mediach społecznościowych i teksty typu „Ekspert ujawnia”, „Zaskakujące słowa”.
2. WYBIERZ TYLKO NAJWAŻNIEJSZE WYDARZENIA DNIA: kluczowe decyzje rządu/parlamentu/prezydenta, geopolityka, bezpieczeństwo, istotne zmiany prawne, makroekonomia i duże ruchy biznesowe.
3. USUŃ DUPLIKATY: jeśli 3 redakcje piszą o tym samym wydarzeniu, połącz je w JEDEN punkt i podepnij do 3 najlepszych linków źródłowych (używaj dokładnie oryginalnych adresów URL z listy poniżej — nie zmyślaj linków).
4. STYL: concrete, neutralny, bez emocji i bez clickbaitowych tytułów. W `summary` podaj 2–3 zdania konkretu (co się stało i jakie ma znaczenie).
5. STRUKTURA:
   - `one_liner`: 1 krótkie zdanie podsumowujące główny temat poranka.
   - `polska`: 5–6 najważniejszych tematów z kraju.
   - `swiat`: 5–6 najważniejszych tematów ze świata.
   - `gospodarka_i_prawo`: 3–4 kluczowe tematy gospodarcze, rynkowe lub prawne (szczególnie z Pulsu Biznesu, DGP, Rzeczpospolitej).

SUROWE DANE:
{articles_text}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DailyBriefing,
            temperature=0.2,
        ),
    )

    return DailyBriefing.model_validate_json(response.text)


def render_section(title: str, items: list[NewsItem]) -> str:
    cards = []
    for item in items:
        links_html = "".join(
            f'<a class="badge" href="{html.escape(s.url)}" target="_blank" rel="noopener noreferrer">{html.escape(s.name)} ↗</a>'
            for s in item.sources
        )
        cards.append(f"""
        <article class="card">
            <h3>{html.escape(item.headline)}</h3>
            <p>{html.escape(item.summary)}</p>
            <div class="sources">{links_html}</div>
        </article>
        """)
    return f"""
    <section>
        <h2>{html.escape(title)}</h2>
        {''.join(cards)}
    </section>
    """


def build_html(briefing: DailyBriefing, total_scanned: int) -> str:
    now_pl = datetime.now(ZoneInfo("Europe/Warsaw"))
    months_pl = [
        "", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
        "lipca", "sierpnia", "września", "października", "listopada", "grudnia"
    ]
    date_str = f"{now_pl.day} {months_pl[now_pl.month]} {now_pl.year}"
    time_str = now_pl.strftime("%H:%M")

    sections_html = (
        render_section("Polska", briefing.polska)
        + render_section("Świat", briefing.swiat)
        + render_section("Gospodarka i Prawo", briefing.gospodarka_i_prawo)
    )

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="theme-color" content="#111827">
    <link rel="manifest" href="manifest.json">
    <title>Poranny Przegląd — {date_str}</title>
    <style>
        :root {{
            --bg: #f8f9fa;
            --card: #ffffff;
            --text: #111827;
            --muted: #4b5563;
            --border: #e5e7eb;
            --accent: #1d4ed8;
            --badge-bg: #eff6ff;
            --badge-text: #1e40af;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0f172a;
                --card: #1e293b;
                --text: #f1f5f9;
                --muted: #94a3b8;
                --border: #334155;
                --accent: #60a5fa;
                --badge-bg: #1e3a8a;
                --badge-text: #dbeafe;
            }}
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.55;
            padding: 20px 16px 48px;
            max-width: 680px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 2px solid var(--text);
            padding-bottom: 14px;
            margin-bottom: 24px;
        }}
        .meta {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--muted);
            margin-bottom: 4px;
        }}
        h1 {{
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
        }}
        .lead {{
            font-size: 0.95rem;
            color: var(--muted);
            font-style: italic;
        }}
        section {{ margin-bottom: 28px; }}
        h2 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent);
            margin-bottom: 12px;
            font-weight: 700;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .card h3 {{
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 8px;
        }}
        .card p {{
            font-size: 0.94rem;
            color: var(--muted);
            margin-bottom: 12px;
        }}
        .sources {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            text-decoration: none;
            background: var(--badge-bg);
            color: var(--badge-text);
            padding: 4px 10px;
            border-radius: 999px;
        }}
        footer {{
            text-align: center;
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <header>
        <div class="meta">{date_str} • Aktualizacja {time_str}</div>
        <h1>Poranny Przegląd Prasy</h1>
        <p class="lead">{html.escape(briefing.one_liner)}</p>
    </header>
    <main>
        {sections_html}
    </main>
    <footer>
        Przeanalizowano {total_scanned} artykułów z 6 zaufanych źródeł (Wyborcza, Rzeczpospolita, PB, DGP, Onet, WP).
    </footer>
</body>
</html>
"""


def main():
    articles = fetch_recent_articles(hours=24)
    print(f"Pobrano {len(articles)} artykułów z ostatnich 24h.")
    if not articles:
        raise RuntimeError("Nie udało się pobrać żadnych artykułów z kanałów RSS.")

    briefing = generate_summary(articles)
    page_html = build_html(briefing, total_scanned=len(articles))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(page_html)
    print("Wygenerowano index.html.")


if __name__ == "__main__":
    main()
