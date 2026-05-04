"""
test_disney_search.py — Test Disney+ ID lookup in isolation, no TV needed.

Run:
    pip install ddgs
    python test_disney_search.py
"""

import re
from ddgs import DDGS
from urllib.parse import unquote

# ── The fixed regex ───────────────────────────────────────────────────────────
# Disney+ URLs look like:
#   https://www.disneyplus.com/en-gb/series/loki/6pARMvILBGzK
#   https://www.disneyplus.com/movies/encanto/3VBGdRAFGXLf
#   https://www.disneyplus.com/series/the-mandalorian/3jLIGMvNPkAf
#
# The old regex missed these because:
#   1. DuckDuckGo sometimes returns URL-encoded hrefs — without unquote() the
#      regex never matches because %2F != /
#   2. The search query "site:disneyplus.com" returns browse/home pages that
#      don't contain /series/ or /movies/ at all, wasting all 5 results.
#      Narrowing to "site:disneyplus.com/series OR site:disneyplus.com/movies"
#      puts the actual title pages first.
#   3. The .*? before (?:series|movies) is non-greedy, so a locale prefix
#      like /en-gb/ is fine, but the segment (?:[^/]+/)* after the keyword
#      was occasionally eating the ID itself on multi-segment paths.

# Disney+ uses several URL shapes depending on region and content type.
# We try each regex in order and take the first match — this means more
# specific patterns (slug/ID) are preferred over looser ones (entity UUID).
DISNEY_REGEXES = [
    # Shape 1 — classic slug/ID:  /series|movies|video/<slug>/<id>
    #   www.disneyplus.com[/en-gb]/series/loki/6pARMvILBGzF
    #   apps.disneyplus.com/bh/movies/encanto/1260076051
    r'(?:apps\.)?disneyplus\.com/[a-z-]{0,8}/?(?:series|movies|video)/[^/?#]+/([a-zA-Z0-9]{6,})',

    # Shape 2 — apps subdomain /shows/ path (episode URLs — grab the show-level ID)
    #   apps.disneyplus.com/bh/shows/loki/1260063451/episode-slug/...
    r'apps\.disneyplus\.com/[a-z]{2}/shows/[^/]+/(\d{7,})',

    # Shape 3 — new entity UUID format:  /browse/entity-<uuid>
    #   www.disneyplus.com[/en-gb]/browse/entity-8f8c5cbb-e5ba-4285-9e2c-86abcac9fd50
    r'disneyplus\.com(?:/[a-z]{2}-[a-z]{2})?/browse/entity-([a-f0-9]{8}-[a-f0-9-]{27})',
]

QUERIES = [
    # Pass 1: broad — whole domain, unquoted (Highly reliable, catches entity + apps URLs)
    'site:disneyplus.com {show_name} watch',
    
    # Pass 2: target apps subdomain directly (Catches shows DDG misses on main domain)
    'site:apps.disneyplus.com {show_name}',
    
    # Pass 3: narrow — only series/movies pages (Moved to last resort because DDG/Yahoo often chokes on the OR operator)
    'site:disneyplus.com/series OR site:disneyplus.com/movies "{show_name}"',
]


# Words stripped before scoring so "watch loki | full episodes | disney+" -> "loki"
STOPWORDS = {"watch", "full", "episodes", "episode", "season", "disney+", "disney",
             "on", "only", "the", "a", "and", "of", "at", "in", "is", "it"}


def _match_score(search_words: list, title: str) -> float:
    """
    Returns a 0-1 score: 1.0 = perfect match, lower = more irrelevant extra words.
    Strips noise words from the title before scoring so "watch loki | disney+"
    scores the same as "loki", but "moana 2" scores lower than "moana" for query "moana".
    """
    title_words = [w for w in re.sub(r'[|,:\\-]', ' ', title).split()
                   if w not in STOPWORDS]
    if not title_words:
        return 0.0
    matching = sum(1 for w in title_words if w in search_words)
    return matching / len(title_words)


def _search_pass(query: str, show_name: str) -> str | None:
    """
    Run one DDG query. Collects ALL verified candidates, scores each by how
    closely the page title matches the query, and returns the best scorer.
    Falls back to the first unverified ID if nothing verifies.
    """
    print(f"\n[Test] Query: {query}")
    try:
        results = list(DDGS().text(query, max_results=8))
    except Exception as e:
        print(f"[Test] Search failed: {e}")
        return None

    fallback_id   = None
    search_words  = [w for w in show_name.lower().split() if w]
    best_id       = None
    best_score    = -1.0

    for result in results:
        url   = unquote(result.get("href", ""))
        title = result.get("title", "").lower()

        print(f"  url  : {url}")
        print(f"  title: {title}")

        show_id = None
        for pattern in DISNEY_REGEXES:
            m = re.search(pattern, url)
            if m:
                show_id = m.group(1)
                break

        if show_id:
            if not fallback_id:
                fallback_id = show_id

            # Only consider results where every search word appears in the title
            if all(word in title for word in search_words):
                score = _match_score(search_words, title)
                print(f"  → regex matched ID: {show_id}  (score {score:.2f})")
                if score > best_score:
                    best_score = score
                    best_id    = show_id
            else:
                print(f"  → regex matched ID: {show_id}  (title mismatch, skipped)")
        else:
            print(f"  → no regex match")

    if best_id:
        print(f"  ✓ Best verified match: {best_id}  (score {best_score:.2f})")
        return best_id

    return f"UNVERIFIED:{fallback_id}" if fallback_id else None


def find_disney_id(show_name: str) -> str | None:
    unverified = None
    for query_template in QUERIES:
        query  = query_template.format(show_name=show_name)
        result = _search_pass(query, show_name)

        if result is None:
            continue  # no IDs at all on this pass, try next query

        if not result.startswith("UNVERIFIED:"):
            return result  # verified match — done

        # Keep the unverified fallback but try the next query first
        unverified = result[len("UNVERIFIED:"):]

    # All passes exhausted without a verified match
    if unverified:
        print(f"\n[Test] No verified match after all passes; best guess: {unverified}")
        return unverified

    print("\n[Test] Could not find a Disney+ ID.")
    return None


# ── Run a few test cases ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_shows = [
        "Loki",
        "The Mandalorian",
        "Encanto",
        "Moana",
    ]

    for show in test_shows:
        result = find_disney_id(show)
        print(f"\n{'='*50}")
        print(f"RESULT  '{show}'  →  {result}")
        print(f"{'='*50}\n")