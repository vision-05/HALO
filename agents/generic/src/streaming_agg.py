from discovery.src.base_agent import BaseAgent
import re
from ddgs import DDGS
from urllib.parse import unquote
import asyncio

class StreamingAggregator(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name, "Aggregator")
        # Each service has a list of query templates tried in order.
        # {show_name} is substituted at search time.
        # The first query that returns a title-verified match wins;
        # if none verify, the first unverified ID from any pass is used.
        self.urls = {
            "netflix":  ["site:netflix.com/title {show_name}"],
            "luna":     ["site:luna.amazon.com/game {show_name}"],
            "disney+":  [
                # Pass 1: broad main domain — most reliable, catches entity UUIDs and slug IDs
                "site:disneyplus.com {show_name} watch",
                # Pass 2: apps subdomain — catches shows DDG misses on main domain
                "site:apps.disneyplus.com {show_name}",
                # Pass 3: narrow series/movies — last resort, DDG/Yahoo often errors on OR operator
                'site:disneyplus.com/series OR site:disneyplus.com/movies "{show_name}"',
            ],
            "spotify":  ["site:open.spotify.com/track {show_name}"],
            "youtube":  ["site:youtube.com/watch {show_name}"],
        }
        self.regexes = {
            "netflix":  r'netflix\.com/title/(\d+)',
            "luna":     r'luna\.amazon\.com/game/.*?/([a-zA-Z0-9]{8,12})(?:[/?#]|$)',
            # FIX: explicit optional locale segment (/en-gb/ etc.) instead of .*?
            # so the non-greedy wildcard can't accidentally consume the ID itself
            "disney+":  [
                r'(?:apps\.)?disneyplus\.com/[a-z-]{0,8}/?(?:series|movies|video)/[^/?#]+/([a-zA-Z0-9]{6,})',
                r'apps\.disneyplus\.com/[a-z]{2}/shows/[^/]+/(\d{7,})',
                r'disneyplus\.com(?:/[a-z]{2}-[a-z]{2})?/browse/entity-([a-f0-9]{8}-[a-f0-9-]{27})',
            ],
            "spotify":  r'open\.spotify\.com/track/([a-zA-Z0-9]{22})(?:[/?#]|$)',
            # YouTube video IDs are always exactly 11 characters
            "youtube":  r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})'
        }

        self.handlers = {
            "get_id_from_title_and_service": self.get_id_from_title,
            "explore_shows_by_search_term":  self.get_titles_from_search
        }
        self.desc = "Can fetch Netflix, Disney+, Luna, Spotify and YouTube IDs from a given title."

    STOPWORDS = {"watch", "full", "episodes", "episode", "season", "disney+",
                 "disney", "on", "only", "the", "a", "and", "of", "at", "in", "hotstar"}

    def _match_score(self, search_words: list, title: str) -> float:
        import re as _re
        title_words = [w for w in _re.sub(r'[|,:\-]', ' ', title).split()
                       if w not in self.STOPWORDS]
        if not title_words:
            return 0.0
        matching = sum(1 for w in title_words if w in search_words)
        return matching / len(title_words)

    def _run_query(self, query: str, site: str, show_name: str):
        """
        Run one DDG query. Returns a verified ID string, "UNVERIFIED:<id>", or None.
        """
        print(f"[StreamingAggregator] Searching: {query}")
        try:
            results = list(DDGS().text(query, max_results=8))
        except Exception as e:
            print(f"[StreamingAggregator] Search error: {e}")
            return None

        fallback_id = None
        best_id     = None
        best_score  = -1.0
        search_words = show_name.split()

        for result in results:
            url   = unquote(result.get("href", ""))
            title = result.get("title", "").lower()
            patterns = self.regexes[site] if isinstance(self.regexes[site], list) else [self.regexes[site]]
            show_id = None
            for pattern in patterns:
                m = re.search(pattern, url)
                if m:
                    show_id = m.group(1)
                    break
            if not show_id:
                continue
            if not fallback_id:
                fallback_id = show_id
            if all(word in title for word in search_words):
                score = self._match_score(search_words, title)
                if score > best_score:
                    best_score = score
                    best_id    = show_id

        if best_id:
            print(f"[StreamingAggregator] Best match: {best_id} (score {best_score:.2f}) for '{show_name}'")
            return best_id
        return f"UNVERIFIED:{fallback_id}" if fallback_id else None

    def get_id_from_title(self, msg: dict):
        show_name = msg["params"]["title"].lower()
        site      = msg["params"]["service"].lower()

        unverified = None
        for query_template in self.urls[site]:
            query  = query_template.format(show_name=show_name)
            result = self._run_query(query, site, show_name)

            if result is None:
                continue
            if not result.startswith("UNVERIFIED:"):
                return result  # verified — done
            if unverified is None:
                unverified = result[len("UNVERIFIED:"):]

        if unverified:
            print(f"[StreamingAggregator] No verified match; best guess: {unverified}")
            return unverified

        print(f"[StreamingAggregator] Could not find ID for '{show_name}' on {site}.")
        return None

    def get_titles_from_search(self, search_string: str) -> list:
        pass

    def add_site(self, site: str, url: str, regex: str):
        self.urls[site] = url
        self.regexes[site] = regex

async def main() -> None:
    stream = StreamingAggregator("StreamingAggregator")
    await stream.run()

asyncio.run(main())