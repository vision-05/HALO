from discovery.src.base_agent import BaseAgent
import subprocess
import asyncio

class TclTv(BaseAgent):
    def __init__(self) -> None:
        super().__init__("TV", "Actuator")

        self.tv_ip = "192.168.1.161"
        self.local_state = {}
        self.handlers = {
            "power_on":                self.turn_onoff,
            "power_off":               self.turn_onoff,
            "netflix":                 self.start_netflix,
            "netflix_play_show_by_id": self.netflix_play_show,
            "disney+":                    self.start_disney,
            "disney+_select_profile":     self.disney_select_profile,
            "disney+_play_show_by_id":    self.disney_play_show,
            "luna_play_game_by_id":    self.play_luna_game,
            "pause":                   self.play_pause,
            "resume":                  self.play_pause,
            "volume_by_percent_level": self.volume_control,
            "home":                    self.home,
            "spotify_play_track_by_id":self.play_spotify_track,
            "spotify_next_track":      self.spotify_next,
            "spotify_prev_track":      self.spotify_prev,
            "youtube":                 self.start_youtube,
            "youtube_play_video_by_id":self.youtube_play_video,
        }
        self.desc = (
            "TV controlling agent. "
            "IMPORTANT: For any play command (netflix, disney+, spotify, youtube, luna) "
            "you MUST call StreamingAggregator.get_id_from_title_and_service FIRST to get the ID, "
            "then pass that ID to the TV play handler. Never pass a title name directly to a play handler."
        )

        subprocess.run(["adb", "connect", self.tv_ip], capture_output=True)

    # ── Helpers ────────────────────────────────────────────────────────────── #

    def _adb(self, *args):
        """Run an adb shell command against the TV."""
        subprocess.run(["adb", "-s", self.tv_ip, "shell"] + list(args))

    async def _bypass_profile_screen(self, wait: float = 5.0):
        """Press Enter to select whichever profile is highlighted (default behaviour)."""
        await asyncio.sleep(wait)
        self._adb("input", "keyevent", "66")  # KEYCODE_ENTER

    async def _tap_profile_by_name(self, profile_name: str) -> bool:
        """
        Dump the on-screen UI with uiautomator, find the node whose text matches
        profile_name (case-insensitive), and tap its centre point.
        Returns True if the profile was found and tapped, False otherwise.
        """
        import xml.etree.ElementTree as ET
        import re

        self._adb("uiautomator", "dump", "/sdcard/ui_dump.xml")
        await asyncio.sleep(0.5)

        result = subprocess.run(
            ["adb", "-s", self.tv_ip, "pull", "/sdcard/ui_dump.xml", "/tmp/ui_dump.xml"],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"[{self.name}] Could not pull UI dump.")
            return False

        try:
            tree = ET.parse("/tmp/ui_dump.xml")
        except ET.ParseError as e:
            print(f"[{self.name}] UI dump parse error: {e}")
            return False

        for node in tree.iter():
            text = node.attrib.get("text", "")
            if text.strip().lower() == profile_name.strip().lower():
                bounds = node.attrib.get("bounds", "")
                coords = re.findall(r'\d+', bounds)
                if len(coords) == 4:
                    x = (int(coords[0]) + int(coords[2])) // 2
                    y = (int(coords[1]) + int(coords[3])) // 2
                    print(f"[{self.name}] Found profile '{text}' at ({x},{y}), tapping.")
                    self._adb("input", "tap", str(x), str(y))
                    return True

        print(f"[{self.name}] Profile '{profile_name}' not found on screen.")
        return False

    # ── Basic controls ─────────────────────────────────────────────────────── #

    async def turn_onoff(self, msg: dict) -> None:
        self._adb("input", "keyevent", "26")

    async def volume_control(self, msg: dict) -> None:
        level = msg["params"].get("level", 10)
        self._adb("media", "volume", "--stream", "3", "--set", str(level))

    async def home(self, msg: dict) -> None:
        self._adb("input", "keyevent", "3")

    async def play_pause(self, msg: dict) -> None:
        self._adb("input", "keyevent", "85")

    # ── App launchers ──────────────────────────────────────────────────────── #

    async def start_netflix(self, msg: dict) -> None:
        self._adb("am", "start", "-n", "com.netflix.ninja/.MainActivity")

    async def start_disney(self, msg: dict) -> None:
        self._adb("monkey", "-p", "com.disney.disneyplus", "-c", "android.intent.category.LAUNCHER", "1")

    # ── Media playback ─────────────────────────────────────────────────────── #

    async def disney_select_profile(self, msg: dict) -> None:
        """
        Open Disney+ and select a named profile, then wait for the home screen
        to fully load so a follow-up disney+_play_show_by_id can fire immediately.

        Two-command pattern:
          1. "Open Disney+ on the kids profile"  →  this method
          2. "Play Loki"                          →  disney_play_show (detects foreground, skips relaunch)
        """
        profile_name = msg["params"].get("profile_name", "")
        if not profile_name:
            print(f"[{self.name}] disney_select_profile: no profile_name provided.")
            return

        await self.start_disney(msg)
        await asyncio.sleep(10.0)  # wait for profile picker screen to appear

        found = await self._tap_profile_by_name(profile_name)
        if not found:
            print(f"[{self.name}] Falling back to default profile selection.")
            self._adb("input", "keyevent", "66")

        # Wait for the home screen to finish loading before returning.
        # This is what makes the two-command pattern reliable — by the time
        # "Play Loki" arrives, Disney+ is fully ready to receive the deep link.
        print(f"[{self.name}] Profile selected, waiting for home screen to load...")
        await asyncio.sleep(6.0)
        print(f"[{self.name}] Disney+ ready.")

    def _search_disney_id(self, show_name: str) -> str | None:
        """
        Fallback: search Disney+ for a show ID directly on the TV agent.
        Used when the LLM calls disney+_play_show_by_id with show_name instead of show_id.
        Mirrors the logic in StreamingAggregator so we don't need a round-trip over the mesh.
        """
        import re as _re
        from ddgs import DDGS
        from urllib.parse import unquote as _unquote

        regexes = [
            r'(?:apps\.)?disneyplus\.com/[a-z-]{0,8}/?(?:series|movies|video)/[^/?#]+/([a-zA-Z0-9]{6,})',
            r'apps\.disneyplus\.com/[a-z]{2}/shows/[^/]+/(\d{7,})',
            r'disneyplus\.com(?:/[a-z]{2}-[a-z]{2})?/browse/entity-([a-f0-9]{8}-[a-f0-9-]{27})',
        ]
        queries = [
            f"site:disneyplus.com {show_name} watch",
            f"site:apps.disneyplus.com {show_name}",
        ]
        stopwords = {"watch","full","episodes","episode","season","disney+",
                     "disney","on","only","the","a","and","of","at","in","hotstar"}

        def score(words, title):
            tw = [w for w in _re.sub(r'[|,:\-]', ' ', title).split() if w not in stopwords]
            return sum(1 for w in tw if w in words) / len(tw) if tw else 0.0

        search_words = show_name.lower().split()
        unverified   = None

        for query in queries:
            try:
                results = list(DDGS().text(query, max_results=8))
            except Exception as e:
                print(f"[{self.name}] Search error: {e}")
                continue

            best_id, best_score = None, -1.0
            for result in results:
                url   = _unquote(result.get("href", ""))
                title = result.get("title", "").lower()
                show_id = None
                for pat in regexes:
                    m = _re.search(pat, url)
                    if m:
                        show_id = m.group(1)
                        break
                if not show_id:
                    continue
                if not unverified:
                    unverified = show_id
                if all(w in title for w in search_words):
                    s = score(search_words, title)
                    if s > best_score:
                        best_score, best_id = s, show_id

            if best_id:
                print(f"[{self.name}] Found Disney+ ID: {best_id} (score {best_score:.2f})")
                return best_id

        if unverified:
            print(f"[{self.name}] Using unverified fallback ID: {unverified}")
            return unverified
        return None

    def _disney_foreground(self) -> bool:
        """Return True if Disney+ is the currently focused app."""
        result = subprocess.run(
            ["adb", "-s", self.tv_ip, "shell", "dumpsys", "window", "windows"],
            capture_output=True, text=True
        )
        return "com.disney.disneyplus" in result.stdout

    def _build_disney_deep_link(self, show_id: str) -> str:
        """Entity UUIDs use /browse/entity-; numeric/alphanumeric IDs use /video/"""
        import re as _re
        if _re.fullmatch(r'[a-f0-9]{8}-[a-f0-9-]{27}', show_id):
            return f"https://www.disneyplus.com/browse/entity-{show_id}"
        return f"https://www.disneyplus.com/video/{show_id}"

    async def disney_play_show(self, msg: dict) -> None:
        """
        Play a Disney+ show by ID, optionally on a specific profile.
        Expected msg params: { "show_id": ["<id>"], "profile_name": "Kids" }  (profile_name optional)

        If only show_name is provided (LLM skipped StreamingAggregator), we do the
        search ourselves as a fallback so the command still works.

        If Disney+ is already in the foreground (e.g. after disney+_select_profile
        was run first), the launch + profile sequence is skipped and the deep link
        fires immediately — no sleep guesswork needed.
        """
        params       = msg["params"]
        profile_name = params.get("profile_name", "")

        # Prefer show_id; fall back to searching by show_name if LLM skipped the aggregator
        if "show_id" in params:
            show_id = params["show_id"][0]
        elif "show_name" in params:
            show_name = params["show_name"]
            print(f"[{self.name}] show_id missing — searching Disney+ for '{show_name}' directly.")
            show_id = self._search_disney_id(show_name)
            if not show_id:
                print(f"[{self.name}] Could not find Disney+ ID for '{show_name}'. Aborting.")
                return
        else:
            print(f"[{self.name}] disney_play_show requires either show_id or show_name in params.")
            return

        if self._disney_foreground():
            # App already open and past profile screen — fire the deep link directly.
            print(f"[{self.name}] Disney+ already in foreground, skipping launch.")
        else:
            # Cold start: launch → profile screen → home screen → deep link.
            await self.start_disney(msg)
            await asyncio.sleep(10.0)  # wait for profile screen

            if profile_name:
                found = await self._tap_profile_by_name(profile_name)
                if not found:
                    print(f"[{self.name}] Profile not found, selecting default.")
                    self._adb("input", "keyevent", "66")
            else:
                self._adb("input", "keyevent", "66")

            await asyncio.sleep(6.0)  # wait for home screen to finish loading

        self._adb("am", "start",
                  "-a", "android.intent.action.VIEW",
                  "-d", self._build_disney_deep_link(show_id),
                  "com.disney.disneyplus")

    async def netflix_play_show(self, msg: dict) -> None:
        show_id = msg["params"]["show_id"][0]
        print(f"[TV] Playing Netflix show ID: {show_id}")
        await self.start_netflix(msg)
        self._adb("am", "start",
                  "-n", "com.netflix.ninja/.MainActivity",
                  "-a", "android.intent.action.VIEW",
                  "-e", "amzn_deeplink_data", str(show_id))

    async def play_spotify_track(self, msg: dict) -> None:
        track_id = msg["params"]["track_id"][0]
        subprocess.run([
            "adb", "shell", "am", "start",
            "-a", "android.intent.action.VIEW",
            "-d", f"spotify:track:{track_id}",
            "-p", "com.spotify.tv.android"
        ])

    async def spotify_next(self, msg: dict) -> None:
        self._adb("input", "keyevent", "87")

    async def spotify_prev(self, msg: dict) -> None:
        self._adb("input", "keyevent", "88")

    # ── YouTube ────────────────────────────────────────────────────────────── #

    async def start_youtube(self, msg: dict) -> None:
        self._adb("am", "start", "-n",
                  "com.google.android.youtube.tv/com.google.android.apps.youtube.tv.activity.ShellActivity")

    async def youtube_play_video(self, msg: dict) -> None:
        """
        Play a YouTube video by ID.
        Expected msg params: { "video_id": ["<11-char-id>"] }

        YouTube video IDs are always 11 characters, e.g. "dQw4w9WgXcQ".
        The vnd.youtube: URI scheme is the most reliable deep link on Android TV —
        it opens the video directly without going through the browser.
        """
        video_id = msg["params"]["video_id"][0]
        print(f"[{self.name}] Playing YouTube video ID: {video_id}")
        self._adb("am", "start",
                  "-a", "android.intent.action.VIEW",
                  "-d", f"vnd.youtube:{video_id}",
                  "com.google.android.youtube.tv")

    async def play_luna_game(self, data: dict) -> None:
        game_id      = data["params"]["game_id"][0]
        LUNA_PACKAGE = "com.amazon.spiderpork"

        if not game_id:
            print(f"[{self.name}] Error: play_luna_game requires a game_id")
            return

        print(f"[{self.name}] Launching Luna game ID: {game_id}", flush=True)

        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "force-stop", LUNA_PACKAGE])
        await asyncio.sleep(1.0)

        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start",
                         "-a", "android.intent.action.VIEW",
                         "-d", f"https://luna.amazon.com/game/{game_id}",
                         LUNA_PACKAGE])
        await asyncio.sleep(5.0)

        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start",
                         "-a", "android.intent.action.VIEW",
                         "-e", "amzn_deeplink_data", str(game_id),
                         LUNA_PACKAGE])

        await self._bypass_profile_screen(wait=8.0)
        await asyncio.sleep(3.0)
        self._adb("input", "keyevent", "66")


async def main() -> None:
    tv = TclTv()
    await tv.run()

asyncio.run(main())