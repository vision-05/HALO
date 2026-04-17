from discovery.src.base_agent import BaseAgent
import requests
import re
from ddgs import DDGS
from urllib.parse import unquote

class StreamingAggregator(BaseAgent):
    def __init__(self, name):
        super().__init__(name, "Aggregator")
        self.urls = {"netflix": "site:netflix.com/title",
                     "disney+": "site:disneyplus.com"}
        self.regexes = {"netflix": r'netflix\.com/title/(\d+)',
                        "disney+": r'disneyplus\.com/.*?(?:series|movies|video)/(?:[^/]+/)*([a-zA-Z0-9]{8,})(?:[/?#]|$)'}
        
        self.handlers = {"get_id_from_title": self.get_id_from_title,
                         "explore_shows_by_search_term": self.get_titles_from_search}

    def get_id_from_title(self, show_name, site="netflix"):
        """
        Uses the DuckDuckGo Search API to find the official Netflix Show ID.
        This bypasses the anti-bot blocks triggered by raw HTML scraping.
        """
        print(f"[LanguageAgent] Searching the web for Netflix ID: '{show_name}'...")
    
        query = f"{self.urls[site]} {show_name}"
    
        try:
            # Initialize the DDGS client
            search_client = DDGS()
        
            # Fetch up to 5 results to give us options
            results = list(search_client.text(query, max_results=5))
        
            fallback_id = None
            
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "").lower()
            
                # The Regex Pattern: Catch IDs of any length (Netflix uses 7, 8, or 9 digits)
                match = re.search(self.regexes[site], url)
            
                if match:
                    show_id = match.group(1)
                
                    # Store the very first ID as a fallback just in case
                    if not fallback_id:
                        fallback_id = show_id
                    
                    # VERIFICATION: Ensure the requested show name is actually in the search result's title!
                    # We split into words to handle variations (e.g. "The Office" matching "The Office (U.S.)")
                    search_words = show_name.lower().split()
                    if all(word in title for word in search_words):
                        print(f"[LanguageAgent] Verified title match! Found ID {show_id} for '{show_name}'")
                        return show_id

            # If no exact match was found, use the first Netflix ID we saw
            if fallback_id:
                print(f"[LanguageAgent] No exact title match. Best guess is ID {fallback_id}")
                return fallback_id

            print(f"[LanguageAgent] Could not find a Netflix ID for '{show_name}'.")
            return None
            
        except Exception as e:
            print(f"[LanguageAgent] Error scraping for ID: {e}")
            return None

    def get_titles_from_search(self, search_string):
        pass

    def add_site(self, site, url, regex):
        self.urls[site] = url
        self.regexes[site] = regex