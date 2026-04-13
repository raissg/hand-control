import requests
import time
import os
import sys

BASE_URL = "https://jobs.apple.com/en-us/search"
PARAMS = {
    "search": "Early Career software",
    "sort": "relevance",
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "apple_jobs_html")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PAGES = 200
DELAY_SECONDS = 2


def scrape_pages():
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(50, MAX_PAGES + 1):
        params = {**PARAMS, "page": page}
        print(f"Fetching page {page}...", end=" ", flush=True)

        try:
            resp = session.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"FAILED: {e}")
            break

        out_path = os.path.join(OUTPUT_DIR, f"page_{page:03d}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp.text)

        size_kb = len(resp.text) / 1024
        print(f"OK ({size_kb:.1f} KB) -> {out_path}")

        if "No results were found" in resp.text or "0 Result" in resp.text:
            print("No more results. Stopping.")
            break

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. Pages saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    scrape_pages()
