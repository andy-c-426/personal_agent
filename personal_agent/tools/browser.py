import ipaddress
import json
import logging
import re
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

_MAX_PAGE_CHARS = 6000


class BrowserSession:
    """Stateful, lazy-start browser wrapper using Playwright (sync API)."""

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._history: list[str] = []

    @property
    def headless(self) -> bool:
        return self._headless

    @headless.setter
    def headless(self, v: bool) -> None:
        if v != self._headless:
            self._headless = v
            self.close()

    def close(self) -> None:
        try:
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._context = None
        self._playwright = None

    def _ensure_started(self) -> None:
        if self._page is not None:
            return
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self._headless)
            self._context = self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/148.0.0.0 Safari/537.36"
                ),
            )
            self._page = self._context.new_page()
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "host not found" in str(e):
                raise RuntimeError(
                    "Chromium not installed. Run: playwright install chromium"
                ) from e
            raise

    def _page_text(self) -> str:
        try:
            self._page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        time.sleep(0.5)
        text = self._page.inner_text("body") or ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > _MAX_PAGE_CHARS:
            text = text[:_MAX_PAGE_CHARS] + "\n\n... [truncated]"
        return text

    def navigate(self, url: str) -> str:
        self._ensure_started()
        self._page.goto(url, timeout=15000)
        self._history.append(url)
        title = self._page.title()
        content = self._page_text()
        return f"Title: {title}\nURL: {self._page.url}\n\n{content}"

    def get_content(self) -> str:
        self._ensure_started()
        title = self._page.title()
        content = self._page_text()
        return f"Title: {title}\nURL: {self._page.url}\n\n{content}"

    def click(self, link_text: str) -> str:
        self._ensure_started()
        link_text_clean = link_text.strip()

        # Try role-based locator first, then text-based fallback
        try:
            link = self._page.get_by_role("link", name=re.compile(re.escape(link_text_clean), re.IGNORECASE))
            if link.count() > 0:
                self._history.append(self._page.url)
                link.first.click()
                self._page.wait_for_load_state("networkidle", timeout=10000)
                return self._page_text()
        except Exception:
            pass

        # Fallback: has-text locator
        try:
            escaped = link_text_clean.replace("'", "\\'")
            locator = self._page.locator(f"a:has-text('{escaped}')")
            if locator.count() > 0:
                self._history.append(self._page.url)
                locator.first.click()
                self._page.wait_for_load_state("networkidle", timeout=10000)
                return self._page_text()
        except Exception:
            pass

        return json.dumps({
            "error": f"No link found matching '{link_text_clean}'",
            "available_links": self._visible_links()[:20],
        })

    def go_back(self) -> str:
        self._ensure_started()
        if not self._history:
            return json.dumps({"error": "No previous page to go back to"})
        self._history.pop()  # current page
        self._page.go_back()
        self._page.wait_for_load_state("networkidle", timeout=10000)
        return self._page_text()

    def _visible_links(self) -> list[str]:
        try:
            links = self._page.locator("a[href]:visible").all()
            texts = []
            for link in links[:50]:
                t = (link.inner_text() or "").strip()
                href = (link.get_attribute("href") or "")[:80]
                if t and not href.startswith("#"):
                    texts.append(f"{t} -> {href}")
            return texts
        except Exception:
            return []

    def google_search(self, query: str) -> str:
        """Search Google and return structured results. Falls back to DuckDuckGo on CAPTCHA."""
        self._ensure_started()

        result = self._try_google(query)
        if result is not None:
            return result

        logger.info("Google search blocked (CAPTCHA), falling back to DuckDuckGo")
        return self._search_duckduckgo(query)

    def _try_google(self, query: str) -> str | None:
        try:
            self._page.goto("https://www.google.com", timeout=15000)
        except Exception:
            return None

        # Accept cookies if the dialog appears
        try:
            self._page.click("button:has-text('Accept all')", timeout=2000)
        except Exception:
            pass

        # Type query
        try:
            q_input = self._page.locator("textarea[name='q']")
            if q_input.count() == 0:
                q_input = self._page.locator("input[name='q']")
            q_input.fill(query)
            q_input.press("Enter")
        except Exception:
            return None

        try:
            self._page.wait_for_selector("#search", timeout=10000)
        except Exception:
            pass

        # Check for CAPTCHA
        if self._is_captcha():
            return None

        return self._parse_google_results()

    def _is_captcha(self) -> bool:
        try:
            if self._page.locator("#captcha-form").count() > 0:
                return True
            if "sorry/index" in self._page.url:
                return True
        except Exception:
            pass
        return False

    def _parse_google_results(self) -> str:
        results = []
        try:
            blocks = self._page.locator("#search .g").all()
            for block in blocks[:10]:
                title_el = block.locator("h3").first
                link_el = block.locator("a[href]").first
                snippet_el = block.locator("[data-sncf], .VwiC3b, span.aCOpRe").first

                title = title_el.inner_text().strip() if title_el.count() > 0 else ""
                url = link_el.get_attribute("href") if link_el.count() > 0 else ""
                snippet = snippet_el.inner_text().strip() if snippet_el.count() > 0 else ""

                # Skip internal Google links
                if not url or "/search?q=" in url or "/preferences" in url:
                    continue
                if not url.startswith("http"):
                    continue

                results.append({"title": title, "url": url, "snippet": snippet})
        except Exception as e:
            logger.debug("Failed to parse Google results: %s", e)

        if not results:
            return json.dumps({"results": [], "message": "Could not parse search results."})
        return json.dumps({"results": results, "source": "google"}, ensure_ascii=False)

    def _search_duckduckgo(self, query: str) -> str:
        try:
            self._page.goto("https://duckduckgo.com/", timeout=15000)
            q_input = self._page.locator("input[name='q']")
            q_input.fill(query)
            q_input.press("Enter")
            self._page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            return json.dumps({"results": [], "error": "DuckDuckGo search failed."})

        results = []
        try:
            articles = self._page.locator("article[data-testid='result']").all()
            for article in articles[:10]:
                title_el = article.locator("h2").first
                link_el = article.locator("a[data-testid='result-title-a']").first
                snippet_el = article.locator("[data-result='snippet']").first

                title = title_el.inner_text().strip() if title_el.count() > 0 else ""
                url = link_el.get_attribute("href") if link_el.count() > 0 else ""
                snippet = snippet_el.inner_text().strip() if snippet_el.count() > 0 else ""

                if not url or not url.startswith("http"):
                    continue

                results.append({"title": title, "url": url, "snippet": snippet})
        except Exception:
            pass

        if not results:
            return json.dumps({"results": [], "message": "Could not parse search results."})
        return json.dumps({"results": results, "source": "duckduckgo"}, ensure_ascii=False)


# --- Tool functions ---


def browser_search(query: str, session: BrowserSession) -> str:
    """Search the web using Google (auto-fallback to DuckDuckGo)."""
    return session.google_search(query)


_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]", "metadata.google.internal"}

_BLOCKED_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),     # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),    # IPv6 link-local
]


def _is_internal_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return True  # block unparseable hosts
    if hostname in _BLOCKED_HOSTS or hostname.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(hostname)
        # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
        if addr.version == 6 and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if addr.is_loopback or addr.is_unspecified:
            return True
        return any(addr in net for net in _BLOCKED_NETS)
    except ValueError:
        pass
    return False


def browser_navigate(url: str, session: BrowserSession) -> str:
    """Navigate to a URL and return the page content."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if _is_internal_url(url):
        return json.dumps({"error": "Navigation to internal/private hosts is blocked."})
    return session.navigate(url)


def browser_get_content(session: BrowserSession) -> str:
    """Re-read the current page content."""
    return session.get_content()


def browser_click(link_text: str, session: BrowserSession) -> str:
    """Click a link on the current page and return the new page content."""
    return session.click(link_text)


def browser_go_back(session: BrowserSession) -> str:
    """Go back to the previous page."""
    return session.go_back()
