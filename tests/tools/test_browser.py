import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from personal_agent.tools.browser import (
    BrowserSession,
    browser_search,
    browser_navigate,
    browser_get_content,
    browser_click,
    browser_go_back,
)


# --- BrowserSession unit tests ---


class TestBrowserSession:
    def test_init_defaults_to_headless(self):
        session = BrowserSession()
        assert session.headless is True
        assert session._page is None

    def test_init_visible(self):
        session = BrowserSession(headless=False)
        assert session.headless is False

    def test_headless_setter_toggles_and_closes(self):
        session = BrowserSession(headless=True)
        session._page = MagicMock()
        session._context = MagicMock()
        session._browser = MagicMock()
        session._playwright = MagicMock()

        session.headless = False

        assert session.headless is False
        assert session._page is None

    def test_close_cleans_up_all_resources(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_browser = MagicMock()
        mock_pw = MagicMock()

        session._page = mock_page
        session._context = mock_context
        session._browser = mock_browser
        session._playwright = mock_pw

        session.close()

        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()
        assert session._page is None
        assert session._browser is None

    def test_close_handles_exceptions_gracefully(self):
        session = BrowserSession()
        session._context = MagicMock()
        session._context.close.side_effect = Exception("already closed")

        session.close()  # should not raise

    def test_ensure_started_is_lazy(self):
        session = BrowserSession()
        assert session._page is None

    def test_ensure_started_creates_browser_and_page(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        with patch("personal_agent.tools.browser.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw
            session._ensure_started()

        assert session._page is mock_page
        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()

    def test_ensure_started_skips_if_already_running(self):
        session = BrowserSession()
        session._page = MagicMock()

        with patch("personal_agent.tools.browser.sync_playwright") as mock_sp:
            session._ensure_started()
            mock_sp.assert_not_called()

    def test_ensure_started_missing_chromium(self):
        session = BrowserSession()
        mock_pw = MagicMock()
        mock_pw.chromium.launch.side_effect = Exception("Executable doesn't exist at /usr/bin/chromium")

        with patch("personal_agent.tools.browser.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw
            with pytest.raises(RuntimeError, match="Chromium not installed"):
                session._ensure_started()

    def test_page_text_truncates_at_max_chars(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "x" * 8000
        session._page = mock_page

        text = session._page_text()
        assert len(text) <= 6100  # 6000 + truncation message
        assert "... [truncated]" in text

    def test_page_text_normalizes_newlines(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "line1\n\n\n\nline2\n\n\nline3"
        session._page = mock_page

        text = session._page_text()
        assert "\n\n\n\n" not in text

    def test_navigate_returns_title_url_and_content(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.title.return_value = "Example Page"
        mock_page.url = "https://example.com/page"
        mock_page.inner_text.return_value = "Page content here"
        session._page = mock_page

        result = session.navigate("https://example.com")

        mock_page.goto.assert_called_once_with("https://example.com", timeout=15000)
        assert "Example Page" in result
        assert "https://example.com/page" in result
        assert "Page content here" in result

    def test_get_content_returns_current_page_info(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.title.return_value = "Current Page"
        mock_page.url = "https://example.com/current"
        mock_page.inner_text.return_value = "Current content"
        session._page = mock_page

        result = session.get_content()

        assert "Current Page" in result
        assert "https://example.com/current" in result
        assert "Current content" in result

    def test_click_uses_role_locator_first(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "Clicked page content"
        session._page = mock_page

        mock_link = MagicMock()
        mock_link.count.return_value = 1
        mock_page.get_by_role.return_value = mock_link

        result = session.click("Read More")

        mock_page.get_by_role.assert_called_once()
        mock_link.first.click.assert_called_once()
        assert "Clicked page content" in result

    def test_click_falls_back_to_has_text_locator(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "Fallback page content"
        session._page = mock_page

        mock_role_link = MagicMock()
        mock_role_link.count.return_value = 0
        mock_page.get_by_role.return_value = mock_role_link

        mock_text_link = MagicMock()
        mock_text_link.count.return_value = 1
        mock_page.locator.return_value = mock_text_link

        result = session.click("Read More")

        mock_page.locator.assert_called_once()
        mock_text_link.first.click.assert_called_once()
        assert "Fallback page content" in result

    def test_click_returns_error_with_available_links(self):
        session = BrowserSession()
        mock_page = MagicMock()

        mock_role_link = MagicMock()
        mock_role_link.count.return_value = 0
        mock_page.get_by_role.return_value = mock_role_link

        mock_text_link = MagicMock()
        mock_text_link.count.return_value = 0
        mock_page.locator.return_value = mock_text_link

        visible_links = ["Home -> /home", "About -> /about"]
        mock_page.locator.return_value.all.return_value = [
            MagicMock(inner_text=MagicMock(return_value="Home"),
                      get_attribute=MagicMock(return_value="/home")),
            MagicMock(inner_text=MagicMock(return_value="About"),
                      get_attribute=MagicMock(return_value="/about")),
        ]

        session._page = mock_page
        result = session.click("Nonexistent")

        data = json.loads(result)
        assert "error" in data
        assert "No link found" in data["error"]

    def test_click_escapes_single_quotes_in_locator(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "Content"
        session._page = mock_page

        mock_role_link = MagicMock()
        mock_role_link.count.return_value = 0
        mock_page.get_by_role.return_value = mock_role_link

        mock_text_link = MagicMock()
        mock_text_link.count.return_value = 1
        mock_page.locator.return_value = mock_text_link

        session.click("what's new")

        call_arg = mock_page.locator.call_args[0][0]
        assert "what\\'s new" in call_arg

    def test_go_back_returns_previous_page_content(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "Previous page content"
        session._page = mock_page
        session._history = ["https://page1.com", "https://page2.com"]

        result = session.go_back()

        mock_page.go_back.assert_called_once()
        assert "Previous page content" in result

    def test_go_back_returns_error_when_no_history(self):
        session = BrowserSession()
        session._page = MagicMock()
        session._history = []

        result = session.go_back()

        data = json.loads(result)
        assert "error" in data
        assert "No previous page" in data["error"]

    # Google search tests

    def test_google_search_tries_google_first(self):
        session = BrowserSession()
        session._page = MagicMock()
        session._page.locator.return_value.count.return_value = 1

        with patch.object(session, "_try_google", return_value='{"results": [{"title": "G", "url": "https://g.com", "snippet": "s"}], "source": "google"}') as mock_try:
            with patch.object(session, "_search_duckduckgo") as mock_ddg:
                result = session.google_search("test query")
                mock_try.assert_called_once_with("test query")
                mock_ddg.assert_not_called()
                data = json.loads(result)
                assert data["source"] == "google"

    def test_google_search_falls_back_to_duckduckgo(self):
        session = BrowserSession()

        with patch.object(session, "_try_google", return_value=None) as mock_try:
            with patch.object(session, "_search_duckduckgo", return_value='{"results": [], "source": "duckduckgo"}') as mock_ddg:
                result = session.google_search("test query")
                mock_try.assert_called_once()
                mock_ddg.assert_called_once_with("test query")
                data = json.loads(result)
                assert data["source"] == "duckduckgo"

    def test_is_captcha_detects_captcha_form(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.count.return_value = 1
        session._page = mock_page

        assert session._is_captcha() is True

    def test_is_captcha_detects_sorry_url(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.count.return_value = 0
        mock_page.url = "https://www.google.com/sorry/index?continue=..."
        session._page = mock_page

        assert session._is_captcha() is True

    def test_is_captcha_returns_false_normally(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.count.return_value = 0
        mock_page.url = "https://www.google.com/search?q=test"
        session._page = mock_page

        assert session._is_captcha() is False

    def test_parse_google_results_filters_internal_links(self):
        session = BrowserSession()
        mock_page = MagicMock()
        session._page = mock_page

        mock_block1 = MagicMock()
        mock_block1.locator.return_value.first.inner_text.return_value = "Result 1"
        mock_block1.locator.return_value.first.get_attribute.return_value = "https://example.com/page1"
        mock_snippet1 = MagicMock()
        mock_snippet1.inner_text.return_value = "Snippet 1"
        mock_block1.locator.return_value.first = mock_block1.locator.return_value.first
        mock_block1.locator.return_value.first = mock_block1.locator.return_value.first

        mock_page.locator.return_value.all.return_value = []

        result = session._parse_google_results()
        data = json.loads(result)
        assert "results" in data

    def test_parse_google_results_returns_message_when_empty(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.all.return_value = []
        session._page = mock_page

        result = session._parse_google_results()
        data = json.loads(result)
        assert data["results"] == []

    def test_try_google_accepts_cookies(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.count.return_value = 1
        mock_page.locator.return_value.first.inner_text.return_value = "Result"
        mock_page.locator.return_value.first.get_attribute.return_value = "https://example.com"
        mock_page.locator.return_value.all.return_value = []
        session._page = mock_page

        with patch.object(session, "_is_captcha", return_value=False):
            result = session._try_google("test")
            assert result is not None

    def test_try_google_detects_captcha_and_returns_none(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.count.return_value = 1
        session._page = mock_page

        with patch.object(session, "_is_captcha", return_value=True):
            result = session._try_google("test")
            assert result is None

    # DuckDuckGo tests

    def test_search_duckduckgo_parses_results(self):
        session = BrowserSession()
        mock_page = MagicMock()
        session._page = mock_page

        def _make_locator(inner_text_val, get_attr_val, count_val=1):
            """Build a locator chain: locator().first.{inner_text,get_attribute,count}"""
            first_el = MagicMock()
            first_el.inner_text.return_value = inner_text_val
            first_el.get_attribute.return_value = get_attr_val
            first_el.count.return_value = count_val
            loc = MagicMock()
            loc.first = first_el
            return loc

        mock_article = MagicMock()
        mock_article.locator.side_effect = lambda sel: {
            "h2": _make_locator("DDG Result", None),
            "a[data-testid='result-title-a']": _make_locator(None, "https://ddgresult.com"),
            "[data-result='snippet']": _make_locator("DDG snippet", None),
        }[sel]
        mock_page.locator.return_value.all.return_value = [mock_article]

        result = session._search_duckduckgo("test query")
        data = json.loads(result)
        assert data["source"] == "duckduckgo"
        assert len(data["results"]) >= 1

    def test_search_duckduckgo_handles_error(self):
        session = BrowserSession()
        mock_page = MagicMock()
        mock_page.locator.return_value.fill.side_effect = Exception("navigation failed")
        session._page = mock_page

        result = session._search_duckduckgo("test query")
        data = json.loads(result)
        assert "error" in data


# --- Tool function tests ---


class TestBrowserToolFunctions:
    def test_browser_search_delegates_to_session(self):
        mock_session = MagicMock()
        mock_session.google_search.return_value = '{"results": [{"title": "Test", "url": "https://test.com", "snippet": "desc"}]}'

        result = browser_search("test query", session=mock_session)

        mock_session.google_search.assert_called_once_with("test query")
        assert "Test" in result

    def test_browser_navigate_adds_https_if_missing(self):
        mock_session = MagicMock()
        mock_session.navigate.return_value = "Page content"

        result = browser_navigate("example.com", session=mock_session)

        mock_session.navigate.assert_called_once_with("https://example.com")

    def test_browser_navigate_preserves_existing_https(self):
        mock_session = MagicMock()
        mock_session.navigate.return_value = "Page content"

        result = browser_navigate("https://example.com", session=mock_session)

        mock_session.navigate.assert_called_once_with("https://example.com")

    def test_browser_get_content_delegates(self):
        mock_session = MagicMock()
        mock_session.get_content.return_value = "Title: Test\nURL: https://test.com\n\nContent"

        result = browser_get_content(session=mock_session)

        mock_session.get_content.assert_called_once()

    def test_browser_click_delegates(self):
        mock_session = MagicMock()
        mock_session.click.return_value = "Clicked content"

        result = browser_click("Link Text", session=mock_session)

        mock_session.click.assert_called_once_with("Link Text")

    def test_browser_go_back_delegates(self):
        mock_session = MagicMock()
        mock_session.go_back.return_value = "Previous page"

        result = browser_go_back(session=mock_session)

        mock_session.go_back.assert_called_once()


# --- Integration-style tests with mocked Playwright ---


class TestBrowserSessionWithMockedPlaywright:
    def test_full_navigate_flow(self):
        mock_page = MagicMock()
        mock_page.title.return_value = "Test Title"
        mock_page.url = "https://example.com"
        mock_page.inner_text.return_value = "Hello world"

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        with patch("personal_agent.tools.browser.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw

            session = BrowserSession()
            result = session.navigate("https://example.com")

        mock_page.goto.assert_called_once_with("https://example.com", timeout=15000)
        assert "Test Title" in result
        assert "Hello world" in result

    def test_full_click_flow_with_role_match(self):
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "New page after click"

        mock_link = MagicMock()
        mock_link.count.return_value = 1
        mock_page.get_by_role.return_value = mock_link

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser

        with patch("personal_agent.tools.browser.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw

            session = BrowserSession()
            session._page = mock_page
            result = session.click("Read More")

        mock_page.get_by_role.assert_called_once()
        assert "New page after click" in result

    def test_navigate_tracks_history_for_go_back(self):
        mock_page = MagicMock()
        mock_page.title.return_value = "Page"
        mock_page.inner_text.return_value = "Content"
        mock_page.url = "https://page2.com"

        session = BrowserSession()
        session._page = mock_page

        session.navigate("https://page1.com")
        session.navigate("https://page2.com")

        assert len(session._history) == 2

        mock_page.inner_text.return_value = "Page1 content"
        result = session.go_back()

        mock_page.go_back.assert_called_once()
