"""Ingestion **légale** de données publiques (flux RSS/Atom uniquement).

Règles appliquées, dans l'ordre, avant toute requête :

  1. **robots.txt** — récupéré et respecté pour chaque hôte (y compris la
     directive `Crawl-delay`). Un `Disallow` fait sauter le flux, sans exception.
  2. **Rate limiting** — délai minimal garanti entre deux requêtes sortantes,
     et arrêt immédiat du flux en cas de `429` / `Retry-After` (on ne réessaie
     pas dans le même cycle : la prochaine exécution aura lieu 2 h plus tard).
  3. **User-Agent identifiable** — jamais d'usurpation de navigateur.
  4. **Aucun scraping de page** — on ne lit que des flux de syndication, dont
     la vocation même est d'être consommés par des machines. Aucun contenu
     protégé n'est téléchargé, aucun paywall n'est contourné.
  5. **Aucune donnée personnelle** — les champs auteur/e-mail éventuellement
     présents dans le flux sont explicitement écartés (RGPD).
"""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from src.config import settings
from src.database import fingerprint

logger = logging.getLogger(__name__)

try:
    import feedparser

    _FEEDPARSER_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - dépendance absente
    feedparser = None  # type: ignore[assignment]
    _FEEDPARSER_ERROR = exc


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

#: Champs de flux susceptibles de contenir des données personnelles : ignorés.
PII_FIELDS = ("author", "author_detail", "publisher_detail", "contributors", "email")


def clean_text(raw: str, max_length: int = 600) -> str:
    """Nettoie un extrait de flux : HTML retiré, entités décodées, espaces normalisés."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_length]


class RobotsPolicy:
    """Cache mémoire des `robots.txt` par hôte, avec délai de politesse."""

    def __init__(self, user_agent: str, timeout: float) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: Dict[str, Optional[RobotFileParser]] = {}

    def _load(self, base: str) -> Optional[RobotFileParser]:
        """Charge le robots.txt d'un hôte. `None` = pas de politique exploitable."""
        if base in self._cache:
            return self._cache[base]

        parser: Optional[RobotFileParser] = None
        try:
            response = requests.get(
                f"{base}/robots.txt",
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
            elif response.status_code in (401, 403):
                # Accès au robots.txt refusé : la RFC 9309 impose de considérer
                # l'ensemble du site comme interdit. On le matérialise ici.
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Disallow: /"])
            # 404 / autre : aucune politique publiée → autorisé par défaut.
        except requests.RequestException as exc:
            logger.info("robots.txt injoignable pour %s (%s) — accès prudent autorisé.", base, exc)

        self._cache[base] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        base = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._load(base)
        if parser is None:
            return True
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:  # pragma: no cover - parser tolérant
            return True

    def crawl_delay(self, url: str) -> float:
        """Délai imposé par l'hôte, 0.0 si non spécifié."""
        parsed = urlparse(url)
        parser = self._cache.get(f"{parsed.scheme}://{parsed.netloc}")
        if parser is None:
            return 0.0
        try:
            delay = parser.crawl_delay(self.user_agent)
            return float(delay) if delay else 0.0
        except Exception:  # pragma: no cover
            return 0.0


class Ingestor:
    """Collecteur de flux RSS/Atom publics, poli et *fail-safe*."""

    def __init__(
        self,
        feeds: Optional[List[str]] = None,
        user_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        min_delay: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.feeds = feeds if feeds is not None else list(settings.rss_feeds)
        self.user_agent = user_agent or settings.http_user_agent
        self.timeout = timeout if timeout is not None else settings.request_timeout
        self.min_delay = min_delay if min_delay is not None else settings.min_delay_between_requests
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/rss+xml, application/xml, text/xml"}
        )
        self.robots = RobotsPolicy(self.user_agent, self.timeout)
        self._last_request_at = 0.0
        #: Diagnostic du dernier cycle, remonté dans le rapport public.
        self.report: Dict[str, str] = {}

    # -- Politesse ----------------------------------------------------------

    def _throttle(self, extra_delay: float = 0.0) -> None:
        """Garantit un délai minimal entre deux requêtes sortantes."""
        required = max(self.min_delay, extra_delay)
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < required:
            time.sleep(required - elapsed)
        self._last_request_at = time.monotonic()

    # -- Récupération -------------------------------------------------------

    def _fetch(self, url: str) -> Optional[bytes]:
        """Télécharge un flux. `None` si interdit, indisponible ou rate-limité."""
        if settings.respect_robots_txt and not self.robots.can_fetch(url):
            logger.warning("Flux ignoré (robots.txt interdit l'accès) : %s", url)
            self.report[url] = "bloqué par robots.txt"
            return None

        self._throttle(self.robots.crawl_delay(url))

        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("Flux injoignable %s : %s", url, exc)
            self.report[url] = f"erreur réseau ({type(exc).__name__})"
            return None

        if response.status_code == 429:
            # Limite atteinte : on abandonne ce flux pour ce cycle. Aucun
            # réessai immédiat — c'est le respect strict du rate limit.
            retry_after = response.headers.get("Retry-After", "non précisé")
            logger.warning("429 sur %s (Retry-After: %s) — flux abandonné ce cycle.", url, retry_after)
            self.report[url] = "limite de requêtes atteinte (429)"
            return None
        if response.status_code >= 400:
            logger.warning("Flux %s indisponible (HTTP %d).", url, response.status_code)
            self.report[url] = f"HTTP {response.status_code}"
            return None

        self.report[url] = "ok"
        return response.content

    # -- Parsing ------------------------------------------------------------

    def _parse(self, payload: bytes, source_url: str) -> List[Dict[str, Any]]:
        """Convertit un flux en items normalisés, sans aucune donnée personnelle."""
        if feedparser is None:  # pragma: no cover - dépendance absente
            logger.error("feedparser indisponible (%s) — flux ignoré.", _FEEDPARSER_ERROR)
            return []

        parsed = feedparser.parse(payload)
        source_name = clean_text(getattr(parsed.feed, "title", "") or urlparse(source_url).netloc, 120)

        items: List[Dict[str, Any]] = []
        for entry in parsed.entries:
            title = clean_text(entry.get("title", ""), 300)
            link = (entry.get("link") or "").strip()
            if not title or not link.startswith(("http://", "https://")):
                continue

            summary = clean_text(entry.get("summary", "") or entry.get("description", ""), 600)

            published_at = None
            struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if struct:
                try:
                    published_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", struct)
                except (TypeError, ValueError):
                    published_at = None

            # ⚠️ RGPD : on ne recopie volontairement AUCUN champ auteur/contact.
            items.append(
                {
                    "fingerprint": fingerprint(link, title),
                    "source": source_name,
                    "source_url": source_url,
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published_at": published_at,
                }
            )
        return items

    # -- API publique -------------------------------------------------------

    def collect(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Collecte les items de tous les flux configurés.

        Ne lève jamais d'exception : un flux en échec est simplement absent du
        résultat. Retourne au plus `limit` items, équitablement répartis entre
        les sources qui ont répondu.
        """
        limit = limit if limit is not None else settings.max_items_per_run
        per_feed: List[List[Dict[str, Any]]] = []

        for url in self.feeds:
            payload = self._fetch(url)
            if not payload:
                continue
            try:
                items = self._parse(payload, url)
            except Exception as exc:  # pragma: no cover - flux malformé
                logger.warning("Flux %s illisible : %s", url, exc)
                self.report[url] = "flux illisible"
                continue
            if items:
                per_feed.append(items)
                logger.info("%d item(s) collecté(s) depuis %s", len(items), url)

        # Répartition en round-robin : aucune source ne monopolise le cycle.
        collected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        index = 0
        while len(collected) < limit and any(index < len(f) for f in per_feed):
            for feed_items in per_feed:
                if index < len(feed_items) and len(collected) < limit:
                    item = feed_items[index]
                    if item["fingerprint"] not in seen:
                        seen.add(item["fingerprint"])
                        collected.append(item)
            index += 1

        logger.info(
            "Ingestion terminée : %d item(s) retenu(s) sur %d flux interrogé(s).",
            len(collected),
            len(self.feeds),
        )
        return collected


__all__ = ["Ingestor", "RobotsPolicy", "clean_text"]
