"""Assemblage et publication des contenus (Markdown + HTML pour GitHub Pages).

Le publisher est le **garant final de la conformité** : c'est lui, et non
l'agent, qui assemble le document livré. Les blocs obligatoires — sources,
mention de transparence, divulgation d'affiliation — sont ajoutés par le code,
donc impossibles à omettre par une dérive du modèle.

Sortie produite dans `docs/` (servi gratuitement par GitHub Pages) :

    docs/
    ├── index.html              # sommaire des dernières publications
    ├── posts/AAAA-MM-JJ-slug.md
    └── posts/AAAA-MM-JJ-slug.html
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote, urlencode, urlparse

import requests

from src.agents.agent import slugify
from src.config import (
    DISCLAIMER_AFFILIATE_DETAIL,
    DISCLAIMER_AUTOMATION,
    DISCLAIMER_AUTOMATION_NO_AFFILIATE,
    DISCLAIMER_SOURCES,
    settings,
)

logger = logging.getLogger(__name__)


def legal_footer() -> List[str]:
    """Bloc de mentions légales adapté à la configuration réelle du site.

    La mention d'affiliation n'est affichée que si des liens affiliés sont
    réellement publiés : annoncer une affiliation inexistante serait tout aussi
    trompeur que de la dissimuler.
    """
    if settings.affiliation_enabled:
        return [
            f"**{DISCLAIMER_AUTOMATION}**",
            "",
            DISCLAIMER_SOURCES,
            "",
            DISCLAIMER_AFFILIATE_DETAIL,
        ]
    return [f"**{DISCLAIMER_AUTOMATION_NO_AFFILIATE}**", "", DISCLAIMER_SOURCES]


HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="index, follow">
<title>{title}</title>
<meta name="description" content="{description}">
{head_extra}
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1d21; --muted: #5b6570;
    --accent: #2f6feb; --border: #e3e7ec; --card: #f7f9fb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14171a; --fg: #e8eaed; --muted: #9aa4ae;
      --accent: #6ea8ff; --border: #2a2f35; --card: #1b1f24;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--fg);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 44rem; margin: 0 auto; }}
  h1 {{ font-size: 1.9rem; line-height: 1.25; margin: 0 0 .5rem; }}
  h2 {{ font-size: 1.25rem; margin: 2rem 0 .6rem; }}
  a {{ color: var(--accent); }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin: .35rem 0; }}
  .meta {{ color: var(--muted); font-size: .875rem; margin-bottom: 2rem; }}
  .legal {{
    margin-top: 2.5rem; padding: 1rem 1.1rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 8px;
    font-size: .85rem; color: var(--muted);
  }}
  .legal p {{ margin: .4rem 0; }}
  hr {{ border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }}
  code {{ background: var(--card); padding: .1rem .3rem; border-radius: 4px; }}
</style>
</head>
<body>
<main>
{content}
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mini-convertisseur Markdown → HTML
# ---------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE_RE = re.compile(r"`([^`]+)`")


def _inline_md(text: str, external_rel: str = "noopener") -> str:
    """Convertit les marqueurs inline. Échappe l'HTML **avant** toute insertion."""
    escaped = html.escape(text, quote=False)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\1</em>", escaped)

    def _link(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        # Seuls http(s) sont autorisés : neutralise javascript: et data:.
        if not url.startswith(("http://", "https://", "#", "./", "posts/")):
            return label
        rel = external_rel if url.startswith("http") else ""
        rel_attr = f' rel="{rel}" target="_blank"' if rel else ""
        return f'<a href="{html.escape(url, quote=True)}"{rel_attr}>{label}</a>'

    return _LINK_RE.sub(_link, escaped)


def markdown_to_html(markdown: str) -> str:
    """Conversion volontairement minimale (titres, listes, paragraphes, liens).

    Suffisant pour le Markdown contrôlé que produit ce système, et sans
    dépendance supplémentaire. Tout contenu est échappé : pas d'injection HTML
    possible depuis un flux RSS.
    """
    lines = markdown.splitlines()
    out: List[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            close_list()
            continue
        if stripped.startswith("---"):
            close_list()
            out.append("<hr>")
            continue
        if stripped.startswith("#"):
            close_list()
            level = min(6, len(stripped) - len(stripped.lstrip("#")))
            out.append(f"<h{level}>{_inline_md(stripped.lstrip('#').strip())}</h{level}>")
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_md(stripped[2:])}</li>")
            continue

        close_list()
        out.append(f"<p>{_inline_md(stripped)}</p>")

    close_list()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class Publisher:
    """Assemble les documents conformes puis les écrit sur disque."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        webhook_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else settings.publish_path
        self.posts_dir = self.output_dir / "posts"
        self.webhook_url = webhook_url if webhook_url is not None else settings.publish_webhook_url
        self.session = session or requests.Session()

    # -- Affiliation --------------------------------------------------------

    @staticmethod
    def affiliate_url(base_url: str, tag: str, target: str = "") -> str:
        """Construit un lien affilié traçable. Chaîne vide si non configuré."""
        if not base_url or not tag:
            return ""
        separator = "&" if "?" in base_url else "?"
        params = {"tag": tag}
        if target:
            params["ref"] = target[:120]
        return f"{base_url}{separator}{urlencode(params, quote_via=quote)}"

    def _affiliate_block(self, theme: str) -> str:
        """Bloc d'affiliation — **toujours** accompagné de sa divulgation.

        Retourne une chaîne vide tant qu'aucun programme d'affiliation n'est
        configuré : par défaut, le système ne publie aucun lien commercial.
        """
        if not settings.affiliation_enabled:
            return ""
        url = self.affiliate_url(settings.affiliate_base_url, settings.affiliate_tag, theme)
        if not url:
            return ""
        return (
            "\n## Ressources recommandées\n\n"
            f"- [Ressources et outils sur ce sujet]({url}) "
            "— *lien affilié, signalé comme tel.*\n"
        )

    # -- Assemblage ---------------------------------------------------------

    def assemble(
        self,
        title: str,
        body: str,
        source_items: Sequence[Dict[str, Any]],
        agent_name: str = "",
        generation: int = 0,
        theme: str = "",
    ) -> str:
        """Produit le document Markdown final, mentions légales incluses.

        C'est ce document complet qui est ensuite soumis à l'évaluateur : la
        note de conformité porte donc bien sur ce qui sera réellement publié.
        """
        published_at = datetime.now(timezone.utc).strftime("%d/%m/%Y à %H:%M UTC")

        parts = [
            f"# {title}",
            "",
            f"*Publié le {published_at} — synthèse automatisée "
            f"(agent « {agent_name} », génération {generation}).*",
            "",
            body.strip(),
            self._affiliate_block(theme),
            "",
            "## Sources",
            "",
        ]

        seen_links = set()
        for item in source_items:
            link = item.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            domain = urlparse(link).netloc or item.get("source", "source")
            parts.append(f"- [{item.get('title', link)}]({link}) — {domain}")

        if not seen_links:
            parts.append("- Aucune source externe pour ce cycle.")

        parts += [
            "",
            "---",
            "",
            *legal_footer(),
            "",
            "*Les liens ci-dessus renvoient vers les publications d'origine, "
            "seules références faisant foi.*",
        ]

        return "\n".join(parts).strip() + "\n"

    # -- Écriture -----------------------------------------------------------

    @staticmethod
    def _meta_description(markdown: str, fallback: str) -> str:
        """Extrait un résumé utile pour le moteur de recherche.

        La ligne de métadonnées (« Publié le… — synthèse automatisée ») faisait
        auparavant office de description : c'est l'extrait affiché dans les
        résultats de recherche, et il ne disait rien du contenu.
        """
        for line in markdown.splitlines():
            text = re.sub(r"[#*`>\[\]]|\(https?://[^)]*\)", "", line).strip()
            if len(text) < 60:
                continue
            if text.lower().startswith(("publié le", "transparence", "les informations")):
                continue
            return text[:180]
        return fallback[:180]

    def _head_extra(self, path_from_root: str, title: str, description: str) -> str:
        """Balises canoniques et Open Graph. Vides si `SITE_URL` est absente."""
        if not settings.site_url:
            return ""
        url = f"{settings.site_url.rstrip('/')}/{path_from_root.lstrip('/')}"
        esc = lambda s: html.escape(s, quote=True)  # noqa: E731
        return (
            f'<link rel="canonical" href="{esc(url)}">\n'
            f'<link rel="alternate" type="application/rss+xml" '
            f'title="{esc(settings.site_title)}" href="{esc(settings.site_url)}/feed.xml">\n'
            f'<meta property="og:type" content="article">\n'
            f'<meta property="og:title" content="{esc(title)}">\n'
            f'<meta property="og:description" content="{esc(description)}">\n'
            f'<meta property="og:url" content="{esc(url)}">\n'
            f'<meta name="twitter:card" content="summary">'
        )

    def publish(self, title: str, markdown: str, slug: Optional[str] = None) -> Dict[str, str]:
        """Écrit le post en Markdown et en HTML. Retourne les chemins produits."""
        self.posts_dir.mkdir(parents=True, exist_ok=True)
        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stem = f"{date_prefix}-{slug or slugify(title)}"

        md_path = self.posts_dir / f"{stem}.md"
        html_path = self.posts_dir / f"{stem}.html"

        md_path.write_text(markdown, encoding="utf-8")

        description = self._meta_description(markdown, title)
        html_path.write_text(
            HTML_TEMPLATE.format(
                title=html.escape(title, quote=True),
                description=html.escape(description, quote=True),
                head_extra=self._head_extra(f"posts/{stem}.html", title, description),
                content=markdown_to_html(markdown) + '\n<p><a href="../index.html">← Toutes les publications</a></p>',
            ),
            encoding="utf-8",
        )

        logger.info("Contenu publié : %s", md_path)
        return {"markdown": str(md_path), "html": str(html_path), "slug": stem}

    def build_index(self, limit: int = 50) -> Optional[Path]:
        """(Re)génère la page d'accueil listant les publications les plus récentes."""
        self.posts_dir.mkdir(parents=True, exist_ok=True)
        posts = sorted(self.posts_dir.glob("*.html"), reverse=True)[:limit]

        lines = [
            "# Veille automatisée",
            "",
            "Synthèses produites par une population d'agents évolutifs, "
            "réévaluée toutes les deux heures.",
            "",
            "[Tableau de bord de l'évolution](dashboard.html) — fitness par "
            "génération, signaux d'évaluation et historique des cycles.",
            "",
            "## Publications",
            "",
        ]
        if posts:
            for post in posts:
                # Nom de fichier : AAAA-MM-JJ-slug → date + titre lisible.
                pretty_date = post.stem[:10]
                title = post.stem[11:].replace("-", " ").capitalize() or post.stem
                lines.append(f"- `{pretty_date}` — [{title}](posts/{post.name})")
        else:
            lines.append("- Aucune publication pour le moment.")

        lines += ["", "---", "", *legal_footer()]

        markdown = "\n".join(lines)
        index_path = self.output_dir / "index.html"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                HTML_TEMPLATE.format(
                    title=html.escape(settings.site_title, quote=True),
                    description=html.escape(
                        "Synthèses de veille produites par une population d'agents évolutifs.",
                        quote=True,
                    ),
                    head_extra=self._head_extra(
                        "index.html",
                        settings.site_title,
                        "Synthèses de veille produites par une population d'agents évolutifs.",
                    ),
                    content=markdown_to_html(markdown),
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture de l'index impossible : %s", exc)
            return None
        return index_path

    # -- Indexabilité -------------------------------------------------------

    def build_sitemap(self) -> Optional[Path]:
        """Écrit `docs/sitemap.xml`. Sans lui, les pages ne sont pas découvertes.

        Requiert `SITE_URL` : un sitemap contenant des URL relatives est ignoré
        par les moteurs, donc on préfère ne rien écrire plutôt qu'un fichier
        invalide qui donnerait l'illusion d'être indexé.
        """
        if not settings.site_url:
            logger.info("SITE_URL non configurée — sitemap non généré.")
            return None
        try:
            self.posts_dir.mkdir(parents=True, exist_ok=True)
            base = settings.site_url.rstrip("/")
            urls = [(f"{base}/", "daily", "1.0"), (f"{base}/dashboard.html", "hourly", "0.6")]
            for post in sorted(self.posts_dir.glob("*.html"), reverse=True):
                urls.append((f"{base}/posts/{post.name}", "monthly", "0.8"))

            body = "\n".join(
                f"  <url><loc>{html.escape(loc, quote=True)}</loc>"
                f"<changefreq>{freq}</changefreq><priority>{prio}</priority></url>"
                for loc, freq, prio in urls
            )
            path = self.output_dir / "sitemap.xml"
            path.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                f"{body}\n</urlset>\n",
                encoding="utf-8",
            )
            return path
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture du sitemap impossible : %s", exc)
            return None

    def build_robots(self) -> Optional[Path]:
        """Écrit `docs/robots.txt`, en autorisant l'indexation et en pointant le sitemap.

        Le site applique aux autres le `robots.txt` qu'il publie pour lui-même :
        il autorise explicitement ce qu'il consomme lui-même par flux.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            lines = ["User-agent: *", "Allow: /", "", "# Données du pipeline, sans intérêt pour l'indexation", "Disallow: /data.json", "Disallow: /status.json", ""]
            if settings.site_url:
                lines.append(f"Sitemap: {settings.site_url.rstrip('/')}/sitemap.xml")
            path = self.output_dir / "robots.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture de robots.txt impossible : %s", exc)
            return None

    def build_feed(self, contents: Sequence[Dict[str, Any]]) -> Optional[Path]:
        """Écrit `docs/feed.xml` : un flux RSS des publications du site.

        C'est le seul canal de diffusion légitime d'un système automatisé : le
        lecteur s'abonne, personne ne reçoit de message non sollicité.
        """
        if not settings.site_url:
            return None
        try:
            base = settings.site_url.rstrip("/")
            items = []
            for c in list(contents)[:30]:
                link = f"{base}/posts/{c.get('slug', '')}.html"
                items.append(
                    "    <item>\n"
                    f"      <title>{html.escape(str(c.get('title', '')), quote=True)}</title>\n"
                    f"      <link>{html.escape(link, quote=True)}</link>\n"
                    f"      <guid isPermaLink=\"true\">{html.escape(link, quote=True)}</guid>\n"
                    f"      <description>{html.escape(DISCLAIMER_AUTOMATION, quote=True)}</description>\n"
                    "    </item>"
                )
            path = self.output_dir / "feed.xml"
            path.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<rss version="2.0"><channel>\n'
                f"    <title>{html.escape(settings.site_title, quote=True)}</title>\n"
                f"    <link>{html.escape(base, quote=True)}</link>\n"
                f"    <description>{html.escape(DISCLAIMER_SOURCES, quote=True)}</description>\n"
                "    <language>fr</language>\n"
                + "\n".join(items)
                + "\n</channel></rss>\n",
                encoding="utf-8",
            )
            return path
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture du flux impossible : %s", exc)
            return None

    def write_dashboard_data(self, payload: Dict[str, Any]) -> Optional[Path]:
        """Écrit `docs/data.json`, la source unique du tableau de bord.

        Séparé de `status.json` à dessein : `status.json` reste l'état compact
        du dernier cycle (lisible dans le résumé du job CI), tandis que ce
        fichier porte les séries historiques dont le tableau de bord a besoin.
        `dashboard.html` est statique et ne change donc pas à chaque cycle.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "data.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8",
            )
            return path
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture des données du tableau de bord impossible : %s", exc)
            return None

    def write_status(self, payload: Dict[str, Any]) -> Optional[Path]:
        """Écrit un état machine-lisible du dernier cycle (`docs/status.json`)."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "status.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return path
        except OSError as exc:  # pragma: no cover
            logger.error("Écriture du statut impossible : %s", exc)
            return None

    # -- Diffusion optionnelle ---------------------------------------------

    def notify_webhook(self, title: str, summary: str, url: str = "") -> bool:
        """Notifie un webhook sortant si configuré. Jamais bloquant.

        Aucun destinataire individuel n'est ciblé : il s'agit d'un canal que
        l'opérateur a lui-même configuré (Discord, Slack, Zapier…). Aucun
        message non sollicité n'est envoyé à un tiers.
        """
        if not self.webhook_url:
            return False
        try:
            response = self.session.post(
                self.webhook_url,
                json={"content": f"**{title}**\n{summary[:400]}\n{url}".strip()},
                timeout=settings.request_timeout,
                headers={"User-Agent": settings.http_user_agent},
            )
            if response.status_code >= 400:
                logger.warning("Webhook rejeté (HTTP %d).", response.status_code)
                return False
            return True
        except requests.RequestException as exc:
            logger.warning("Webhook injoignable : %s", exc)
            return False


__all__ = ["Publisher", "markdown_to_html"]
