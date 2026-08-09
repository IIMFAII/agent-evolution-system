"""Configuration globale du système.

Principes appliqués ici :
  * **Aucun secret en dur.** Tout provient de l'environnement (fichier `.env`
    en local, GitHub Secrets en CI). L'absence de secret n'est jamais fatale :
    le système bascule en mode dégradé hors-ligne.
  * **Validation stricte** via pydantic : une valeur d'environnement corrompue
    est rejetée puis remplacée par le défaut sûr, plutôt que de propager une
    erreur au milieu du pipeline.
  * **Politesse réseau par défaut** : timeout court, délai entre requêtes,
    User-Agent identifiable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

try:  # python-dotenv est optionnel : en CI, les variables sont déjà injectées.
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover - dépendance absente = comportement normal
    pass


# Racine du projet (dossier contenant `main.py`).
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Textes légaux — injectés systématiquement dans TOUT contenu publié.
# ---------------------------------------------------------------------------

# Mention utilisée lorsqu'un lien affilié est effectivement présent.
DISCLAIMER_AUTOMATION = (
    "Transparence : ce contenu est rédigé à l'aide d'outils d'analyse "
    "automatisés et contient des liens affiliés."
)

# Variante utilisée quand aucun programme d'affiliation n'est configuré.
# Annoncer des liens affiliés inexistants serait une déclaration inexacte :
# la mention doit décrire la réalité du contenu publié, pas une intention.
DISCLAIMER_AUTOMATION_NO_AFFILIATE = (
    "Transparence : ce contenu est rédigé à l'aide d'outils d'analyse "
    "automatisés. Il ne contient aucun lien affilié ni contenu sponsorisé."
)

DISCLAIMER_SOURCES = (
    "Les informations sont synthétisées à partir de flux publics librement "
    "consultables, cités en fin d'article. Aucune donnée personnelle n'est "
    "collectée ni traitée."
)

DISCLAIMER_AFFILIATE_DETAIL = (
    "Certains liens sont des liens d'affiliation : un achat effectué via ces "
    "liens peut générer une commission, sans surcoût pour vous. Cela "
    "n'influence pas le contenu de l'analyse."
)

#: Marqueurs recherchés par l'évaluateur pour valider la conformité d'un texte.
COMPLIANCE_MARKERS = ("Transparence", "automatis")


# ---------------------------------------------------------------------------
# Helpers de lecture d'environnement (tolérants aux valeurs invalides)
# ---------------------------------------------------------------------------


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = _env_str(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Flux RSS de repli : tous publiquement destinés à la syndication.
DEFAULT_FEEDS: List[str] = [
    "https://hnrss.org/frontpage",  # Hacker News (flux officiel)
    "https://feeds.arstechnica.com/arstechnica/technology-lab",  # Ars Technica
    "https://www.lemondeinformatique.fr/flux-rss/thematique/tout-les-articles/rss.xml",
]


class Settings(BaseModel):
    """Configuration validée du système. Instanciée une seule fois (`settings`)."""

    # --- LLM ---------------------------------------------------------------
    groq_api_key: Optional[str] = Field(default=None, repr=False)
    groq_model: str = "llama-3.1-8b-instant"
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # --- Évolution ---------------------------------------------------------
    population_size: int = Field(default=5, ge=2, le=50)
    elite_size: int = Field(default=1, ge=1)
    base_mutation_rate: float = Field(default=0.30, ge=0.0, le=1.0)
    max_items_per_run: int = Field(default=12, ge=1, le=100)

    # --- Ingestion ---------------------------------------------------------
    rss_feeds: List[str] = Field(default_factory=lambda: list(DEFAULT_FEEDS))
    http_user_agent: str = (
        "AgentEvolutionBot/1.0 (+https://github.com/; contact via GitHub issues)"
    )
    request_timeout: float = Field(default=15.0, gt=0, le=60)
    min_delay_between_requests: float = Field(default=2.0, ge=0.0, le=30.0)
    respect_robots_txt: bool = True

    # --- Affiliation -------------------------------------------------------
    affiliate_tag: str = ""
    affiliate_base_url: str = ""

    # --- Monétisation ------------------------------------------------------
    #: Audience mensuelle réelle. `None` = non mesurée. L'analyse la traite
    #: comme nulle : on ne monétise pas ce qu'on ne sait pas compter.
    monthly_audience: Optional[int] = None
    #: False dès qu'un humain relit les contenus avant publication. Ce drapeau
    #: ouvre les canaux dont les CGU exigent une valeur ajoutée humaine — ne le
    #: passer à False que si la relecture a réellement lieu.
    fully_automated_content: bool = True

    # --- Publication -------------------------------------------------------
    publish_dir: str = "docs"
    publish_webhook_url: str = ""

    # --- Divers ------------------------------------------------------------
    db_path: str = "data/evolution.db"
    log_level: str = "INFO"

    @field_validator("elite_size")
    @classmethod
    def _elite_not_too_big(cls, v: int, info) -> int:
        """L'élite ne peut pas absorber toute la population (sinon plus d'évolution)."""
        population = info.data.get("population_size", 5)
        return max(1, min(v, max(1, population - 1)))

    @field_validator("rss_feeds")
    @classmethod
    def _only_http_feeds(cls, v: List[str]) -> List[str]:
        """Filtre défensif : uniquement des URL http(s), pas de schéma exotique."""
        feeds = [u for u in v if u.startswith(("http://", "https://"))]
        return feeds or list(DEFAULT_FEEDS)

    # --- Propriétés dérivées ----------------------------------------------

    @property
    def llm_enabled(self) -> bool:
        """True si une clé API est disponible. Sinon : mode dégradé hors-ligne."""
        return bool(self.groq_api_key)

    @property
    def affiliation_enabled(self) -> bool:
        """L'affiliation reste désactivée tant qu'aucun tag n'est configuré."""
        return bool(self.affiliate_tag and self.affiliate_base_url)

    @property
    def db_file(self) -> Path:
        path = Path(self.db_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def publish_path(self) -> Path:
        path = Path(self.publish_dir)
        return path if path.is_absolute() else BASE_DIR / path


def load_settings() -> Settings:
    """Construit la configuration depuis l'environnement.

    En cas d'incohérence (valeur hors bornes par exemple), on retombe sur la
    configuration par défaut plutôt que de faire échouer tout le pipeline.
    """
    try:
        return Settings(
            groq_api_key=_env_str("GROQ_API_KEY") or None,
            groq_model=_env_str("GROQ_MODEL", "llama-3.1-8b-instant"),
            population_size=_env_int("POPULATION_SIZE", 5),
            elite_size=_env_int("ELITE_SIZE", 1),
            base_mutation_rate=_env_float("BASE_MUTATION_RATE", 0.30),
            max_items_per_run=_env_int("MAX_ITEMS_PER_RUN", 12),
            rss_feeds=_env_list("RSS_FEEDS", DEFAULT_FEEDS),
            http_user_agent=_env_str(
                "HTTP_USER_AGENT",
                "AgentEvolutionBot/1.0 (+https://github.com/; contact via GitHub issues)",
            ),
            request_timeout=_env_float("REQUEST_TIMEOUT", 15.0),
            min_delay_between_requests=_env_float("MIN_DELAY_BETWEEN_REQUESTS", 2.0),
            affiliate_tag=_env_str("AFFILIATE_TAG"),
            affiliate_base_url=_env_str("AFFILIATE_BASE_URL"),
            monthly_audience=(
                _env_int("MONTHLY_AUDIENCE", 0) if _env_str("MONTHLY_AUDIENCE") else None
            ),
            fully_automated_content=_env_str("FULLY_AUTOMATED_CONTENT", "true").lower()
            not in ("false", "0", "no"),
            publish_dir=_env_str("PUBLISH_DIR", "docs"),
            publish_webhook_url=_env_str("PUBLISH_WEBHOOK_URL"),
            db_path=_env_str("DB_PATH", "data/evolution.db"),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )
    except ValidationError as exc:  # pragma: no cover - garde-fou
        logging.getLogger(__name__).warning(
            "Configuration invalide (%s) — retour aux valeurs par défaut.", exc
        )
        return Settings()


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure le logging applicatif (idempotent)."""
    resolved = (level or settings.log_level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("agent-evolution")


#: Instance unique partagée par tous les modules.
settings = load_settings()
