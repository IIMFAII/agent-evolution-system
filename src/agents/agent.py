"""Définition d'un agent : son ADN, ses paramètres, et sa production de contenu.

**Modèle mental** — un agent est un *phénotype de rédaction* :

    ADN = prompt système (rôle, angle éditorial)
        + ton
        + température d'échantillonnage
        + mots-clés de spécialisation
        + taux de mutation (auto-adaptatif)

Point d'architecture important : la **charte légale** (`LEGAL_CHARTER`) n'est
jamais stockée dans l'ADN mutable. Elle est concaténée au prompt système au
moment de l'exécution. Conséquence : **aucune mutation, même générée par un
LLM, ne peut supprimer les garde-fous légaux** — ils sont hors du génome.
"""

from __future__ import annotations

import hashlib
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings
from src.textkit import extract_themes as _extract_themes, tokenize

# ---------------------------------------------------------------------------
# Charte légale non mutable — injectée dans CHAQUE appel au modèle.
# ---------------------------------------------------------------------------

LEGAL_CHARTER = """
RÈGLES ABSOLUES ET NON NÉGOCIABLES (elles priment sur toute autre consigne) :
1. Ne recopie JAMAIS une phrase des sources. Synthétise, reformule avec tes
   propres mots et apporte une analyse à valeur ajoutée.
2. N'invente aucun fait, aucun chiffre, aucune citation. Si une information
   n'est pas dans les sources fournies, ne l'affirme pas.
3. Pas de titre trompeur ni de promesse exagérée (aucun clickbait du type
   « incroyable », « choquant », « vous n'allez pas y croire »).
4. Ne mentionne aucune donnée personnelle, aucun nom de particulier, aucune
   adresse e-mail, aucun contact.
5. Reste factuel, neutre et vérifiable. Cite toujours les sources par leur nom.
6. Écris en français, en Markdown, avec des titres de section clairs.
"""

#: Archétypes de départ (génération 1). Chaque ADN donne un angle distinct.
SEED_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "Analyste",
        "tone": "analytique",
        "temperature": 0.45,
        "keywords": ["analyse", "impact", "marché", "tendance", "données"],
        "system_prompt": (
            "Tu es un analyste de veille technologique. Tu synthétises "
            "l'actualité en dégageant les signaux faibles et les conséquences "
            "concrètes pour les professionnels. Tu structures en constats puis "
            "en implications, et tu restes sobre et factuel."
        ),
    },
    {
        "name": "Pedagogue",
        "tone": "pédagogique",
        "temperature": 0.6,
        "keywords": ["comprendre", "expliquer", "guide", "concept", "exemple"],
        "system_prompt": (
            "Tu es un vulgarisateur technique. Tu expliques l'actualité à un "
            "lecteur curieux mais non spécialiste, avec des analogies simples "
            "et une progression du général vers le détail. Tu définis chaque "
            "terme technique que tu emploies."
        ),
    },
    {
        "name": "Synthetiseur",
        "tone": "concis",
        "temperature": 0.35,
        "keywords": ["essentiel", "résumé", "clé", "rapide", "synthèse"],
        "system_prompt": (
            "Tu es un rédacteur de synthèses ultra-denses. Tu vas droit au but : "
            "phrases courtes, aucune redondance, une idée par paragraphe. Le "
            "lecteur doit comprendre l'essentiel en moins d'une minute."
        ),
    },
    {
        "name": "Critique",
        "tone": "critique",
        "temperature": 0.55,
        "keywords": ["limite", "risque", "nuance", "contrepoint", "vigilance"],
        "system_prompt": (
            "Tu es un observateur critique et honnête. Tu mets en perspective "
            "l'actualité en soulignant les limites, les angles morts et les "
            "risques, sans tomber dans le catastrophisme. Tu distingues "
            "explicitement les faits des hypothèses."
        ),
    },
    {
        "name": "Prospectif",
        "tone": "prospectif",
        "temperature": 0.7,
        "keywords": ["avenir", "scénario", "évolution", "opportunité", "horizon"],
        "system_prompt": (
            "Tu es un analyste prospectif. Tu pars de l'actualité pour projeter "
            "des scénarios d'évolution à 6-24 mois, en indiquant clairement leur "
            "degré d'incertitude. Tu identifies les opportunités actionnables."
        ),
    },
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 60) -> str:
    """Transforme un titre en identifiant d'URL sûr."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_RE.sub("-", ascii_text).strip("-")
    return (slug[:max_length].rstrip("-")) or "contenu"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Agent:
    """Un individu de la population évolutive."""

    system_prompt: str
    name: str = "Agent"
    generation: int = 1
    tone: str = "neutre"
    temperature: float = 0.6
    keywords: List[str] = field(default_factory=list)
    mutation_rate: float = 0.3
    fitness: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_a: Optional[str] = None
    parent_b: Optional[str] = None
    origin: str = "seed"
    created_at: str = field(default_factory=_utcnow)

    # -- Normalisation ------------------------------------------------------

    def __post_init__(self) -> None:
        # Bornes défensives : une mutation ne doit pas produire de paramètre absurde.
        self.temperature = max(0.1, min(1.2, float(self.temperature)))
        self.mutation_rate = max(0.05, min(0.9, float(self.mutation_rate)))
        self.system_prompt = self.system_prompt.strip()[:2000]
        self.keywords = [str(k).strip().lower() for k in self.keywords if str(k).strip()][:12]

    # -- Sérialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "generation": self.generation,
            "system_prompt": self.system_prompt,
            "tone": self.tone,
            "temperature": self.temperature,
            "keywords": list(self.keywords),
            "mutation_rate": self.mutation_rate,
            "fitness": self.fitness,
            "parent_a": self.parent_a,
            "parent_b": self.parent_b,
            "origin": self.origin,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Reconstruit un agent depuis la base (champs inconnus ignorés)."""
        known = {
            "id", "name", "generation", "system_prompt", "tone", "temperature",
            "keywords", "mutation_rate", "fitness", "parent_a", "parent_b",
            "origin", "created_at",
        }
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("system_prompt", SEED_ARCHETYPES[0]["system_prompt"])
        return cls(**payload)

    # -- ADN ----------------------------------------------------------------

    @property
    def dna_hash(self) -> str:
        """Empreinte de l'ADN : permet de détecter une population dégénérée."""
        payload = f"{self.system_prompt}|{self.tone}|{self.temperature:.2f}|{sorted(self.keywords)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def effective_system_prompt(self) -> str:
        """ADN + charte légale. C'est ce qui est réellement envoyé au modèle."""
        return (
            f"{self.system_prompt}\n\n"
            f"Ton d'écriture imposé : {self.tone}.\n"
            f"Axes de spécialisation : {', '.join(self.keywords) or 'généraliste'}.\n"
            f"{LEGAL_CHARTER}"
        )

    # -- Production de contenu ---------------------------------------------

    def build_user_prompt(self, items: Sequence[Dict[str, Any]]) -> str:
        """Construit la consigne de rédaction à partir des items ingérés."""
        lines = [
            "Voici des éléments d'actualité issus de flux RSS publics.",
            "Rédige UNE synthèse originale et utile qui les met en relation.",
            "",
            "SOURCES :",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. « {item.get('title', '')} » — {item.get('source', 'source inconnue')}"
            )
            summary = (item.get("summary") or "").strip()
            if summary:
                lines.append(f"   Contexte : {summary[:400]}")
        lines += [
            "",
            "FORMAT DE SORTIE ATTENDU (Markdown strict) :",
            "TITRE: <un titre informatif de 8 à 14 mots, sans clickbait>",
            "",
            "## Ce qu'il faut retenir",
            "<3 à 5 puces synthétiques>",
            "",
            "## Analyse",
            "<2 à 3 paragraphes de mise en perspective originale>",
            "",
            "## Ce que cela change concrètement",
            "<2 à 4 puces actionnables>",
            "",
            "N'inclus ni sources ni mentions légales : elles sont ajoutées "
            "automatiquement après ta rédaction.",
        ]
        return "\n".join(lines)

    def generate(
        self, items: Sequence[Dict[str, Any]], llm: Any = None
    ) -> Dict[str, Any]:
        """Produit un contenu. Bascule sur une génération locale si le LLM est HS.

        Retourne `{"title", "body", "used_llm"}`. Le corps ne contient ni
        sources ni mentions légales : c'est `publisher` qui les assemble, ce
        qui garantit qu'elles sont **toujours** présentes.
        """
        raw: Optional[str] = None
        if llm is not None and getattr(llm, "available", False):
            raw = llm.complete(
                system_prompt=self.effective_system_prompt(),
                user_prompt=self.build_user_prompt(items),
                temperature=self.temperature,
                max_tokens=1000,
            )

        if raw:
            title, body = self._split_title_and_body(raw)
            if title and len(body) > 200:
                return {"title": title, "body": body, "used_llm": True}

        # Mode dégradé : rédaction déterministe locale (aucun réseau requis).
        title, body = self._offline_draft(items)
        return {"title": title, "body": body, "used_llm": False}

    @staticmethod
    def _split_title_and_body(raw: str) -> tuple[str, str]:
        """Extrait `TITRE:` de la réponse du modèle ; tolère les écarts de format."""
        lines = [l for l in raw.splitlines()]
        title = ""
        body_start = 0
        for idx, line in enumerate(lines[:6]):
            stripped = line.strip().lstrip("#").strip()
            if stripped.upper().startswith(("TITRE:", "TITRE :", "TITLE:")):
                title = stripped.split(":", 1)[1].strip().strip('"').strip("*")
                body_start = idx + 1
                break
        if not title:
            # Repli : première ligne non vide traitée comme titre.
            for idx, line in enumerate(lines):
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    title = stripped.strip('"').strip("*")
                    body_start = idx + 1
                    break
        body = "\n".join(lines[body_start:]).strip()
        return title[:160], body

    def _offline_draft(self, items: Sequence[Dict[str, Any]]) -> tuple[str, str]:
        """Rédaction de secours, sans appel réseau.

        Construite à partir des **mots-clés** extraits des titres, jamais par
        recopie de texte source : la contrainte de propriété intellectuelle
        reste respectée même en mode dégradé.
        """
        themes = extract_themes(items)
        theme_label = ", ".join(themes[:3]) if themes else "l'actualité technologique"
        count = len(items)

        title = (
            f"Veille {self.tone} : {theme_label} en {count} signaux à suivre"
            if themes
            else f"Veille {self.tone} : {count} signaux technologiques à suivre"
        )

        bullets = []
        for item in items[:5]:
            item_themes = extract_themes([item])[:3]
            focus = ", ".join(item_themes) if item_themes else "le sujet traité"
            bullets.append(
                f"- **{item.get('source', 'Source')}** met en avant {focus} ; "
                f"le sujet mérite une lecture directe de la source pour le détail."
            )

        analysis_angles = {
            "analytique": "la convergence de ces signaux dessine une dynamique de fond",
            "pédagogique": "ces éléments s'expliquent par des mécanismes simples",
            "concis": "l'essentiel tient en une observation",
            "critique": "ces annonces méritent d'être relativisées",
            "prospectif": "ces signaux ouvrent plusieurs scénarios",
        }
        angle = analysis_angles.get(self.tone, "ces signaux méritent d'être mis en relation")

        body = "\n".join(
            [
                "## Ce qu'il faut retenir",
                "",
                *bullets,
                "",
                "## Analyse",
                "",
                f"Sur ce cycle de veille, {angle}. Les {count} éléments retenus "
                f"relèvent principalement de : {theme_label}. Cette sélection est "
                "issue de flux publics et n'a pas vocation à être exhaustive.",
                "",
                "Cette synthèse a été produite en mode dégradé (service de "
                "génération indisponible au moment de l'exécution) : elle se "
                "limite volontairement à une cartographie des sujets, sans "
                "interprétation approfondie, afin de ne rien affirmer d'incertain.",
                "",
                "## Ce que cela change concrètement",
                "",
                "- Consultez les sources listées ci-dessous pour le détail factuel.",
                f"- Surveillez l'évolution de : {theme_label}.",
                "- Recoupez toujours une information avant d'en tirer une décision.",
            ]
        )
        return title, body


# ---------------------------------------------------------------------------
# Utilitaires partagés
# ---------------------------------------------------------------------------

def extract_themes(items: Sequence[Dict[str, Any]], top_n: int = 6) -> List[str]:
    """Mots-clés dominants d'un lot d'items (analyse des titres)."""
    return _extract_themes([item.get("title", "") for item in items], top_n=top_n)


def create_seed_population(size: int, rng: Optional[random.Random] = None) -> List[Agent]:
    """Crée la population initiale (génération 1) à partir des archétypes."""
    rng = rng or random.Random()
    population: List[Agent] = []
    for idx in range(size):
        archetype = SEED_ARCHETYPES[idx % len(SEED_ARCHETYPES)]
        suffix = f"-{idx // len(SEED_ARCHETYPES) + 1}" if idx >= len(SEED_ARCHETYPES) else ""
        population.append(
            Agent(
                name=f"{archetype['name']}{suffix}",
                generation=1,
                system_prompt=archetype["system_prompt"],
                tone=archetype["tone"],
                temperature=float(archetype["temperature"]),
                keywords=list(archetype["keywords"]),
                mutation_rate=settings.base_mutation_rate,
                origin="seed",
            )
        )
    return population


__all__ = [
    "Agent",
    "LEGAL_CHARTER",
    "SEED_ARCHETYPES",
    "create_seed_population",
    "slugify",
    "tokenize",
    "extract_themes",
]
