"""Fonction de *fitness* : signaux proxy mesurables en moins de 2 heures.

On ne dispose pas de vraies métriques d'audience à l'échelle d'un cycle de 2 h.
La fitness combine donc cinq signaux calculables immédiatement :

    fitness = 0.30 · conformité      (mentions légales, absence de données perso)
            + 0.20 · lisibilité      (Flesch adapté au français, Kandel & Moles)
            + 0.20 · pertinence      (couverture des mots-clés vs sources, anti-bourrage)
            + 0.20 · potentiel CTR   (analyse du titre, heuristique + modèle léger)
            + 0.10 · originalité     (faible recouvrement n-grammes avec les sources)

**La conformité est un verrou, pas une pondération.** Si les mentions légales
obligatoires manquent, ou si une donnée personnelle est détectée, la fitness
est plafonnée très bas : un contenu non conforme ne peut jamais gagner la
sélection, quelles que soient ses autres qualités.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from src.agents.agent import tokenize
from src.config import COMPLIANCE_MARKERS, settings

logger = logging.getLogger(__name__)


# --- Pondérations ----------------------------------------------------------

WEIGHTS = {
    "compliance": 0.30,
    "readability": 0.20,
    "keyword_score": 0.20,
    "ctr_score": 0.20,
    "originality": 0.10,
}

#: Plafond de fitness appliqué à un contenu non conforme (verrou légal).
NON_COMPLIANT_CAP = 0.10

# --- Détecteurs de non-conformité -----------------------------------------

#: Données personnelles (RGPD) — leur présence invalide le contenu.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}")

#: Accroches trompeuses interdites (publicité mensongère / clickbait abusif).
CLICKBAIT_RE = re.compile(
    r"\b("
    r"incroyable|choquant|hallucinant|vous\s+n['e]\s*allez\s+pas\s+y\s+croire"
    r"|astuce\s+secrète|le\s+secret\s+que|100\s*%\s+garanti|argent\s+facile"
    r"|miracle|révolutionnaire|jamais\s+vu|urgent\s*!|dernière\s+chance"
    r"|gagnez\s+de\s+l'argent|sans\s+effort"
    r")\b",
    re.IGNORECASE,
)

#: Promesses non vérifiables (publicité mensongère).
OVERPROMISE_RE = re.compile(
    r"\b(garanti|assuré à 100|infaillible|sans risque|résultats? garantis?)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouyàâäéèêëîïôöùûüœ]+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Signaux élémentaires
# ---------------------------------------------------------------------------


def count_syllables_fr(word: str) -> int:
    """Compte approximatif de syllabes en français (groupes de voyelles).

    Le « e » muet final n'est pas compté, sauf s'il constitue la seule syllabe.
    """
    word = word.lower().strip()
    if not word:
        return 0
    groups = _VOWEL_GROUP_RE.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_score(text: str) -> float:
    """Lisibilité normalisée [0,1] via l'indice de Kandel & Moles (Flesch FR).

        207 − 1.015 · (mots/phrases) − 73.6 · (syllabes/mots)

    L'échelle brute (0 = très difficile, 100 = très facile) est ensuite recentrée
    sur la bande 40–80, qui correspond à un article de veille bien calibré : trop
    simple est aussi pénalisant que trop absconé.
    """
    words = re.findall(r"[\wà-öø-ÿ'-]+", text or "")
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    if len(words) < 20 or not sentences:
        return 0.0

    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = sum(count_syllables_fr(w) for w in words) / len(words)
    raw = 207 - 1.015 * words_per_sentence - 73.6 * syllables_per_word
    raw = max(0.0, min(100.0, raw))

    # Bande optimale 40–80 → 1.0 ; décroissance linéaire de part et d'autre.
    if 40 <= raw <= 80:
        return 1.0
    if raw < 40:
        return max(0.0, raw / 40)
    return max(0.0, 1.0 - (raw - 80) / 20)


def keyword_score(
    text: str, agent_keywords: Sequence[str], source_items: Sequence[Dict[str, Any]]
) -> float:
    """Pertinence thématique : couverture des sujets sources, sans bourrage.

    Deux moitiés :
      * **couverture** — part des thèmes issus des titres sources réellement
        traités dans le texte (le contenu parle bien de ce qu'il prétend) ;
      * **spécialisation** — part des mots-clés d'ADN de l'agent présents.

    Une densité supérieure à 4 % pour un même terme est considérée comme du
    *keyword stuffing* et pénalise le score.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    token_set = set(tokens)

    source_themes = set()
    for item in source_items:
        source_themes.update(tokenize(item.get("title", "")))
    coverage = (
        len(token_set & source_themes) / min(len(source_themes), 30)
        if source_themes
        else 0.5
    )
    coverage = min(1.0, coverage)

    keywords = [k.lower() for k in agent_keywords]
    specialization = (
        sum(1 for k in keywords if k in text.lower()) / len(keywords) if keywords else 0.5
    )

    score = 0.6 * coverage + 0.4 * specialization

    # Pénalité anti-bourrage.
    from collections import Counter

    most_common = Counter(tokens).most_common(1)
    if most_common:
        density = most_common[0][1] / len(tokens)
        if density > 0.04:
            score *= max(0.3, 1.0 - (density - 0.04) * 10)

    return max(0.0, min(1.0, score))


def ctr_heuristic(title: str) -> float:
    """Estimation heuristique du potentiel de clic d'un titre (sans réseau).

    Critères : longueur utile (45–75 caractères), présence d'un repère chiffré,
    richesse lexicale, absence de capitales criardes et de ponctuation excessive.
    Tout marqueur de clickbait est **pénalisé**, jamais récompensé : l'objectif
    est un titre attractif *et* honnête.
    """
    title = (title or "").strip()
    if not title:
        return 0.0

    score = 0.5
    length = len(title)
    if 45 <= length <= 75:
        score += 0.20
    elif 30 <= length < 45 or 75 < length <= 95:
        score += 0.08
    else:
        score -= 0.15

    if re.search(r"\d", title):  # un repère chiffré ancre la promesse
        score += 0.10

    informative_words = [w for w in tokenize(title) if len(w) > 4]
    if len(informative_words) >= 4:
        score += 0.10

    if CLICKBAIT_RE.search(title):
        score -= 0.45
    if OVERPROMISE_RE.search(title):
        score -= 0.25
    if title.isupper() or sum(1 for c in title if c.isupper()) > len(title) * 0.4:
        score -= 0.20
    if title.count("!") + title.count("?") > 1:
        score -= 0.15

    return max(0.0, min(1.0, score))


def originality_score(text: str, source_items: Sequence[Dict[str, Any]]) -> float:
    """Détecte la recopie : recouvrement de n-grammes avec les textes sources.

    Un 8-gramme commun avec une source signe un copier-coller littéral et
    ramène le score à 0 (violation de la propriété intellectuelle).
    """
    body_tokens = tokenize(text)
    if len(body_tokens) < 30:
        return 0.0

    source_text = " ".join(
        f"{i.get('title', '')} {i.get('summary', '')}" for i in source_items
    )
    source_tokens = tokenize(source_text)
    if len(source_tokens) < 10:
        return 1.0

    def ngrams(tokens: List[str], n: int) -> set:
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    # Verrou : recopie littérale détectée.
    if ngrams(body_tokens, 8) & ngrams(source_tokens, 8):
        logger.warning("Recopie littérale détectée (8-gramme commun) — originalité nulle.")
        return 0.0

    body_tri = ngrams(body_tokens, 3)
    source_tri = ngrams(source_tokens, 3)
    if not body_tri:
        return 0.0
    overlap = len(body_tri & source_tri) / len(body_tri)

    # Un recouvrement faible (<10 %) est normal et sain : on reste sur le sujet
    # sans recopier. Au-delà de 35 %, la reformulation est jugée insuffisante.
    if overlap <= 0.10:
        return 1.0
    if overlap >= 0.35:
        return 0.0
    return 1.0 - (overlap - 0.10) / 0.25


def compliance_report(text: str, title: str = "") -> Dict[str, Any]:
    """Contrôle légal du contenu final. Retourne le détail des manquements."""
    full = f"{title}\n{text}"
    issues: List[str] = []

    # 1. Transparence sur l'automatisation (obligatoire, cf. config).
    has_disclaimer = all(marker.lower() in full.lower() for marker in COMPLIANCE_MARKERS)
    if not has_disclaimer:
        issues.append("mention de transparence absente")

    # 2. Citation des sources.
    has_sources = "source" in full.lower() and "http" in full.lower()
    if not has_sources:
        issues.append("sources non citées")

    # 3. Données personnelles (RGPD).
    if EMAIL_RE.search(full):
        issues.append("adresse e-mail détectée (RGPD)")
    if PHONE_RE.search(full):
        issues.append("numéro de téléphone détecté (RGPD)")

    # 4. Publicité mensongère / clickbait.
    if CLICKBAIT_RE.search(full):
        issues.append("accroche trompeuse (clickbait)")
    if OVERPROMISE_RE.search(full):
        issues.append("promesse non vérifiable")

    # 5. Divulgation de l'affiliation si des liens affiliés sont présents.
    has_affiliate_link = bool(settings.affiliate_tag) and settings.affiliate_tag in full
    if has_affiliate_link and "affilié" not in full.lower():
        issues.append("lien affilié non divulgué")

    # Le score de conformité est binaire dans les faits : conforme ou non.
    score = 1.0 if not issues else max(0.0, 1.0 - 0.34 * len(issues))
    return {
        "compliance": round(score, 4),
        "compliant": not issues,
        "issues": issues,
        "has_disclaimer": has_disclaimer,
        "has_sources": has_sources,
    }


# ---------------------------------------------------------------------------
# Évaluateur
# ---------------------------------------------------------------------------


class Evaluator:
    """Calcule la fitness d'un contenu produit par un agent."""

    def __init__(self, llm: Any = None, use_llm_for_titles: bool = True) -> None:
        self.llm = llm
        self.use_llm_for_titles = use_llm_for_titles

    def _llm_title_score(self, title: str) -> Optional[float]:
        """Note de titre par le modèle léger (`llama-3.1-8b-instant`). `None` si HS."""
        if not self.use_llm_for_titles or self.llm is None:
            return None
        if not getattr(self.llm, "available", False):
            return None

        raw = self.llm.complete(
            system_prompt=(
                "Tu notes le potentiel de clic HONNÊTE d'un titre d'article de veille "
                "sur une échelle de 0 à 10. Un titre informatif, précis et spécifique "
                "obtient une note haute. Un titre vague, ou au contraire racoleur et "
                "exagéré, obtient une note basse. Réponds UNIQUEMENT par un nombre."
            ),
            user_prompt=f"Titre : {title}",
            temperature=0.0,
            max_tokens=8,
        )
        if not raw:
            return None
        match = re.search(r"\d+(?:[.,]\d+)?", raw)
        if not match:
            return None
        try:
            value = float(match.group().replace(",", "."))
        except ValueError:
            return None
        return max(0.0, min(1.0, value / 10.0))

    def evaluate(
        self,
        title: str,
        body: str,
        source_items: Sequence[Dict[str, Any]],
        agent_keywords: Sequence[str] = (),
        document: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retourne le détail des signaux et la fitness agrégée [0,1].

        Deux textes distincts sont analysés, et c'est important :
          * `document` (document final assemblé, mentions légales et sources
            incluses) sert au **contrôle de conformité** — c'est ce qui sera
            réellement publié ;
          * `body` (rédaction brute de l'agent) sert aux signaux de **qualité**.
            La bibliographie citerait sinon les titres sources mot pour mot et
            ferait chuter à tort le score d'originalité.

        Ne lève jamais d'exception : un contenu vide obtient simplement 0.
        """
        if not (title or "").strip() or not (body or "").strip():
            # Un agent qui ne produit rien ne peut pas être sélectionné.
            return {
                "fitness": 0.0,
                "compliance": 0.0,
                "readability": 0.0,
                "keyword_score": 0.0,
                "ctr_score": 0.0,
                "originality": 0.0,
                "compliant": False,
                "issues": ["contenu vide"],
                "word_count": 0,
            }

        try:
            document = document if document is not None else body
            compliance = compliance_report(document, title)
            readability = readability_score(body)
            keywords = keyword_score(body, agent_keywords, source_items)
            originality = originality_score(body, source_items)

            heuristic_ctr = ctr_heuristic(title)
            llm_ctr = self._llm_title_score(title)
            # Le modèle affine l'heuristique, il ne la remplace pas : en cas
            # d'indisponibilité, la note reste stable et comparable.
            ctr = heuristic_ctr if llm_ctr is None else 0.5 * heuristic_ctr + 0.5 * llm_ctr

            breakdown = {
                "compliance": compliance["compliance"],
                "readability": round(readability, 4),
                "keyword_score": round(keywords, 4),
                "ctr_score": round(ctr, 4),
                "originality": round(originality, 4),
            }
            fitness = sum(WEIGHTS[key] * value for key, value in breakdown.items())

            # Verrou légal : un contenu non conforme ne peut pas être sélectionné.
            if not compliance["compliant"]:
                fitness = min(fitness, NON_COMPLIANT_CAP)
                logger.warning(
                    "Contenu non conforme (%s) — fitness plafonnée à %.2f.",
                    ", ".join(compliance["issues"]),
                    NON_COMPLIANT_CAP,
                )

            breakdown.update(
                {
                    "fitness": round(max(0.0, min(1.0, fitness)), 4),
                    "compliant": compliance["compliant"],
                    "issues": compliance["issues"],
                    "ctr_heuristic": round(heuristic_ctr, 4),
                    "ctr_llm": None if llm_ctr is None else round(llm_ctr, 4),
                    "word_count": len(tokenize(body)),
                }
            )
            return breakdown

        except Exception as exc:  # pragma: no cover - filet de sécurité global
            logger.error("Évaluation impossible (%s) — fitness nulle.", exc)
            return {
                "fitness": 0.0,
                "compliance": 0.0,
                "readability": 0.0,
                "keyword_score": 0.0,
                "ctr_score": 0.0,
                "originality": 0.0,
                "compliant": False,
                "issues": [f"erreur d'évaluation : {type(exc).__name__}"],
            }


__all__ = [
    "Evaluator",
    "WEIGHTS",
    "compliance_report",
    "readability_score",
    "keyword_score",
    "ctr_heuristic",
    "originality_score",
]
