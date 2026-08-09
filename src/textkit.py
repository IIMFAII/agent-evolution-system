"""Primitives d'analyse de texte — **bibliothèque standard uniquement**.

Ce module rassemble ce que partagent l'évaluateur interne du pipeline et le
produit d'audit vendable : découpage en mots, lisibilité française, détection
d'accroches trompeuses et de données personnelles, analyse de titre.

Pourquoi il existe séparément
-----------------------------
`content_audit` est destiné à tourner seul, dans un conteneur minimal, sans la
configuration du pipeline de veille. Or il dépendait de `evaluator`, qui dépend
de `config`, qui dépend de pydantic et de python-dotenv. Le produit traînait
donc deux dépendances tierces dont il n'a aucun usage — et la première tentative
d'emballage a effectivement buté sur un conflit de versions.

Extraire les primitives ici règle le problème sans dupliquer une ligne :
`evaluator` et `content_audit` importent tous deux depuis ce module, qui
n'importe que `re` et `collections`. Moins de dépendances signifie un
conteneur plus petit, un build plus rapide, moins de modes de panne, et un
coût d'exécution plus bas — ce qui compte quand on facture à l'appel.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Découpage
# ---------------------------------------------------------------------------

#: Mots-outils français et anglais, exclus de l'analyse thématique.
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "en", "au", "aux",
    "pour", "par", "sur", "avec", "dans", "que", "qui", "est", "sont", "son",
    "ses", "ce", "cette", "ces", "plus", "moins", "tout", "tous", "leur", "il",
    "elle", "on", "nous", "vous", "ils", "elles", "ne", "pas", "the", "a", "an",
    "of", "to", "in", "for", "on", "with", "and", "or", "is", "are", "as", "at",
    "by", "from", "it", "its", "this", "that", "be", "has", "have", "new",
}

_WORD_RE = re.compile(r"[a-zà-öø-ÿ0-9][a-zà-öø-ÿ0-9'-]{2,}", re.IGNORECASE)
_WORD_COUNT_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouyàâäéèêëîïôöùûüœ]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Découpe un texte en mots significatifs (minuscules, sans mots-outils)."""
    return [
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if w.lower() not in STOPWORDS and len(w) > 2
    ]


def count_words(text: str) -> int:
    """Nombre de mots **bruts**, mots-outils compris.

    À ne pas confondre avec `len(tokenize(...))` : le second retire les
    mots-outils et les mots courts, et vaut environ la moitié du premier sur du
    français courant. Appliquer un seuil de « contenu mince » aux jetons filtrés
    déclare mince un article de longueur normale.
    """
    return len(_WORD_COUNT_RE.findall(text or ""))


def extract_themes(texts: Sequence[str], top_n: int = 6) -> List[str]:
    """Mots-clés dominants d'un ensemble de textes.

    Le classement combine **récurrence** et **spécificité** : un mot présent
    dans plusieurs textes l'emporte, et à fréquence égale on préfère les termes
    longs, plus porteurs de sens que les mots courts et génériques.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(w for w in tokenize(text) if len(w) >= 4)
    ranked = sorted(counter.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return [word for word, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Lisibilité
# ---------------------------------------------------------------------------


def count_syllables_fr(word: str) -> int:
    """Compte approximatif de syllabes en français (groupes de voyelles).

    Le « e » muet final n'est pas compté, sauf s'il constitue la seule syllabe.
    """
    word = word.lower().strip()
    if not word:
        return 0
    count = len(_VOWEL_GROUP_RE.findall(word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_score(text: str) -> float:
    """Lisibilité normalisée [0,1] via l'indice de Kandel & Moles (Flesch FR).

        207 − 1.015 · (mots/phrases) − 73.6 · (syllabes/mots)

    L'échelle brute (0 = très difficile, 100 = très facile) est recentrée sur la
    bande 40–80, qui correspond à un article bien calibré : trop simple est aussi
    pénalisant que trop absconé.
    """
    words = re.findall(r"[\wà-öø-ÿ'-]+", text or "")
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    if len(words) < 20 or not sentences:
        return 0.0

    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = sum(count_syllables_fr(w) for w in words) / len(words)
    raw = max(0.0, min(100.0, 207 - 1.015 * words_per_sentence - 73.6 * syllables_per_word))

    if 40 <= raw <= 80:
        return 1.0
    if raw < 40:
        return max(0.0, raw / 40)
    return max(0.0, 1.0 - (raw - 80) / 20)


# ---------------------------------------------------------------------------
# Détecteurs de non-conformité
# ---------------------------------------------------------------------------

#: Données personnelles (RGPD) — leur présence invalide un contenu publiable.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}")

#: Accroches trompeuses (publicité mensongère / clickbait abusif).
CLICKBAIT_RE = re.compile(
    r"\b("
    r"incroyable|choquant|hallucinant|vous\s+n['e]\s*allez\s+pas\s+y\s+croire"
    r"|astuce\s+secrète|le\s+secret\s+que|100\s*%\s+garanti|argent\s+facile"
    r"|miracle|révolutionnaire|jamais\s+vu|urgent\s*!|dernière\s+chance"
    r"|gagnez\s+de\s+l'argent|sans\s+effort"
    r")\b",
    re.IGNORECASE,
)

#: Promesses non vérifiables.
OVERPROMISE_RE = re.compile(
    r"\b(garanti|assuré à 100|infaillible|sans risque|résultats? garantis?)\b",
    re.IGNORECASE,
)


def ctr_heuristic(title: str) -> float:
    """Estimation heuristique du potentiel de clic d'un titre, sans réseau.

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

    if re.search(r"\d", title):
        score += 0.10
    if len([w for w in tokenize(title) if len(w) > 4]) >= 4:
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


def ngrams(tokens: Sequence[str], n: int) -> set:
    """Ensemble des n-grammes d'une suite de jetons."""
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Similarité de Jaccard, 0 si l'un des ensembles est vide."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


__all__ = [
    "STOPWORDS",
    "tokenize",
    "count_words",
    "extract_themes",
    "count_syllables_fr",
    "readability_score",
    "ctr_heuristic",
    "ngrams",
    "jaccard",
    "EMAIL_RE",
    "PHONE_RE",
    "CLICKBAIT_RE",
    "OVERPROMISE_RE",
]
