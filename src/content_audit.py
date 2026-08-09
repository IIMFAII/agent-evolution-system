"""Audit de conformité de contenu — le produit vendable extrait de ce système.

Pourquoi ce produit
-------------------
Depuis la mise à jour de mars 2026, les moteurs sanctionnent le « contenu à
l'échelle » : de nombreuses pages produites sans supervision éditoriale et sans
valeur ajoutée. Les équipes de contenu ont donc un besoin concret et daté :
savoir, **avant publication**, si leur corpus ressemble à une ferme de gabarits.

Les outils de lisibilité existants notent un texte isolé. Ils passent à côté du
signal qui compte réellement ici, qui est **inter-documents** : cinquante
articles individuellement corrects mais bâtis sur le même moule constituent
exactement le motif visé. C'est la différence apportée par ce module.

Ce qu'il mesure
---------------
Par document :
  * lisibilité française (Kandel & Moles) ;
  * recopie littérale depuis les sources fournies (n-grammes) ;
  * mention de transparence sur l'automatisation ;
  * accroches trompeuses et promesses non vérifiables ;
  * données personnelles (RGPD) ;
  * indices de contenu mince (volume, structure).

Par corpus :
  * **répétition inter-documents** : similarité par bandes de n-grammes ;
  * **uniformité structurelle** : même squelette de titres d'un texte à l'autre ;
  * un niveau de risque global, assorti de recommandations actionnables.

Ce module ne dépend d'aucun réseau et d'aucune clé d'API : il est entièrement
déterministe, ce qui le rend facturable à l'exécution sans coût variable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.textkit import (
    CLICKBAIT_RE,
    EMAIL_RE,
    OVERPROMISE_RE,
    PHONE_RE,
    count_words,
    ctr_heuristic,
    jaccard as _jaccard,
    ngrams as _shingles,
    readability_score,
    tokenize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seuils
# ---------------------------------------------------------------------------

#: En dessous, le texte est considéré comme mince quelle que soit sa qualité.
#: Compté sur les mots **bruts**, pas sur les jetons filtrés : `tokenize()`
#: retire les mots-outils et les mots courts, si bien qu'un article de 300 mots
#: n'en conserve que ~140. Appliquer le seuil aux jetons déclarait « mince »
#: un texte de longueur parfaitement normale — un faux positif, c'est-à-dire
#: le défaut qui décrédibilise le plus vite un outil d'audit payant.
THIN_CONTENT_WORDS = 150

#: Part de 5-grammes partagés au-delà de laquelle deux documents sont jugés
#: bâtis sur le même gabarit. Calibré pour ne pas déclencher sur un simple
#: champ lexical commun : deux articles du même domaine partagent du
#: vocabulaire, pas des séquences de cinq mots.
TEMPLATE_SIMILARITY = 0.18

#: Part de documents fortement similaires au-delà de laquelle le corpus entier
#: présente le motif « ferme de gabarits ».
CORPUS_TEMPLATE_RATIO = 0.35

#: Marqueurs d'une divulgation d'automatisation, insensibles à la casse.
DISCLOSURE_PATTERNS = re.compile(
    r"(automatis|intelligence artificielle|\bia\b|généré par|assisté par|"
    r"ai-generated|generated with)",
    re.IGNORECASE,
)

RISK_LABELS = {
    "faible": "Aucun signal préoccupant.",
    "modéré": "Signaux à corriger avant publication à grande échelle.",
    "élevé": "Profil proche de ce que les moteurs sanctionnent.",
    "critique": "Publication déconseillée en l'état.",
}


def _heading_skeleton(text: str) -> Tuple[str, ...]:
    """Squelette de titres d'un document (niveaux Markdown, sans le libellé).

    Deux articles qui partagent exactement la même ossature de sections sont un
    indice de gabarit, indépendamment des mots employés.
    """
    return tuple(
        f"h{len(line) - len(line.lstrip('#'))}"
        for line in text.splitlines()
        if line.strip().startswith("#")
    )


# ---------------------------------------------------------------------------
# Audit d'un document
# ---------------------------------------------------------------------------


def audit_document(
    text: str,
    title: str = "",
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Audite un document isolé. Ne lève jamais d'exception.

    `sources` contient les textes de référence servant à détecter la recopie
    littérale. Sans eux, le score de recopie n'est pas calculé plutôt que d'être
    inventé : un `None` explicite vaut mieux qu'un faux zéro rassurant.
    """
    text = text or ""
    tokens = tokenize(text)          # jetons filtrés : similarité, recopie
    words = count_words(text)        # mots bruts : volume réel du texte
    issues: List[str] = []

    # --- Recopie littérale --------------------------------------------------
    verbatim: Optional[float] = None
    if sources:
        source_tokens = tokenize(" ".join(sources))
        if len(source_tokens) >= 10 and len(tokens) >= 10:
            doc8, src8 = _shingles(tokens, 8), _shingles(source_tokens, 8)
            if doc8 & src8:
                verbatim = 1.0
                issues.append("recopie littérale détectée (séquence de 8 mots identique)")
            else:
                doc3, src3 = _shingles(tokens, 3), _shingles(source_tokens, 3)
                verbatim = round(len(doc3 & src3) / len(doc3), 4) if doc3 else 0.0
                if verbatim > 0.35:
                    issues.append(f"reformulation insuffisante ({verbatim:.0%} de trigrammes communs)")

    # --- Transparence -------------------------------------------------------
    disclosed = bool(DISCLOSURE_PATTERNS.search(text))
    if not disclosed:
        issues.append("aucune mention de production automatisée")

    # --- Loyauté ------------------------------------------------------------
    full = f"{title}\n{text}"
    if CLICKBAIT_RE.search(full):
        issues.append("accroche trompeuse")
    if OVERPROMISE_RE.search(full):
        issues.append("promesse non vérifiable")

    # --- RGPD ---------------------------------------------------------------
    personal: List[str] = []
    if EMAIL_RE.search(full):
        personal.append("adresse e-mail")
    if PHONE_RE.search(full):
        personal.append("numéro de téléphone")
    issues.extend(f"donnée personnelle exposée : {p}" for p in personal)

    # --- Contenu mince ------------------------------------------------------
    thin = words < THIN_CONTENT_WORDS
    if thin:
        issues.append(f"contenu mince ({words} mots, seuil {THIN_CONTENT_WORDS})")

    readability = readability_score(text)
    if readability < 0.4 and words >= 80:
        issues.append("lisibilité faible : phrases trop longues ou trop denses")

    return {
        "title": title,
        "words": words,
        "content_words": len(tokens),
        "readability": round(readability, 4),
        "title_quality": round(ctr_heuristic(title), 4) if title else None,
        "verbatim_overlap": verbatim,
        "disclosure_present": disclosed,
        "personal_data": personal,
        "thin_content": thin,
        "issues": issues,
        "publishable": not issues,
    }


# ---------------------------------------------------------------------------
# Audit d'un corpus — la valeur différenciante
# ---------------------------------------------------------------------------


def audit_corpus(
    documents: Sequence[Dict[str, Any]],
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Audite un lot et mesure la **répétition inter-documents**.

    C'est le signal que les outils par document manquent : cinquante textes
    individuellement corrects mais coulés dans le même moule constituent
    exactement le motif de « contenu à l'échelle » que les moteurs pénalisent.

    Chaque entrée de `documents` est un dict `{"text": ..., "title": ...}`.
    """
    docs = [d for d in documents if (d.get("text") or "").strip()]
    if not docs:
        return {
            "documents": 0,
            "risk": "faible",
            "risk_reason": "aucun document à auditer",
            "reports": [],
            "corpus": {},
            "recommendations": [],
        }

    reports = [
        audit_document(d.get("text", ""), d.get("title", ""), sources=d.get("sources") or sources)
        for d in docs
    ]

    # --- Similarité par paires ---------------------------------------------
    shingle_sets = [_shingles(tokenize(d.get("text", "")), 5) for d in docs]
    skeletons = [_heading_skeleton(d.get("text", "")) for d in docs]

    pairs: List[Tuple[int, int, float]] = []
    similar_docs: set = set()
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = _jaccard(shingle_sets[i], shingle_sets[j])
            pairs.append((i, j, sim))
            if sim >= TEMPLATE_SIMILARITY:
                similar_docs.update((i, j))

    similarities = [s for _, _, s in pairs]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    max_similarity = max(similarities) if similarities else 0.0
    template_ratio = len(similar_docs) / len(docs)

    # Uniformité structurelle : même ossature de titres d'un texte à l'autre.
    distinct_skeletons = len({s for s in skeletons if s})
    structural_uniformity = (
        1.0 - (distinct_skeletons / len(docs)) if any(skeletons) else 0.0
    )

    # --- Niveau de risque ---------------------------------------------------
    thin_ratio = sum(1 for r in reports if r["thin_content"]) / len(reports)
    undisclosed = sum(1 for r in reports if not r["disclosure_present"]) / len(reports)

    risk, reason = "faible", "Corpus varié, aucun motif de gabarit détecté."
    if template_ratio >= CORPUS_TEMPLATE_RATIO or structural_uniformity >= 0.8:
        risk = "élevé"
        reason = (
            f"{template_ratio:.0%} des documents partagent des séquences de 5 mots "
            f"au-delà du seuil de gabarit, et {structural_uniformity:.0%} d'uniformité "
            "structurelle : le corpus ressemble à une production sur moule."
        )
    elif avg_similarity >= TEMPLATE_SIMILARITY * 0.6 or thin_ratio >= 0.5:
        risk = "modéré"
        reason = (
            f"Similarité moyenne {avg_similarity:.0%} et {thin_ratio:.0%} de "
            "documents minces : la variété du corpus est insuffisante."
        )
    if template_ratio >= 0.7 and thin_ratio >= 0.5:
        risk = "critique"
        reason = (
            "Documents à la fois minces et bâtis sur le même gabarit : c'est le "
            "profil type sanctionné par les politiques anti-contenu à l'échelle."
        )

    # --- Recommandations actionnables --------------------------------------
    recommendations: List[str] = []
    if template_ratio >= CORPUS_TEMPLATE_RATIO:
        recommendations.append(
            "Diversifier les plans : varier l'ordre des sections et les formulations "
            "d'accroche plutôt que de dériver chaque texte du même squelette."
        )
    if thin_ratio >= 0.3:
        recommendations.append(
            f"Étoffer les documents sous {THIN_CONTENT_WORDS} mots utiles, ou les "
            "fusionner : un texte mince ne se rattrape pas par le volume de pages."
        )
    if undisclosed > 0:
        recommendations.append(
            f"Ajouter une mention de production automatisée sur {undisclosed:.0%} "
            "des documents qui en sont dépourvus."
        )
    if any(r["personal_data"] for r in reports):
        recommendations.append(
            "Retirer les données personnelles détectées avant toute publication (RGPD)."
        )
    if any(r["verbatim_overlap"] == 1.0 for r in reports):
        recommendations.append(
            "Réécrire les passages recopiés mot pour mot depuis les sources."
        )
    if not recommendations:
        recommendations.append("Aucune action requise : le corpus est publiable en l'état.")

    return {
        "documents": len(docs),
        "risk": risk,
        "risk_label": RISK_LABELS[risk],
        "risk_reason": reason,
        "corpus": {
            "avg_similarity": round(avg_similarity, 4),
            "max_similarity": round(max_similarity, 4),
            "template_ratio": round(template_ratio, 4),
            "structural_uniformity": round(structural_uniformity, 4),
            "thin_ratio": round(thin_ratio, 4),
            "undisclosed_ratio": round(undisclosed, 4),
            "distinct_structures": distinct_skeletons,
        },
        "most_similar_pair": (
            {
                "a": docs[max(pairs, key=lambda p: p[2])[0]].get("title", ""),
                "b": docs[max(pairs, key=lambda p: p[2])[1]].get("title", ""),
                "similarity": round(max_similarity, 4),
            }
            if pairs
            else None
        ),
        "recommendations": recommendations,
        "reports": reports,
    }


__all__ = [
    "audit_document",
    "audit_corpus",
    "THIN_CONTENT_WORDS",
    "TEMPLATE_SIMILARITY",
]
