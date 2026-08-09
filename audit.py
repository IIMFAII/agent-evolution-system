"""Interface en ligne de commande de l'audit de conformité.

Contrat volontairement minimal — **JSON en entrée, JSON en sortie** — parce que
c'est ce que toutes les places de marché de développeurs savent envelopper :
un conteneur, une entrée standard, une sortie standard. Aucun réseau, aucune
clé d'API, aucun coût variable : l'exécution est déterministe et facturable
telle quelle.

Usages :

    # Auditer un dossier de fichiers Markdown ou texte
    python audit.py --dir docs/posts

    # Auditer un lot fourni sur l'entrée standard
    echo '{"documents":[{"title":"T","text":"..."}]}' | python audit.py

    # Sortie lisible par un humain plutôt que JSON
    python audit.py --dir docs/posts --format text

Codes de sortie : 0 si le risque est faible ou modéré, 2 s'il est élevé ou
critique — de quoi bloquer une chaîne d'intégration continue avant publication.
"""

from __future__ import annotations

import argparse
import html
import re
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.content_audit import audit_corpus

#: Extensions considérées comme du contenu textuel auditable.
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".html"}

#: Risques qui font échouer la commande (utile en intégration continue).
FAILING_RISKS = {"élevé", "critique"}


#: Ordre de préférence quand un même contenu existe en plusieurs formats.
#: Sans cette déduplication, un dossier contenant `article.md` ET `article.html`
#: audite deux fois le même texte et gonfle artificiellement la similarité du
#: corpus — le tout premier essai de cet outil est tombé dans le piège.
SUFFIX_PRIORITY = {".md": 0, ".markdown": 1, ".txt": 2, ".html": 3}

_TAG_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)


def _strip_html(text: str) -> str:
    """Retire le balisage : sans cela, les noms de balises polluent l'analyse."""
    return html.unescape(_TAG_RE.sub(" ", text))


def _load_from_dir(path: Path) -> List[Dict[str, Any]]:
    """Charge les documents d'un dossier, un seul format par contenu.

    Le titre vient du premier `# ` Markdown ou du premier `<h1>`.
    """
    best: Dict[Path, Path] = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES:
            continue
        key = file.with_suffix("")
        current = best.get(key)
        if current is None or SUFFIX_PRIORITY[file.suffix.lower()] < SUFFIX_PRIORITY[current.suffix.lower()]:
            best[key] = file

    documents: List[Dict[str, Any]] = []
    for file in sorted(best.values()):
        try:
            raw = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[ignoré] {file} : {exc}", file=sys.stderr)
            continue
        text = _strip_html(raw) if file.suffix.lower() == ".html" else raw

        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            match = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.IGNORECASE | re.DOTALL)
            title = _strip_html(match.group(1)).strip() if match else file.stem

        documents.append({"title": title, "text": text, "path": str(file)})
    return documents


def _render_text(result: Dict[str, Any]) -> str:
    """Rendu lisible : le même contenu que le JSON, pour un œil humain."""
    c = result.get("corpus", {})
    lines = [
        f"Documents audités      : {result['documents']}",
        f"Risque                 : {result['risk'].upper()} — {result.get('risk_label', '')}",
        f"  {result.get('risk_reason', '')}",
        "",
        "Corpus",
        f"  Similarité moyenne   : {c.get('avg_similarity', 0):.1%}",
        f"  Similarité maximale  : {c.get('max_similarity', 0):.1%}",
        f"  Documents sur gabarit: {c.get('template_ratio', 0):.1%}",
        f"  Uniformité de plan   : {c.get('structural_uniformity', 0):.1%}",
        f"  Documents minces     : {c.get('thin_ratio', 0):.1%}",
        f"  Sans mention IA      : {c.get('undisclosed_ratio', 0):.1%}",
    ]
    pair = result.get("most_similar_pair")
    if pair and pair.get("similarity", 0) > 0:
        lines += [
            "",
            f"Paire la plus proche ({pair['similarity']:.1%})",
            f"  A : {pair['a'][:70]}",
            f"  B : {pair['b'][:70]}",
        ]
    lines += ["", "Recommandations"]
    lines += [f"  - {r}" for r in result.get("recommendations", [])]

    flagged = [r for r in result.get("reports", []) if r["issues"]]
    if flagged:
        lines += ["", f"Documents à corriger ({len(flagged)})"]
        for r in flagged[:15]:
            lines.append(f"  · {(r['title'] or '(sans titre)')[:64]}")
            for issue in r["issues"]:
                lines.append(f"      {issue}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit de conformité d'un corpus de contenus (risque « contenu à l'échelle », "
        "recopie, transparence, RGPD, lisibilité)."
    )
    parser.add_argument("--dir", type=Path, help="Dossier de fichiers à auditer.")
    parser.add_argument("--input", type=Path, help="Fichier JSON d'entrée (sinon : entrée standard).")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--fail-on-risk",
        action="store_true",
        help="Retourne 2 si le risque est élevé ou critique (utile en CI).",
    )
    args = parser.parse_args(argv)

    sources: List[str] = []
    if args.dir:
        if not args.dir.is_dir():
            print(f"Dossier introuvable : {args.dir}", file=sys.stderr)
            return 1
        documents = _load_from_dir(args.dir)
    else:
        try:
            raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
            payload = json.loads(raw or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Entrée illisible : {exc}", file=sys.stderr)
            return 1
        documents = payload.get("documents") or []
        sources = payload.get("sources") or []

    if not documents:
        print("Aucun document à auditer.", file=sys.stderr)
        return 1

    result = audit_corpus(documents, sources=sources or None)

    if args.format == "text":
        print(_render_text(result))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.fail_on_risk and result["risk"] in FAILING_RISKS:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
