"""Point d'entrée de l'Actor Apify : audit de conformité d'un corpus.

Enveloppe mince autour de `src.content_audit`. Toute la logique métier vit dans
le module, testé hors-ligne par la suite de tests du dépôt ; ce fichier ne fait
que traduire le protocole d'entrée/sortie de la plateforme.

Propriété importante, et argument de vente : **cet Actor n'ouvre aucune
connexion réseau**. Il ne visite aucun site, n'extrait rien, ne manipule aucun
identifiant. Il reçoit du texte, calcule, renvoie du JSON. Il ne peut donc
enfreindre les conditions d'aucun tiers, et son coût de calcul reste minime.
"""

from __future__ import annotations

import sys
from pathlib import Path

from apify import Actor

# Le code métier vit à la racine du dépôt, pas dans .actor/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.content_audit import audit_corpus  # noqa: E402


async def main() -> None:
    async with Actor:
        payload = await Actor.get_input() or {}

        documents = payload.get("documents") or []
        sources = payload.get("sources") or None

        if not documents:
            # Échec explicite plutôt que résultat vide : un utilisateur qui paie
            # doit savoir que son entrée était mal formée, pas croire que son
            # corpus est irréprochable.
            await Actor.fail(
                status_message=(
                    "Aucun document fourni. Attendu : "
                    '{"documents": [{"title": "...", "text": "..."}]}'
                )
            )
            return

        Actor.log.info("Audit de %d document(s).", len(documents))
        result = audit_corpus(documents, sources=sources)

        # Une ligne par document dans le jeu de données : c'est le format que
        # les utilisateurs exportent en CSV pour trier leurs corrections.
        for report in result["reports"]:
            await Actor.push_data(report)

        # La synthèse de corpus — la vraie valeur ajoutée — est stockée à part,
        # car elle porte sur l'ensemble et non sur une ligne.
        summary = {k: v for k, v in result.items() if k != "reports"}
        await Actor.set_value("SUMMARY", summary)

        Actor.log.info(
            "Risque %s — %s", result["risk"].upper(), result["risk_reason"]
        )
        await Actor.set_status_message(
            f"Risque {result['risk']} sur {result['documents']} document(s)."
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
