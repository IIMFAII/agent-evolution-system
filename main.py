"""Point d'entrée : un cycle complet d'évolution (exécuté toutes les 2 h).

Séquence d'un cycle :

    1. Ingestion    — flux RSS publics, robots.txt respecté, dédoublonnage
    2. Amorçage     — rechargement de la dernière population depuis SQLite
    3. Production   — chaque agent rédige une synthèse (LLM ou repli local)
    4. Assemblage   — mentions légales et sources ajoutées par le code
    5. Évaluation   — fitness = conformité, lisibilité, pertinence, CTR, originalité
    6. Publication  — seul le meilleur contenu conforme est publié
    7. Évolution    — élitisme + croisement + mutation → génération suivante
    8. Persistance  — agents, scores et contenus écrits en base

**Contrat fail-safe** : aucune étape ne peut faire planter le pipeline.
Chaque défaillance externe (réseau, quota, flux mort) dégrade le cycle et est
tracée en base, mais le processus se termine proprement.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.agent import Agent, slugify
from src.agents.genetic_engine import GeneticEngine
from src.config import settings, setup_logging
from src.database import Database
from src.evaluator import WEIGHTS, Evaluator
from src.ingestor import Ingestor
from src.llm import get_llm_client
from src.publisher import Publisher

logger = setup_logging()

#: Fitness minimale exigée pour qu'un contenu soit publié.
PUBLISH_THRESHOLD = 0.45


def _dominant_theme(items: List[Dict[str, Any]]) -> str:
    """Thème dominant du cycle (sert de contexte au bloc d'affiliation)."""
    from src.agents.agent import extract_themes

    themes = extract_themes(items, top_n=2)
    return " ".join(themes) if themes else "veille technologique"


def _dashboard_payload(
    db: Database,
    cycle: Dict[str, Any],
    sources: Dict[str, str],
    agents: Optional[List[Dict[str, Any]]] = None,
    population: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble la source de données unique du tableau de bord (`docs/data.json`).

    Alimenté à chaque cycle, y compris ceux qui ne publient rien : le tableau de
    bord doit refléter l'activité réelle du système, pas seulement ses succès.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cycle": cycle,
        "population": population or {},
        "agents": agents or [],
        "generations": db.generation_stats(limit=60),
        "runs": db.recent_runs(limit=40),
        "publications": db.recent_contents(limit=20),
        "sources": sources,
        "weights": WEIGHTS,
        "signal_labels": {
            "compliance": "Conformité",
            "readability": "Lisibilité",
            "keyword_score": "Pertinence",
            "ctr_score": "Potentiel CTR",
            "originality": "Originalité",
        },
    }


def run_cycle(dry_run: bool = False, skip_ingest: bool = False) -> int:
    """Exécute un cycle complet. Retourne un code de sortie POSIX (0 = OK)."""
    db = Database(settings.db_file)
    generation = db.current_generation()
    run_id = db.start_run(generation)
    notes: List[str] = []
    sources: Dict[str, str] = {}

    # --- 1. Ingestion ------------------------------------------------------
    items: List[Dict[str, Any]] = []
    if not skip_ingest:
        ingestor = Ingestor()
        try:
            collected = ingestor.collect()
        except Exception as exc:  # pragma: no cover - filet global
            logger.error("Ingestion en échec (%s) — cycle poursuivi à vide.", exc)
            collected = []
            notes.append(f"ingestion: {type(exc).__name__}")
        sources = dict(ingestor.report)

        items = db.filter_new_items(collected)
        if collected and not items:
            logger.info("Tous les items collectés ont déjà été traités — rien de neuf.")

    if not items:
        # Pas de matière première : on s'arrête proprement, sans consommer de
        # quota LLM et sans publier de contenu vide.
        logger.info("Aucun nouvel item exploitable — cycle interrompu proprement.")
        db.finish_run(run_id, status="no_data", items_ingested=0, notes="; ".join(notes))
        if not dry_run:
            # Le tableau de bord est rafraîchi même sans publication, sinon il
            # laisserait croire que le système est à l'arrêt.
            Publisher().write_dashboard_data(
                _dashboard_payload(
                    db,
                    cycle={
                        "run_id": run_id,
                        "generation_evaluated": generation,
                        "items_ingested": 0,
                        "status": "no_data",
                        "publication": {"published": False, "reason": "aucun item nouveau"},
                    },
                    sources=sources,
                )
            )
        return 0

    logger.info("%d nouvel(s) item(s) à traiter sur ce cycle.", len(items))

    # --- 2. Amorçage de la population --------------------------------------
    llm = get_llm_client()
    if not llm.available:
        notes.append("llm indisponible (mode dégradé)")

    engine = GeneticEngine(llm=llm)
    population: List[Agent] = engine.bootstrap(db.load_latest_generation())
    current_generation = population[0].generation if population else 1
    # Persistés avant l'évaluation : les scores référencent les agents (clé
    # étrangère), donc la population doit exister en base au préalable.
    db.save_agents([a.to_dict() for a in population])

    # --- 3-5. Production, assemblage, évaluation ---------------------------
    publisher = Publisher()
    evaluator = Evaluator(llm=llm)
    theme = _dominant_theme(items)
    results: List[Dict[str, Any]] = []

    for agent in population:
        try:
            draft = agent.generate(items, llm=llm)
            document = publisher.assemble(
                title=draft["title"],
                body=draft["body"],
                source_items=items,
                agent_name=agent.name,
                generation=agent.generation,
                theme=theme,
            )
            breakdown = evaluator.evaluate(
                title=draft["title"],
                body=draft["body"],      # qualité rédactionnelle
                document=document,       # conformité du document publié
                source_items=items,
                agent_keywords=agent.keywords,
            )
        except Exception as exc:  # pragma: no cover - un agent défaillant ne bloque pas
            logger.error("Agent %s en échec (%s) — fitness nulle.", agent.name, exc)
            db.save_score(run_id, agent.id, agent.generation, {"fitness": 0.0})
            agent.fitness = 0.0
            continue

        agent.fitness = float(breakdown["fitness"])
        db.save_score(run_id, agent.id, agent.generation, breakdown)
        results.append({"agent": agent, "draft": draft, "document": document, "score": breakdown})

        logger.info(
            "%-14s fitness=%.3f (conformité %.2f | lisibilité %.2f | pertinence %.2f "
            "| CTR %.2f | originalité %.2f)%s",
            agent.name,
            agent.fitness,
            breakdown.get("compliance", 0.0),
            breakdown.get("readability", 0.0),
            breakdown.get("keyword_score", 0.0),
            breakdown.get("ctr_score", 0.0),
            breakdown.get("originality", 0.0),
            "" if draft["used_llm"] else "  [repli local]",
        )

    db.save_agents([a.to_dict() for a in population])  # mise à jour des fitness

    # --- 6. Publication du meilleur contenu conforme -----------------------
    published_info: Dict[str, Any] = {"published": False, "reason": "aucun contenu évalué"}
    eligible = [
        r
        for r in results
        if r["score"].get("compliant") and r["score"]["fitness"] >= PUBLISH_THRESHOLD
    ]
    best = max(eligible, key=lambda r: r["score"]["fitness"]) if eligible else None

    if best is None:
        reason = (
            "aucun contenu au-dessus du seuil de qualité/conformité"
            if results
            else "aucun contenu produit"
        )
        logger.warning("Publication ignorée : %s.", reason)
        published_info = {"published": False, "reason": reason}
        notes.append(reason)
    elif dry_run:
        logger.info("[dry-run] Contenu retenu : « %s » (non écrit sur disque).", best["draft"]["title"])
        published_info = {"published": False, "reason": "dry-run", "title": best["draft"]["title"]}
    else:
        agent = best["agent"]
        slug = slugify(best["draft"]["title"])
        paths = publisher.publish(best["draft"]["title"], best["document"], slug=slug)
        publisher.build_index()
        db.save_content(
            run_id=run_id,
            agent_id=agent.id,
            generation=agent.generation,
            title=best["draft"]["title"],
            body=best["document"],
            slug=paths["slug"],
            fitness=best["score"]["fitness"],
            published=True,
        )
        publisher.notify_webhook(
            title=best["draft"]["title"],
            summary=f"Fitness {best['score']['fitness']:.2f} — agent {agent.name} "
            f"(génération {agent.generation}).",
        )
        published_info = {
            "published": True,
            "title": best["draft"]["title"],
            "agent": agent.name,
            "fitness": best["score"]["fitness"],
            "path": paths["markdown"],
        }
        logger.info("Publié : %s (fitness %.3f)", paths["markdown"], best["score"]["fitness"])

    # --- 7. Évolution ------------------------------------------------------
    summary = GeneticEngine.summarize(population)
    # La meilleure fitness du cycle précédent dit au moteur s'il progresse ou
    # s'il stagne : c'est ce qui pilote l'auto-adaptation du taux de mutation.
    previous_best = db.best_fitness_for_generation(current_generation - 1)
    next_population = engine.evolve(population, previous_best=previous_best)
    db.save_agents([a.to_dict() for a in next_population])

    # --- 8. Persistance & rapport -----------------------------------------
    db.record_items(items)
    db.prune_items()

    # Le cycle est clos AVANT l'export : sinon le tableau de bord relit son
    # propre run encore marqué « running », et affiche en permanence un dernier
    # cycle « en cours » avec zéro item.
    db.finish_run(
        run_id,
        status="success",
        items_ingested=len(items),
        notes="; ".join(notes),
    )

    if not dry_run:
        publisher.write_dashboard_data(
            _dashboard_payload(
                db,
                cycle={
                    "run_id": run_id,
                    "generation_evaluated": current_generation,
                    "next_generation": next_population[0].generation
                    if next_population
                    else current_generation,
                    "items_ingested": len(items),
                    "llm_enabled": llm.available,
                    "llm_calls": getattr(llm, "calls_made", 0),
                    "status": "success",
                    "publication": published_info,
                },
                sources=sources,
                agents=db.scores_for_run(run_id),
                population=summary,
            )
        )
        publisher.write_status(
            {
                "run_id": run_id,
                "generation_evaluated": current_generation,
                "next_generation": next_population[0].generation if next_population else current_generation,
                "items_ingested": len(items),
                "llm_enabled": llm.available,
                "llm_calls": getattr(llm, "calls_made", 0),
                "population": summary,
                "publication": published_info,
                "history": db.evolution_history(limit=10),
            }
        )

    logger.info(
        "Cycle terminé — génération %s évaluée (meilleure fitness %.3f), "
        "génération %s prête.",
        current_generation,
        summary.get("best_fitness", 0.0),
        next_population[0].generation if next_population else current_generation,
    )
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Système autonome d'agents évolutifs (veille + synthèse légale)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exécute le cycle sans rien écrire dans le dossier de publication.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="N'interroge aucun flux réseau (utile pour un test hors-ligne).",
    )
    args = parser.parse_args(argv)

    try:
        return run_cycle(dry_run=args.dry_run, skip_ingest=args.skip_ingest)
    except KeyboardInterrupt:
        logger.warning("Interruption manuelle.")
        return 130
    except Exception:
        # Dernier filet : on trace la pile complète et on sort en erreur, mais
        # sans jamais relancer le cycle (aucune boucle de réessai).
        logger.error("Erreur fatale non rattrapée :\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
