"""Persistance SQLite de l'évolution (générations, agents, scores, contenus).

Choix techniques :
  * `sqlite3` (bibliothèque standard) — zéro dépendance, zéro coût, un seul
    fichier versionnable dans le dépôt Git.
  * Le mode journal reste `DELETE` (défaut) et non `WAL` : on veut un fichier
    unique et cohérent pour le `git commit` automatique du workflow.

⚠️ **RGPD** : ce schéma ne contient **aucune donnée personnelle**. On ne stocke
que des métadonnées de contenus publics (titre, URL, résumé, source) et des
paramètres d'agents. Aucun champ email/identité/IP n'existe et il ne doit pas
en être ajouté.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    generation    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'running',
    items_ingested INTEGER NOT NULL DEFAULT 0,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    generation    INTEGER NOT NULL,
    system_prompt TEXT NOT NULL,
    tone          TEXT NOT NULL,
    temperature   REAL NOT NULL,
    keywords      TEXT NOT NULL,          -- JSON list
    mutation_rate REAL NOT NULL,
    fitness       REAL NOT NULL DEFAULT 0.0,
    parent_a      TEXT,
    parent_b      TEXT,
    origin        TEXT NOT NULL DEFAULT 'seed',  -- seed | elite | crossover | mutation
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    generation    INTEGER NOT NULL,
    fitness       REAL NOT NULL,
    readability   REAL NOT NULL DEFAULT 0.0,
    keyword_score REAL NOT NULL DEFAULT 0.0,
    ctr_score     REAL NOT NULL DEFAULT 0.0,
    compliance    REAL NOT NULL DEFAULT 0.0,
    originality   REAL NOT NULL DEFAULT 0.0,
    details       TEXT,                    -- JSON
    created_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS contents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    generation    INTEGER NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL,
    slug          TEXT NOT NULL,
    fitness       REAL NOT NULL DEFAULT 0.0,
    published     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Métadonnées d'items publics ingérés : sert au dédoublonnage entre cycles.
CREATE TABLE IF NOT EXISTS ingested_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT NOT NULL UNIQUE,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    link          TEXT NOT NULL,
    summary       TEXT,
    published_at  TEXT,
    ingested_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_generation ON scores(generation);
CREATE INDEX IF NOT EXISTS idx_agents_generation ON agents(generation);
CREATE INDEX IF NOT EXISTS idx_contents_run ON contents(run_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(*parts: str) -> str:
    """Empreinte stable d'un item (dédoublonnage), sans donnée personnelle."""
    payload = "|".join(p.strip().lower() for p in parts if p)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class Database:
    """Gestionnaire SQLite. Crée le fichier et le schéma à la volée si absents."""

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- Infrastructure -----------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connexion transactionnelle : commit si succès, rollback sinon."""
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
        except sqlite3.Error as exc:  # pragma: no cover - disque en lecture seule…
            logger.error("Impossible d'initialiser la base %s : %s", self.path, exc)
            raise

    # -- Runs ---------------------------------------------------------------

    def start_run(self, generation: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, generation, status) VALUES (?, ?, 'running')",
                (_utcnow(), generation),
            )
            return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str = "success",
        items_ingested: int = 0,
        notes: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs
                      SET finished_at = ?, status = ?, items_ingested = ?, notes = ?
                    WHERE id = ?""",
                (_utcnow(), status, items_ingested, notes[:2000], run_id),
            )

    # -- Agents -------------------------------------------------------------

    def current_generation(self) -> int:
        """Numéro de la dernière génération enregistrée (0 si base vierge)."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(generation) AS g FROM agents").fetchone()
        return int(row["g"]) if row and row["g"] is not None else 0

    def save_agents(self, agents: List[Dict[str, Any]]) -> None:
        """Insère (ou met à jour) une population complète."""
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO agents
                       (id, name, generation, system_prompt, tone, temperature,
                        keywords, mutation_rate, fitness, parent_a, parent_b,
                        origin, created_at)
                   VALUES (:id, :name, :generation, :system_prompt, :tone, :temperature,
                           :keywords, :mutation_rate, :fitness, :parent_a, :parent_b,
                           :origin, :created_at)
                   ON CONFLICT(id) DO UPDATE SET
                       fitness = excluded.fitness,
                       system_prompt = excluded.system_prompt,
                       mutation_rate = excluded.mutation_rate""",
                [
                    {
                        "id": a["id"],
                        "name": a["name"],
                        "generation": a["generation"],
                        "system_prompt": a["system_prompt"],
                        "tone": a["tone"],
                        "temperature": a["temperature"],
                        "keywords": json.dumps(a.get("keywords", []), ensure_ascii=False),
                        "mutation_rate": a["mutation_rate"],
                        "fitness": a.get("fitness", 0.0),
                        "parent_a": a.get("parent_a"),
                        "parent_b": a.get("parent_b"),
                        "origin": a.get("origin", "seed"),
                        "created_at": a.get("created_at", _utcnow()),
                    }
                    for a in agents
                ],
            )

    def load_generation(self, generation: int) -> List[Dict[str, Any]]:
        """Recharge une génération d'agents (liste de dicts prêts pour `Agent.from_dict`)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE generation = ? ORDER BY fitness DESC",
                (generation,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            try:
                data["keywords"] = json.loads(data["keywords"])
            except (TypeError, ValueError):
                data["keywords"] = []
            result.append(data)
        return result

    def load_latest_generation(self) -> List[Dict[str, Any]]:
        """Dernière population connue ; liste vide si la base est neuve."""
        generation = self.current_generation()
        return self.load_generation(generation) if generation else []

    # -- Scores & contenus --------------------------------------------------

    def save_score(
        self, run_id: int, agent_id: str, generation: int, breakdown: Dict[str, Any]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO scores
                       (run_id, agent_id, generation, fitness, readability,
                        keyword_score, ctr_score, compliance, originality,
                        details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    agent_id,
                    generation,
                    float(breakdown.get("fitness", 0.0)),
                    float(breakdown.get("readability", 0.0)),
                    float(breakdown.get("keyword_score", 0.0)),
                    float(breakdown.get("ctr_score", 0.0)),
                    float(breakdown.get("compliance", 0.0)),
                    float(breakdown.get("originality", 0.0)),
                    json.dumps(breakdown, ensure_ascii=False),
                    _utcnow(),
                ),
            )

    def save_content(
        self,
        run_id: int,
        agent_id: str,
        generation: int,
        title: str,
        body: str,
        slug: str,
        fitness: float,
        published: bool = False,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO contents
                       (run_id, agent_id, generation, title, body, slug,
                        fitness, published, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    agent_id,
                    generation,
                    title,
                    body,
                    slug,
                    float(fitness),
                    1 if published else 0,
                    _utcnow(),
                ),
            )
            return int(cur.lastrowid)

    # -- Items ingérés ------------------------------------------------------

    def filter_new_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retire les items déjà vus lors d'un cycle précédent (dédoublonnage)."""
        if not items:
            return []
        with self._connect() as conn:
            known = {
                row["fingerprint"]
                for row in conn.execute("SELECT fingerprint FROM ingested_items")
            }
        return [i for i in items if i.get("fingerprint") not in known]

    def record_items(self, items: List[Dict[str, Any]]) -> int:
        """Mémorise les items traités. Retourne le nombre effectivement inséré."""
        if not items:
            return 0
        with self._connect() as conn:
            cur = conn.executemany(
                """INSERT OR IGNORE INTO ingested_items
                       (fingerprint, source, title, link, summary, published_at, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item.get("fingerprint", fingerprint(item.get("link", ""))),
                        item.get("source", "")[:200],
                        item.get("title", "")[:500],
                        item.get("link", "")[:1000],
                        (item.get("summary") or "")[:2000],
                        item.get("published_at"),
                        _utcnow(),
                    )
                    for item in items
                ],
            )
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def prune_items(self, keep_last: int = 2000) -> None:
        """Borne la taille de la base (le dépôt Git reste léger sur le long terme)."""
        with self._connect() as conn:
            conn.execute(
                """DELETE FROM ingested_items
                    WHERE id NOT IN (
                        SELECT id FROM ingested_items ORDER BY id DESC LIMIT ?
                    )""",
                (keep_last,),
            )

    # -- Reporting ----------------------------------------------------------

    def evolution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fitness moyenne et maximale par génération (suivi de la progression)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT generation,
                          COUNT(*)      AS population,
                          AVG(fitness)  AS avg_fitness,
                          MAX(fitness)  AS best_fitness
                     FROM scores
                    GROUP BY generation
                    ORDER BY generation DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def best_fitness_for_generation(self, generation: int) -> Optional[float]:
        """Meilleure fitness observée pour une génération. `None` si jamais évaluée.

        Sert à l'auto-adaptation du taux de mutation : c'est la mémoire qui
        permet au moteur de savoir s'il progresse ou s'il stagne d'un cycle à
        l'autre, alors que chaque exécution démarre dans un processus neuf.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(fitness) AS best FROM scores WHERE generation = ?",
                (generation,),
            ).fetchone()
        return float(row["best"]) if row and row["best"] is not None else None

    def best_agent_ever(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agents ORDER BY fitness DESC, generation DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # -- Export pour le tableau de bord -------------------------------------

    def scores_for_run(self, run_id: int) -> List[Dict[str, Any]]:
        """Détail des cinq signaux par agent pour un cycle donné."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.agent_id, s.fitness, s.readability, s.keyword_score,
                          s.ctr_score, s.compliance, s.originality,
                          a.name, a.tone, a.origin, a.temperature, a.mutation_rate
                     FROM scores s
                     JOIN agents a ON a.id = s.agent_id
                    WHERE s.run_id = ?
                    ORDER BY s.fitness DESC""",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_runs(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Derniers cycles exécutés, du plus récent au plus ancien."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, started_at, finished_at, generation, status,
                          items_ingested, notes
                     FROM runs
                    ORDER BY id DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_contents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Dernières publications, avec l'agent qui les a produites."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.title, c.slug, c.fitness, c.generation, c.created_at,
                          a.name AS agent_name, a.tone
                     FROM contents c
                     LEFT JOIN agents a ON a.id = c.agent_id
                    WHERE c.published = 1
                    ORDER BY c.id DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def generation_stats(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Série temporelle de la fitness par génération, du plus ancien au récent.

        Ordre chronologique croissant : c'est ce qu'attend un graphe de tendance,
        et l'inverser côté client serait une source d'erreur silencieuse.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT generation,
                          COUNT(*)     AS population,
                          AVG(fitness) AS avg_fitness,
                          MAX(fitness) AS best_fitness,
                          MIN(fitness) AS worst_fitness
                     FROM scores
                    GROUP BY generation
                    ORDER BY generation DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
