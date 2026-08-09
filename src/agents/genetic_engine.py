"""Moteur d'évolution : sélection, élitisme, croisement et mutation.

Boucle génétique appliquée à chaque cycle (toutes les 2 h) :

    population(G) --évaluation--> fitness
                  --élitisme----> les meilleurs passent en G+1 intacts
                  --sélection---> tournoi binaire sur le reste
                  --croisement--> fusion des ADN de 2 parents
                  --mutation----> réécriture du prompt (Groq) ou variation locale
                  ==> population(G+1)

Garde-fous :
  * **Fail-safe LLM** : toute mutation LLM échouée retombe sur une mutation
    locale déterministe. L'évolution continue même sans réseau.
  * **Validation de génome** : un prompt muté est rejeté s'il est vide, trop
    court, ou s'il contient une consigne interdite (copie littérale, collecte
    d'e-mails, contournement de règles…).
  * **Anti-dégénérescence** : si la diversité génétique s'effondre, le taux de
    mutation augmente automatiquement.
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence

from src.agents.agent import Agent, LEGAL_CHARTER, create_seed_population
from src.config import settings

logger = logging.getLogger(__name__)


#: Motifs interdits dans un ADN muté (protection contre la dérive du LLM).
FORBIDDEN_PATTERNS = re.compile(
    r"\b("
    r"copie(?:r|z)?\s+(?:mot\s+à\s+mot|littéralement|à\s+l'identique)"
    r"|copier[- ]coller"
    r"|verbatim"
    r"|adresse[s]?\s+e-?mail"
    r"|collecte[rz]?\s+des\s+(?:données|emails|e-mails|contacts)"
    r"|scrap(?:e|er|ing)"
    r"|ignore[rz]?\s+les\s+(?:règles|consignes|instructions)"
    r"|spam"
    r"|clickbait"
    r"|invente[rz]?\s+(?:des\s+)?(?:faits|chiffres|statistiques)"
    r")\b",
    re.IGNORECASE,
)

#: Fragments d'ADN utilisés par la mutation locale (mode hors-ligne).
LOCAL_MUTATION_FRAGMENTS = [
    "Tu ouvres systématiquement par le constat le plus concret.",
    "Tu privilégies les phrases de moins de 20 mots.",
    "Tu illustres chaque idée abstraite par un cas d'usage réel.",
    "Tu signales explicitement le niveau de certitude de chaque affirmation.",
    "Tu termines chaque section par une question ouverte utile au lecteur.",
    "Tu compares systématiquement l'information à l'état de l'art antérieur.",
    "Tu hiérarchises les informations de la plus structurante à la plus anecdotique.",
    "Tu évites tout jargon non défini dans le paragraphe qui l'introduit.",
]

#: Vocabulaire de spécialisation injectable par mutation.
KEYWORD_POOL = [
    "sécurité", "open source", "cloud", "performance", "coût", "adoption",
    "réglementation", "écosystème", "productivité", "infrastructure",
    "interopérabilité", "souveraineté", "maintenance", "accessibilité",
]

#: Ton possible d'un agent (le croisement peut en hériter).
TONE_POOL = ["analytique", "pédagogique", "concis", "critique", "prospectif"]


class GeneticEngine:
    """Fait évoluer une population d'agents d'une génération à la suivante."""

    def __init__(
        self,
        llm: Any = None,
        population_size: Optional[int] = None,
        elite_size: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.llm = llm
        self.population_size = population_size or settings.population_size
        self.elite_size = max(1, min(elite_size or settings.elite_size, self.population_size - 1))
        self.rng = rng or random.Random()

    # -- Amorçage -----------------------------------------------------------

    def bootstrap(self, stored: Sequence[Dict[str, Any]]) -> List[Agent]:
        """Reconstruit la population depuis la base, ou la crée si base vierge.

        Ajuste aussi la taille si `POPULATION_SIZE` a changé entre deux cycles.
        """
        if not stored:
            logger.info("Base vierge — création de la population initiale (génération 1).")
            return create_seed_population(self.population_size, self.rng)

        population = [Agent.from_dict(row) for row in stored]

        if len(population) > self.population_size:
            population.sort(key=lambda a: a.fitness, reverse=True)
            population = population[: self.population_size]
        elif len(population) < self.population_size:
            missing = self.population_size - len(population)
            generation = population[0].generation if population else 1
            extra = create_seed_population(missing, self.rng)
            for agent in extra:
                agent.generation = generation
            population.extend(extra)
            logger.info("Population complétée avec %d nouvel(s) agent(s).", missing)

        return population

    # -- Sélection ----------------------------------------------------------

    def _tournament(self, population: Sequence[Agent], k: int = 2) -> Agent:
        """Sélection par tournoi : k candidats tirés au sort, le meilleur gagne."""
        contenders = self.rng.sample(list(population), min(k, len(population)))
        return max(contenders, key=lambda a: a.fitness)

    @staticmethod
    def _diversity(population: Sequence[Agent]) -> float:
        """Part d'ADN distincts dans la population (1.0 = diversité maximale)."""
        if not population:
            return 1.0
        return len({a.dna_hash for a in population}) / len(population)

    @staticmethod
    def _rate_drift(current_best: float, previous_best: Optional[float]) -> float:
        """Ajustement du taux de mutation, décidé au niveau de la population.

        Inspiré de la règle du 1/5 de Rechenberg : **on explore quand ça
        stagne, on exploite quand ça progresse**.

        Le signal doit être global, pas individuel. Un critère par agent — du
        type « cet agent fait-il mieux que la moyenne ? » — s'effondre toujours
        vers une borne : les parents issus d'un tournoi sont par construction
        au-dessus de la moyenne, et le taux de mutation collapse au minimum,
        génération après génération. La progression de la population entre deux
        cycles, elle, est un signal non biaisé.
        """
        if previous_best is None:
            return 0.0  # premier cycle : aucun historique, aucun ajustement
        if current_best > previous_best + 0.01:
            return -0.03  # ça progresse : on affine autour de ce qui marche
        if current_best < previous_best - 0.01:
            return 0.05  # régression : il faut réexplorer plus franchement
        return 0.02  # plateau : exploration progressive pour en sortir

    # -- Croisement ---------------------------------------------------------

    def crossover(self, parent_a: Agent, parent_b: Agent, generation: int) -> Agent:
        """Fusionne deux ADN : phrases alternées + moyenne des paramètres.

        On découpe chaque prompt parent en phrases et on en reprend une moitié
        de chacun : le descendant hérite d'une structure de raisonnement mixte.
        """
        sentences_a = [s.strip() for s in re.split(r"(?<=[.!?])\s+", parent_a.system_prompt) if s.strip()]
        sentences_b = [s.strip() for s in re.split(r"(?<=[.!?])\s+", parent_b.system_prompt) if s.strip()]

        cut_a = max(1, len(sentences_a) // 2)
        cut_b = max(1, len(sentences_b) // 2)
        merged = sentences_a[:cut_a] + sentences_b[cut_b:]
        child_prompt = " ".join(merged).strip() or parent_a.system_prompt

        # Mélange des mots-clés des deux parents, sans doublon.
        keywords = list(dict.fromkeys(parent_a.keywords + parent_b.keywords))
        self.rng.shuffle(keywords)

        return Agent(
            name=f"{parent_a.name[:6]}x{parent_b.name[:6]}",
            generation=generation,
            system_prompt=child_prompt,
            tone=self.rng.choice([parent_a.tone, parent_b.tone]),
            temperature=(parent_a.temperature + parent_b.temperature) / 2,
            keywords=keywords[:8],
            mutation_rate=(parent_a.mutation_rate + parent_b.mutation_rate) / 2,
            parent_a=parent_a.id,
            parent_b=parent_b.id,
            origin="crossover",
        )

    # -- Mutation -----------------------------------------------------------

    def mutate(
        self,
        agent: Agent,
        pressure: float = 1.0,
        reference_fitness: Optional[float] = None,
        rate_drift: float = 0.0,
    ) -> Agent:
        """Applique une mutation à l'ADN d'un agent.

        `pressure` (>1) amplifie ponctuellement le taux quand la diversité chute.
        `rate_drift` ajuste durablement le taux de mutation héritable ; il est
        calculé au niveau de la population par `evolve()` (voir `_rate_drift`).
        `reference_fitness` n'est utilisé que pour informer le LLM du niveau de
        la lignée : un descendant n'a pas encore de score propre.

        La mutation LLM est tentée en premier ; en cas d'indisponibilité ou de
        génome invalide, on retombe sur la mutation locale.
        """
        rate = min(0.95, agent.mutation_rate * pressure)
        reference = agent.fitness if reference_fitness is None else reference_fitness

        # --- Paramètres numériques (toujours mutés, c'est gratuit) ---------
        if self.rng.random() < rate:
            agent.temperature = max(0.1, min(1.2, agent.temperature + self.rng.uniform(-0.15, 0.15)))
        if self.rng.random() < rate:
            agent.tone = self.rng.choice(TONE_POOL)
        if self.rng.random() < rate:
            pool = [k for k in KEYWORD_POOL if k not in agent.keywords]
            if pool:
                agent.keywords = (agent.keywords + [self.rng.choice(pool)])[-8:]
        if self.rng.random() < rate and len(agent.keywords) > 2:
            agent.keywords.pop(self.rng.randrange(len(agent.keywords)))

        # --- ADN textuel ---------------------------------------------------
        if self.rng.random() < rate:
            mutated = self._llm_mutate(agent, reference) or self._local_mutate(agent)
            if mutated:
                agent.system_prompt = mutated
                agent.origin = "mutation" if agent.origin == "elite" else agent.origin

        # --- Auto-adaptation du taux de mutation ---------------------------
        agent.mutation_rate = max(0.05, min(0.9, agent.mutation_rate + rate_drift))
        return agent

    def _llm_mutate(self, agent: Agent, reference_fitness: Optional[float] = None) -> Optional[str]:
        """Demande au modèle une variante du prompt système. `None` si indisponible."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return None

        instruction = (
            "Tu es un ingénieur en optimisation de prompts. On te donne le prompt "
            "système d'un agent rédacteur de veille. Produis UNE variante améliorée.\n"
            "Contraintes :\n"
            "- Garde le même métier (rédaction de synthèses de veille en français).\n"
            "- Change l'angle éditorial, la structure de raisonnement OU le niveau "
            "de détail — pas les trois à la fois.\n"
            "- 2 à 4 phrases, 400 caractères maximum.\n"
            "- N'écris QUE le nouveau prompt, sans guillemets ni commentaire.\n"
            "- N'introduis jamais de consigne illégale ou contraire à l'éthique "
            "(copie littérale, collecte de données personnelles, appâts à clic).\n"
            f"{LEGAL_CHARTER}"
        )
        score = agent.fitness if reference_fitness is None else reference_fitness
        user = (
            f"Prompt actuel (score de fitness {score:.2f}/1.00, "
            f"ton « {agent.tone} ») :\n\"\"\"\n{agent.system_prompt}\n\"\"\""
        )
        candidate = self.llm.complete(
            system_prompt=instruction,
            user_prompt=user,
            temperature=0.9,
            max_tokens=250,
        )
        cleaned = self._sanitize_genome(candidate)
        if cleaned is None and candidate:
            logger.info("Mutation LLM rejetée par la validation — repli local.")
        return cleaned

    @staticmethod
    def _sanitize_genome(candidate: Optional[str]) -> Optional[str]:
        """Valide un ADN candidat. Retourne `None` si non conforme."""
        if not candidate:
            return None
        text = candidate.strip().strip('"').strip("`").strip()
        # Le modèle préfixe parfois sa réponse : on retire l'amorce.
        text = re.sub(r"^(voici|nouveau prompt|prompt)\s*:?\s*", "", text, flags=re.IGNORECASE)
        if not (80 <= len(text) <= 1200):
            return None
        if FORBIDDEN_PATTERNS.search(text):
            logger.warning("ADN muté contenant une consigne interdite — rejeté.")
            return None
        return text

    def _local_mutate(self, agent: Agent) -> str:
        """Mutation déterministe hors-ligne : ajout/remplacement d'un fragment d'ADN."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", agent.system_prompt) if s.strip()]
        fragment = self.rng.choice(LOCAL_MUTATION_FRAGMENTS)

        if fragment in agent.system_prompt:
            # Le fragment est déjà présent : on en retire un autre à la place
            # pour éviter l'accumulation infinie de consignes.
            if len(sentences) > 2:
                sentences.pop(self.rng.randrange(1, len(sentences)))
        elif len(sentences) >= 5:
            sentences[self.rng.randrange(1, len(sentences))] = fragment
        else:
            sentences.append(fragment)

        return " ".join(sentences)[:1200]

    # -- Évolution complète -------------------------------------------------

    def evolve(
        self, population: Sequence[Agent], previous_best: Optional[float] = None
    ) -> List[Agent]:
        """Produit la génération suivante à partir d'une population **déjà évaluée**.

        `previous_best` est la meilleure fitness de la génération précédente
        (lue en base). Elle pilote l'auto-adaptation du taux de mutation.
        """
        if not population:
            return create_seed_population(self.population_size, self.rng)

        ranked = sorted(population, key=lambda a: a.fitness, reverse=True)
        next_generation = ranked[0].generation + 1
        rate_drift = self._rate_drift(ranked[0].fitness, previous_best)

        diversity = self._diversity(ranked)
        pressure = 1.0 if diversity >= 0.6 else 1.6
        if pressure > 1.0:
            logger.info(
                "Diversité génétique faible (%.2f) — pression de mutation portée à x%.1f.",
                diversity,
                pressure,
            )

        children: List[Agent] = []

        # 1) Élitisme : les meilleurs ADN traversent la génération intacts.
        for elite in ranked[: self.elite_size]:
            clone = Agent.from_dict(elite.to_dict())
            clone.id = uuid.uuid4().hex[:16]  # nouvel enregistrement, même génome
            clone.generation = next_generation
            clone.parent_a = elite.id
            clone.parent_b = None
            clone.origin = "elite"
            clone.name = elite.name
            children.append(clone)

        # 2) Reproduction : tournoi → croisement → mutation.
        guard = 0
        while len(children) < self.population_size and guard < self.population_size * 10:
            guard += 1  # borne dure : jamais de boucle infinie
            parent_a = self._tournament(ranked)
            parent_b = self._tournament(ranked)
            if parent_b.id == parent_a.id and len(ranked) > 1:
                parent_b = self._tournament([a for a in ranked if a.id != parent_a.id])

            child = self.crossover(parent_a, parent_b, next_generation)
            child = self.mutate(
                child,
                pressure,
                # Le descendant n'a pas encore de score : on décrit sa lignée.
                reference_fitness=(parent_a.fitness + parent_b.fitness) / 2,
                rate_drift=rate_drift,
            )
            child.fitness = 0.0  # la fitness se regagne à chaque génération

            # Anti-clonage : un ADN strictement identique à un frère est re-muté.
            if any(c.dna_hash == child.dna_hash for c in children):
                child.system_prompt = self._local_mutate(child)
                child.__post_init__()
            children.append(child)

        # Filet de sécurité si la boucle a été interrompue par le garde-fou.
        while len(children) < self.population_size:
            filler = create_seed_population(1, self.rng)[0]
            filler.generation = next_generation
            children.append(filler)

        # Nommage lisible. Concaténer les noms des parents ("ProspexProspe")
        # dégénère en quelques générations : tous les descendants finissent
        # homonymes et illisibles dans les rapports. Le nom porte donc le ton
        # (déjà muté à ce stade) et la position ; la filiation reste tracée par
        # les colonnes parent_a / parent_b.
        for index, child in enumerate(children, start=1):
            if child.origin != "elite":
                child.name = f"{child.tone.capitalize()}-{next_generation}.{index}"

        logger.info(
            "Génération %d créée : %d agents (%d élite(s), diversité %.2f, "
            "taux de mutation %+.2f).",
            next_generation,
            len(children),
            self.elite_size,
            self._diversity(children),
            rate_drift,
        )
        return children

    # -- Reporting ----------------------------------------------------------

    @staticmethod
    def summarize(population: Sequence[Agent]) -> Dict[str, Any]:
        """Statistiques de population pour les logs et le rapport public."""
        if not population:
            return {"population": 0, "best_fitness": 0.0, "avg_fitness": 0.0}
        scores = [a.fitness for a in population]
        best = max(population, key=lambda a: a.fitness)
        return {
            "population": len(population),
            "generation": best.generation,
            "best_fitness": round(best.fitness, 4),
            "best_agent": best.name,
            "best_agent_id": best.id,
            "avg_fitness": round(sum(scores) / len(scores), 4),
            "diversity": round(GeneticEngine._diversity(population), 3),
        }


def select_best(population: Sequence[Agent]) -> Optional[Agent]:
    """Agent le plus performant d'une population (None si vide)."""
    return max(population, key=lambda a: a.fitness) if population else None


__all__ = ["GeneticEngine", "select_best", "FORBIDDEN_PATTERNS"]
