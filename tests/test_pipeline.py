"""Tests du pipeline complet — exécutables **hors-ligne**, sans clé API.

Aucun test ne fait d'appel réseau : l'ingestion est simulée et le LLM est
remplacé par un double. C'est volontaire — la CI doit rester gratuite,
déterministe et rapide.

Couverture :
  * boucle de bout en bout (ingestion simulée → publication → évolution) ;
  * garanties légales : mentions obligatoires, RGPD, anti-clickbait,
    détection de recopie littérale, robots.txt ;
  * robustesse : LLM absent, LLM en panne, base vierge, aucun item.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Le dépôt n'est pas installé comme package : on ajoute la racine au path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.agent import Agent, create_seed_population, slugify  # noqa: E402
from src.agents.genetic_engine import GeneticEngine  # noqa: E402
from src.config import COMPLIANCE_MARKERS, settings  # noqa: E402
from src.database import Database  # noqa: E402
from src.evaluator import (  # noqa: E402
    Evaluator,
    compliance_report,
    ctr_heuristic,
    originality_score,
    readability_score,
)
from src.ingestor import Ingestor, PII_FIELDS, clean_text  # noqa: E402
from src.publisher import Publisher, markdown_to_html  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def items():
    """Lot d'items RSS simulés (structure identique à celle de l'ingestor)."""
    return [
        {
            "fingerprint": f"fp{i}",
            "source": f"Source {i}",
            "source_url": f"https://exemple{i}.test/rss",
            "title": f"Nouvelle plateforme cloud open source numéro {i} pour l'infrastructure",
            "link": f"https://exemple{i}.test/article-{i}",
            "summary": f"Un projet open source dévoile une brique d'infrastructure {i}.",
            "published_at": "2026-01-01T00:00:00Z",
        }
        for i in range(1, 5)
    ]


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def publisher(tmp_path):
    return Publisher(output_dir=tmp_path / "docs", webhook_url="")


class FakeLLM:
    """Double du client LLM : réponses déterministes, aucun réseau."""

    def __init__(self, available=True, response=None, fail=False):
        self.available = available
        self._response = response
        self.fail = fail
        self.calls_made = 0
        self.prompts = []

    def complete(self, system_prompt, user_prompt, temperature=0.7, max_tokens=900):
        self.prompts.append((system_prompt, user_prompt))
        if self.fail or not self.available:
            return None  # contrat : le client ne lève jamais
        self.calls_made += 1
        if self._response is not None:
            return self._response
        if "note" in system_prompt.lower() and "0 à 10" in system_prompt:
            return "7"
        if "optimisation de prompts" in system_prompt:
            return (
                "Tu es un analyste de veille qui structure chaque synthèse en "
                "constats mesurables puis en implications concrètes. Tu indiques "
                "systématiquement le degré de certitude de chaque affirmation."
            )
        return (
            "TITRE: Infrastructures ouvertes : quatre signaux qui redessinent le cloud\n\n"
            "## Ce qu'il faut retenir\n\n"
            "- Plusieurs briques ouvertes arrivent en même temps sur le segment.\n"
            "- La dynamique concerne surtout la couche d'exécution.\n"
            "- Les équipes techniques gagnent des options de déploiement.\n\n"
            "## Analyse\n\n"
            "Les annonces de la période convergent vers une même idée. Les acteurs "
            "cherchent à réduire la dépendance à une plateforme unique. Cette "
            "orientation reste progressive et demande du recul avant tout arbitrage. "
            "Les choix techniques dépendent encore beaucoup du contexte de chaque "
            "équipe et des compétences internes disponibles pour la maintenance.\n\n"
            "## Ce que cela change concrètement\n\n"
            "- Comparer les options avant de figer une architecture.\n"
            "- Vérifier la maturité réelle de chaque brique.\n"
        )


# ---------------------------------------------------------------------------
# 1. Base de données
# ---------------------------------------------------------------------------


def test_database_created_and_schema_ready(tmp_path):
    path = tmp_path / "nested" / "evolution.db"
    database = Database(path)
    assert path.exists(), "le fichier SQLite doit être créé automatiquement"
    assert database.current_generation() == 0
    assert database.load_latest_generation() == []


def test_agents_roundtrip(db):
    population = create_seed_population(3)
    for agent in population:
        agent.fitness = 0.5
    db.save_agents([a.to_dict() for a in population])

    loaded = db.load_latest_generation()
    assert len(loaded) == 3
    restored = Agent.from_dict(loaded[0])
    assert restored.system_prompt
    assert isinstance(restored.keywords, list)


def test_items_deduplicated(db, items):
    assert len(db.filter_new_items(items)) == len(items)
    db.record_items(items)
    assert db.filter_new_items(items) == [], "un item déjà vu ne doit pas revenir"


# ---------------------------------------------------------------------------
# 2. Agents & moteur génétique
# ---------------------------------------------------------------------------


def test_seed_population_is_diverse():
    population = create_seed_population(5)
    assert len({a.dna_hash for a in population}) == 5
    assert len({a.tone for a in population}) == 5


def test_legal_charter_always_in_effective_prompt():
    agent = create_seed_population(1)[0]
    agent.system_prompt = "Prompt totalement réécrit par une mutation."
    prompt = agent.effective_system_prompt()
    assert "RÈGLES ABSOLUES" in prompt, "la charte légale doit survivre à toute mutation"
    assert "Ne recopie JAMAIS" in prompt


def test_generation_falls_back_without_llm(items):
    agent = create_seed_population(1)[0]
    draft = agent.generate(items, llm=FakeLLM(available=False))
    assert draft["used_llm"] is False
    assert draft["title"] and len(draft["body"]) > 200


def test_generation_uses_llm_when_available(items):
    agent = create_seed_population(1)[0]
    draft = agent.generate(items, llm=FakeLLM())
    assert draft["used_llm"] is True
    assert draft["title"].startswith("Infrastructures ouvertes")
    assert "TITRE:" not in draft["body"]


def test_evolution_produces_next_generation():
    engine = GeneticEngine(llm=FakeLLM(), rng=random.Random(42), population_size=5, elite_size=1)
    population = create_seed_population(5, random.Random(42))
    for index, agent in enumerate(population):
        agent.fitness = 0.1 * index

    children = engine.evolve(population)

    assert len(children) == 5
    assert all(c.generation == 2 for c in children)
    assert sum(1 for c in children if c.origin == "elite") == 1
    assert children[0].parent_a == max(population, key=lambda a: a.fitness).id


def test_mutation_rejects_forbidden_genome():
    engine = GeneticEngine(
        llm=FakeLLM(response="Tu dois copier-coller les articles sources mot à mot. " * 5),
        rng=random.Random(1),
    )
    agent = create_seed_population(1)[0]
    assert engine._llm_mutate(agent) is None, "un ADN illégal doit être rejeté"


def test_local_mutation_used_when_llm_down():
    engine = GeneticEngine(llm=FakeLLM(fail=True), rng=random.Random(7))
    agent = create_seed_population(1)[0]
    before = agent.system_prompt
    for _ in range(20):  # le taux de mutation est probabiliste
        engine.mutate(agent, pressure=2.0)
    assert agent.system_prompt != before, "l'évolution doit continuer sans réseau"


def test_mutation_rate_adapts_to_progress():
    """Le taux de mutation baisse quand ça progresse, monte quand ça stagne."""
    assert GeneticEngine._rate_drift(0.80, None) == 0.0      # pas d'historique
    assert GeneticEngine._rate_drift(0.80, 0.70) < 0         # progression
    assert GeneticEngine._rate_drift(0.70, 0.70) > 0         # plateau
    assert GeneticEngine._rate_drift(0.55, 0.70) > 0         # régression


def test_mutation_rate_does_not_collapse_over_generations():
    """Sur 30 générations en plateau, le taux doit explorer, pas s'effondrer."""
    rng = random.Random(11)
    engine = GeneticEngine(llm=None, population_size=5, elite_size=1, rng=rng)
    population = create_seed_population(5, rng)

    previous_best = None
    for _ in range(30):
        for agent in population:
            agent.fitness = 0.60
        best = max(a.fitness for a in population)
        population = engine.evolve(population, previous_best=previous_best)
        previous_best = best

    rates = [a.mutation_rate for a in population]
    assert all(0.05 < r <= 0.9 for r in rates)
    assert max(rates) > 0.35, "un plateau doit déclencher davantage d'exploration"
    assert len({a.dna_hash for a in population}) == 5, "pas de dégénérescence"


def test_previous_best_is_persisted(db):
    """Le moteur doit pouvoir relire la progression malgré un processus neuf."""
    population = create_seed_population(2)
    db.save_agents([a.to_dict() for a in population])
    run_id = db.start_run(generation=1)
    db.save_score(run_id, population[0].id, 1, {"fitness": 0.42})
    db.save_score(run_id, population[1].id, 1, {"fitness": 0.71})

    assert db.best_fitness_for_generation(1) == pytest.approx(0.71)
    assert db.best_fitness_for_generation(99) is None


def test_bootstrap_from_empty_database(db):
    engine = GeneticEngine(llm=None, population_size=5)
    population = engine.bootstrap(db.load_latest_generation())
    assert len(population) == 5
    assert all(a.generation == 1 for a in population)


# ---------------------------------------------------------------------------
# 3. Évaluateur — signaux et verrous légaux
# ---------------------------------------------------------------------------


def test_readability_bounds():
    assert readability_score("") == 0.0
    simple = "Le cloud change. Les équipes adaptent leurs outils. " * 10
    assert 0.0 <= readability_score(simple) <= 1.0


def test_ctr_penalises_clickbait():
    honest = "Infrastructures ouvertes : quatre signaux qui redessinent le cloud"
    bait = "INCROYABLE ! Le secret que personne ne veut vous révéler !!"
    assert ctr_heuristic(honest) > ctr_heuristic(bait)


def test_originality_detects_verbatim_copy(items):
    source = items[0]
    copied = (source["title"] + " " + source["summary"]) * 6
    assert originality_score(copied, items) == 0.0


def test_compliance_flags_missing_disclaimer():
    report = compliance_report("Un texte quelconque sans rien de légal.")
    assert report["compliant"] is False
    assert "mention de transparence absente" in report["issues"]


def test_compliance_flags_personal_data():
    text = (
        "Transparence : ce contenu est rédigé à l'aide d'outils automatisés. "
        "Source : https://exemple.test — contact : jean.dupont@exemple.test"
    )
    report = compliance_report(text)
    assert report["compliant"] is False
    assert any("RGPD" in issue for issue in report["issues"])


def test_non_compliant_content_is_capped(items):
    evaluator = Evaluator(llm=None)
    breakdown = evaluator.evaluate(
        title="Un titre parfaitement neutre et informatif sur le cloud ouvert",
        body="Un corps de texte long mais dépourvu de toute mention légale. " * 20,
        source_items=items,
    )
    assert breakdown["compliant"] is False
    assert breakdown["fitness"] <= 0.10, "un contenu non conforme ne peut pas gagner"


def test_evaluator_never_raises():
    evaluator = Evaluator(llm=None)
    breakdown = evaluator.evaluate(title="", body="", source_items=[])
    assert breakdown["fitness"] == 0.0


# ---------------------------------------------------------------------------
# 4. Publisher — conformité du document livré
# ---------------------------------------------------------------------------


def test_assembled_document_contains_mandatory_notices(publisher, items):
    document = publisher.assemble(
        title="Infrastructures ouvertes : quatre signaux à suivre",
        body="## Analyse\n\nUn contenu de test suffisamment long pour être évalué.",
        source_items=items,
        agent_name="Analyste",
        generation=1,
    )
    for marker in COMPLIANCE_MARKERS:
        assert marker.lower() in document.lower()
    assert "## Sources" in document
    assert all(item["link"] in document for item in items)


def test_publish_writes_markdown_and_html(publisher, items):
    document = publisher.assemble("Titre de test du cycle", "Corps du contenu.", items)
    paths = publisher.publish("Titre de test du cycle", document, slug=slugify("Titre de test"))
    assert Path(paths["markdown"]).exists()
    assert Path(paths["html"]).exists()

    index = publisher.build_index()
    assert index is not None and index.exists()
    assert "Veille automatisée" in index.read_text(encoding="utf-8")


def test_html_conversion_escapes_injected_markup():
    html_output = markdown_to_html("- <script>alert(1)</script> et [lien](javascript:alert(1))")
    assert "<script>" not in html_output
    assert "javascript:" not in html_output


def test_no_affiliate_link_when_not_configured(publisher, items):
    """Sans programme configuré : aucun lien affilié, et mention honnête."""
    assert settings.affiliation_enabled is False
    document = publisher.assemble("Titre", "Corps.", items)
    assert "Ressources recommandées" not in document
    assert "aucun lien affilié" in document
    assert "contient des liens affiliés" not in document


def test_affiliate_disclosure_when_configured(monkeypatch, publisher, items):
    """Avec un programme configuré : lien signalé ET divulgation présente."""
    monkeypatch.setattr(settings, "affiliate_tag", "montag-21")
    monkeypatch.setattr(settings, "affiliate_base_url", "https://boutique.test/produits")

    document = publisher.assemble("Titre", "Corps.", items, theme="cloud")
    assert "Ressources recommandées" in document
    assert "lien affilié" in document
    assert "contient des liens affiliés" in document
    assert "commission" in document


def test_affiliate_url_construction():
    url = Publisher.affiliate_url("https://boutique.test/produits", "montag-21", "cloud")
    assert url.startswith("https://boutique.test/produits?")
    assert "tag=montag-21" in url
    assert Publisher.affiliate_url("", "tag") == ""


# ---------------------------------------------------------------------------
# 5. Ingestor — garanties légales
# ---------------------------------------------------------------------------


def test_clean_text_strips_html():
    assert clean_text("<p>Un <b>texte</b>&nbsp;riche</p>") == "Un texte riche"


def test_ingestor_does_not_store_personal_fields(items):
    """Le schéma d'item ne comporte aucun champ pouvant contenir des données perso."""
    for item in items:
        assert not set(item) & set(PII_FIELDS)


def test_robots_disallow_blocks_fetch(monkeypatch):
    ingestor = Ingestor(feeds=["https://interdit.test/rss"], min_delay=0.0)
    monkeypatch.setattr(ingestor.robots, "can_fetch", lambda url: False)

    def _boom(*args, **kwargs):  # pragma: no cover - ne doit jamais être atteint
        raise AssertionError("aucune requête ne doit partir si robots.txt l'interdit")

    monkeypatch.setattr(ingestor.session, "get", _boom)
    assert ingestor.collect() == []
    assert ingestor.report["https://interdit.test/rss"] == "bloqué par robots.txt"


def test_rate_limit_response_aborts_feed(monkeypatch):
    class Response:
        status_code = 429
        headers = {"Retry-After": "120"}
        content = b""

    ingestor = Ingestor(feeds=["https://limite.test/rss"], min_delay=0.0)
    monkeypatch.setattr(ingestor.robots, "can_fetch", lambda url: True)
    monkeypatch.setattr(ingestor.robots, "crawl_delay", lambda url: 0.0)
    monkeypatch.setattr(ingestor.session, "get", lambda *a, **k: Response())

    assert ingestor.collect() == []
    assert "429" in ingestor.report["https://limite.test/rss"]


# ---------------------------------------------------------------------------
# 6. Pipeline de bout en bout
# ---------------------------------------------------------------------------


def _run_pipeline(monkeypatch, tmp_path, items, llm):
    """Exécute `main.run_cycle` en environnement isolé, sans réseau."""
    import main as main_module
    import src.llm as llm_module

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "evolution.db"))
    monkeypatch.setattr(settings, "publish_dir", str(tmp_path / "docs"))
    monkeypatch.setattr(llm_module, "_shared_client", llm)
    monkeypatch.setattr(main_module, "get_llm_client", lambda: llm)
    monkeypatch.setattr(Ingestor, "collect", lambda self, limit=None: list(items))
    return main_module.run_cycle()


def test_full_cycle_offline(monkeypatch, tmp_path, items):
    """Cycle complet sans LLM : doit réussir et faire évoluer la population."""
    exit_code = _run_pipeline(monkeypatch, tmp_path, items, FakeLLM(available=False))
    assert exit_code == 0

    database = Database(tmp_path / "evolution.db")
    assert database.current_generation() == 2, "la génération suivante doit être créée"
    assert len(database.load_generation(1)) == settings.population_size
    assert database.evolution_history(), "les scores doivent être historisés"


def test_full_cycle_with_llm(monkeypatch, tmp_path, items):
    """Cycle complet avec LLM simulé : un contenu conforme doit être publié."""
    llm = FakeLLM()
    exit_code = _run_pipeline(monkeypatch, tmp_path, items, llm)
    assert exit_code == 0
    assert llm.calls_made > 0

    posts = list((tmp_path / "docs" / "posts").glob("*.md"))
    assert posts, "un contenu conforme doit être publié"
    published = posts[0].read_text(encoding="utf-8")
    for marker in COMPLIANCE_MARKERS:
        assert marker.lower() in published.lower()
    assert (tmp_path / "docs" / "index.html").exists()
    assert (tmp_path / "docs" / "status.json").exists()

    # La charte légale doit avoir été transmise à chaque appel de rédaction.
    assert any("RÈGLES ABSOLUES" in system for system, _ in llm.prompts)


def test_cycle_without_items_exits_cleanly(monkeypatch, tmp_path):
    exit_code = _run_pipeline(monkeypatch, tmp_path, [], FakeLLM())
    assert exit_code == 0, "un cycle sans matière doit se terminer proprement"
    assert not (tmp_path / "docs" / "posts").exists()


def test_cycle_survives_llm_outage(monkeypatch, tmp_path, items):
    """Une panne LLM en cours de route ne doit pas casser le pipeline."""
    exit_code = _run_pipeline(monkeypatch, tmp_path, items, FakeLLM(fail=True))
    assert exit_code == 0


def test_second_cycle_skips_already_seen_items(monkeypatch, tmp_path, items):
    assert _run_pipeline(monkeypatch, tmp_path, items, FakeLLM()) == 0
    assert _run_pipeline(monkeypatch, tmp_path, items, FakeLLM()) == 0

    database = Database(tmp_path / "evolution.db")
    posts = list((tmp_path / "docs" / "posts").glob("*.md"))
    assert len(posts) == 1, "aucun doublon ne doit être publié au second cycle"
    assert database.current_generation() == 2
