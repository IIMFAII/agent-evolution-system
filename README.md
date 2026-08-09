# Système autonome d'agents évolutifs

Population d'agents rédacteurs qui **évolue toutes les 2 heures** pour produire
de la veille technologique synthétisée. Infrastructure 100 % gratuite
(GitHub Actions + GitHub Pages + offre gratuite Groq), autonome, et conçue dès
l'origine pour rester dans un cadre légal strict.

À chaque cycle, cinq agents rédigent une synthèse à partir des mêmes sources ;
une fonction de *fitness* les départage ; le meilleur contenu est publié ; les
meilleurs ADN se reproduisent, mutent, et forment la génération suivante.

---

## Sommaire

- [Principe](#principe)
- [Cadre légal et éthique](#cadre-légal-et-éthique)
- [Architecture](#architecture)
- [Installation locale](#installation-locale)
- [Déploiement sur GitHub](#déploiement-sur-github)
- [Configuration](#configuration)
- [Fonction de fitness](#fonction-de-fitness)
- [Monétisation](#monétisation--ce-qui-est-automatisable-et-ce-qui-ne-lest-pas)
- [Comportement en mode dégradé](#comportement-en-mode-dégradé)
- [Tests](#tests)
- [Limites connues](#limites-connues)

---

## Principe

```
 ┌────────────┐   flux RSS publics    ┌────────────┐
 │ ingestor   │──────────────────────▶│  items     │
 └────────────┘  robots.txt + délais  └─────┬──────┘
                                            │
      ┌─────────────────────────────────────┴────────────────┐
      │  Population de N agents (ADN = prompt + ton + params) │
      └─────────────────────────┬────────────────────────────┘
                                │ chacun rédige une synthèse
                                ▼
 ┌────────────┐   assemblage    ┌────────────┐   fitness    ┌────────────┐
 │ publisher  │────────────────▶│  document  │─────────────▶│ evaluator  │
 │ (mentions  │                 │  complet   │              │ (5 signaux)│
 │  légales)  │                 └────────────┘              └─────┬──────┘
 └─────┬──────┘                                                   │
       │ publie le meilleur contenu conforme                      │
       ▼                                                          ▼
   docs/ (GitHub Pages)                        ┌──────────────────────────┐
                                               │ genetic_engine           │
                                               │ élitisme → tournoi →     │
                                               │ croisement → mutation    │
                                               └────────┬─────────────────┘
                                                        ▼
                                               génération suivante → SQLite
```

Un agent n'est pas du code : c'est un **ADN textuel**. Ce qui évolue, ce sont
les prompts système, les tons, les températures et les mots-clés de
spécialisation. La sélection est darwinienne : ce qui obtient une meilleure
fitness se reproduit davantage.

### Auto-adaptation du taux de mutation

Le taux de mutation n'est pas figé : il suit une règle de type Rechenberg —
**on exploite quand ça progresse, on explore quand ça stagne**. Le moteur
compare la meilleure fitness du cycle à celle du cycle précédent (relue en
base, chaque exécution démarrant dans un processus neuf) et ajuste le taux
hérité par les descendants.

L'ajustement est délibérément **global et non individuel**. Un critère par
agent — « cet agent fait-il mieux que la moyenne ? » — s'effondre toujours vers
une borne, puisque les parents issus d'un tournoi sont par construction
au-dessus de la moyenne. Vérifié en simulation sur 40 générations : progression
régulière ⇒ taux ≈ 0.33 ; plateau ⇒ taux ≈ 0.75 (exploration accrue).

---

## Cadre légal et éthique

Ces règles sont **implémentées dans le code**, pas seulement documentées.

| Contrainte | Où c'est appliqué |
|---|---|
| Aucune donnée personnelle (RGPD) | `ingestor.py` ne recopie aucun champ auteur/contact ; le schéma SQLite n'a aucune colonne d'identité ; `evaluator.py` rejette tout contenu contenant un e-mail ou un téléphone. |
| Aucun message non sollicité | Aucun envoi d'e-mail, aucun MP, aucune publication sur un réseau social. Le seul canal sortant est un webhook **que vous configurez vous-même**. |
| Sources autorisées uniquement | `ingestor.py` ne lit que des flux RSS/Atom, destinés par nature à la consommation automatisée. Aucun scraping de page HTML. |
| `robots.txt` respecté | `RobotsPolicy` charge et applique le `robots.txt` de chaque hôte, y compris `Crawl-delay`. Un `Disallow` fait sauter le flux. |
| Rate limits respectés | Délai minimal garanti entre requêtes ; un `429` abandonne le flux pour le cycle, sans réessai. |
| Transparence | `publisher.legal_footer()` ajoute la mention d'automatisation à **chaque** document. Impossible à omettre : c'est le code qui assemble, pas le modèle. |
| Affiliation honnête | Aucun lien affilié tant qu'aucun programme n'est configuré. Quand il y en a, chaque lien est signalé *et* la divulgation complète est ajoutée. La mention décrit la réalité du contenu : pas de « liens affiliés » annoncés s'il n'y en a pas. |
| Pas de clickbait ni de promesse mensongère | `evaluator.py` pénalise lourdement les accroches trompeuses et les promesses non vérifiables ; elles sont aussi interdites dans la charte envoyée au modèle. |
| Propriété intellectuelle | `originality_score()` détecte la recopie : un 8-gramme commun avec une source annule le score. Les sources sont systématiquement citées et liées. |
| Aucun secret en dur | Tout passe par l'environnement (`.env` local, GitHub Secrets en CI). `.env` est git-ignoré. |
| Pas de boucle infinie | Retries bornés, backoff plafonné, disjoncteur LLM, garde-fou d'itérations dans le moteur génétique, `timeout-minutes` sur le job CI. |

### La charte légale est hors du génome

Point d'architecture volontaire : `LEGAL_CHARTER` (dans `src/agents/agent.py`)
n'est **jamais stockée dans l'ADN mutable**. Elle est concaténée au prompt
système au moment de l'appel. Une mutation, même générée par un LLM, ne peut
donc pas supprimer les garde-fous. En complément, tout ADN muté passe par
`_sanitize_genome()`, qui rejette les consignes interdites.

---

## Architecture

```
agent-evolution-system/
├── .github/workflows/run_agents.yml   # cron '0 */2 * * *' + commit auto
├── src/
│   ├── config.py          # configuration validée (pydantic) + mentions légales
│   ├── database.py        # SQLite : runs, agents, scores, contenus, items
│   ├── ingestor.py        # RSS légal : robots.txt, rate limit, anti-PII
│   ├── llm.py             # client Groq + disjoncteur fail-safe
│   ├── evaluator.py       # fonction de fitness (5 signaux + verrou légal)
│   ├── monetization.py    # pistes de revenu légales + verrous CGU
│   ├── publisher.py       # assemblage conforme → Markdown + HTML
│   └── agents/
│       ├── agent.py           # ADN, production de contenu, repli hors-ligne
│       └── genetic_engine.py  # élitisme, tournoi, croisement, mutation
├── tests/test_pipeline.py # 42 tests, 100 % hors-ligne
├── main.py                # un cycle complet
├── data/evolution.db      # créée automatiquement
└── docs/                  # site publié (GitHub Pages)
```

`src/llm.py` n'apparaissait pas dans la spécification initiale : il a été
extrait parce que **deux** modules appellent le modèle (le moteur génétique
pour les mutations, l'évaluateur pour la notation des titres). Centraliser
l'accès réseau garantit un disjoncteur unique et un quota maîtrisé.

---

## Installation locale

```bash
git clone <votre-dépôt> && cd agent-evolution-system

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # puis renseignez GROQ_API_KEY (facultatif)

python -m pytest tests/ -q   # 42 tests, aucun appel réseau
python main.py               # un cycle complet
```

Options utiles :

```bash
python main.py --dry-run      # évalue et journalise, n'écrit rien dans docs/
python main.py --skip-ingest  # aucun appel réseau du tout
```

La base `data/evolution.db` et le dossier `docs/` sont créés automatiquement.

---

## Déploiement sur GitHub

1. **Poussez le dépôt** sur GitHub.
2. **Clé API** : `Settings → Secrets and variables → Actions → New repository
   secret`, nom `GROQ_API_KEY` (clé gratuite sur <https://console.groq.com>).
   *Le système fonctionne aussi sans clé, en mode dégradé.*
3. **Permissions du workflow** : `Settings → Actions → General → Workflow
   permissions` → **Read and write permissions** (nécessaire pour recommiter la
   base et les contenus).
4. **GitHub Pages** : `Settings → Pages` → source `Deploy from a branch`,
   branche `main`, dossier `/docs`.
5. **Premier lancement** : onglet `Actions` → *Cycle d'évolution des agents* →
   `Run workflow`. Ensuite, le cron prend le relais toutes les 2 heures.

Secrets et variables reconnus par le workflow :

| Nom | Type | Rôle |
|---|---|---|
| `GROQ_API_KEY` | secret | Clé API Groq. Absente ⇒ mode dégradé. |
| `AFFILIATE_TAG` | secret | Identifiant d'affiliation. Absent ⇒ aucun lien commercial. |
| `PUBLISH_WEBHOOK_URL` | secret | Webhook sortant optionnel. |
| `GROQ_MODEL`, `RSS_FEEDS`, `AFFILIATE_BASE_URL`, `POPULATION_SIZE`, `MAX_ITEMS_PER_RUN` | variables | Réglages non sensibles. |

> **Note GitHub** : un workflow planifié est automatiquement désactivé après
> 60 jours d'inactivité du dépôt. Les commits produits par le job suffisent en
> général à l'éviter.

---

## Configuration

Toutes les options sont documentées dans [`.env.example`](.env.example). Les
principales :

| Variable | Défaut | Effet |
|---|---|---|
| `POPULATION_SIZE` | `5` | Nombre d'agents par génération. |
| `ELITE_SIZE` | `1` | Agents conservés intacts (élitisme). Toujours < population. |
| `BASE_MUTATION_RATE` | `0.30` | Taux de mutation initial ; il s'auto-adapte ensuite. |
| `MAX_ITEMS_PER_RUN` | `12` | Items traités par cycle (maîtrise du quota LLM). |
| `RSS_FEEDS` | 3 flux tech | Flux à ingérer, séparés par des virgules. |
| `MIN_DELAY_BETWEEN_REQUESTS` | `2.0` | Politesse réseau, en secondes. |
| `AFFILIATE_TAG` / `AFFILIATE_BASE_URL` | vides | Affiliation désactivée tant que les deux ne sont pas remplis. |

Une valeur invalide n'interrompt jamais le démarrage : `config.py` la remplace
par le défaut sûr et journalise l'incident.

---

## Fonction de fitness

Sur un cycle de 2 heures, aucune métrique d'audience réelle n'est disponible.
La fitness combine donc cinq **signaux proxy** calculables immédiatement :

```
fitness = 0.30 · conformité    mentions légales, RGPD, sources citées
        + 0.20 · lisibilité    Flesch adapté au français (Kandel & Moles)
        + 0.20 · pertinence    couverture des thèmes sources, anti-bourrage
        + 0.20 · potentiel CTR analyse du titre (heuristique + modèle léger)
        + 0.10 · originalité   faible recouvrement n-grammes avec les sources
```

**La conformité est un verrou, pas une pondération.** Un contenu non conforme
voit sa fitness plafonnée à `0.10` : il ne peut jamais gagner la sélection,
quelles que soient ses autres qualités. Et seul un contenu conforme dépassant
le seuil `PUBLISH_THRESHOLD` (`0.45`) est publié — un cycle peut légitimement
ne rien publier.

Deux textes distincts sont analysés : la **conformité** porte sur le document
final assemblé (celui qui sera publié), les **signaux de qualité** sur la
rédaction brute de l'agent. Sans cette séparation, la bibliographie — qui cite
les titres sources mot pour mot — ferait chuter à tort le score d'originalité.

Le CTR est le seul signal qui utilise le LLM (`llama-3.1-8b-instant` note le
titre de 0 à 10). La note du modèle est **moyennée** avec l'heuristique, jamais
substituée : si le LLM est indisponible, les scores restent comparables entre
cycles.

---

## Comportement en mode dégradé

Le contrat est explicite : **aucune défaillance externe ne fait planter le
pipeline.**

| Situation | Comportement |
|---|---|
| `GROQ_API_KEY` absente | Rédaction et mutations déterministes locales. Le cycle publie quand même. |
| Quota Groq atteint (`429`) | Disjoncteur ouvert immédiatement, aucun réessai, bascule hors-ligne. |
| Clé refusée (`401`) | Idem, sans insister. |
| Un flux RSS mort ou interdit | Flux ignoré, les autres sont traités. Motif tracé dans `docs/status.json`. |
| Tous les flux en échec | Cycle terminé proprement en statut `no_data`, aucun quota consommé. |
| Aucun item nouveau | Idem : rien n'est republié en double. |
| Un agent qui plante | Fitness nulle pour cet agent, les autres continuent. |
| Aucun contenu conforme | Rien n'est publié, l'évolution se poursuit malgré tout. |

Chaque cycle écrit `docs/status.json` (génération, fitness, publication,
historique) et une ligne dans la table `runs`.

---

## Tests

```bash
python -m pytest tests/ -q
```

42 tests, exécution en moins d'une seconde, **aucun appel réseau** : le LLM est
remplacé par un double et l'ingestion est simulée. La CI reste ainsi gratuite
et déterministe. Ils couvrent la boucle de bout en bout, les garanties légales
(mentions obligatoires, RGPD, anti-clickbait, détection de recopie,
`robots.txt`, `429`) et la robustesse (LLM absent, LLM en panne, base vierge,
aucun item).

---

## Monétisation : ce qui est automatisable, et ce qui ne l'est pas

Le module [`src/monetization.py`](src/monetization.py) cherche seul des pistes de
revenu **légales** à chaque cycle, les classe, et **bloque par le code** celles
dont les CGU sont incompatibles avec un contenu entièrement généré. Le résultat
est visible sur le tableau de bord.

### La limite est un fait, pas un choix de conception

**Un agent ne peut pas encaisser d'argent.** Tout canal d'encaissement impose un
KYC : identité vérifiée, justificatifs, données fiscales, acceptation de CGU par
une personne. Un job cron ne peut pas ouvrir ces comptes — il ne peut
qu'utiliser les identifiants d'un compte déjà ouvert par un humain.

Corollaire souvent ignoré : une fois le compte ouvert, **le versement est déjà
automatique**. Les réseaux paient seuls sur le RIB enregistré, à leur échéance.
La partie « collecte et envoie » ne demande aucun code — elle demande un compte.

### Canaux bloqués par le code

| Canal | Pourquoi il est fermé |
|---|---|
| **Micro-tâches rémunérées** (annotation, cartographie de données) | Ces plateformes rémunèrent explicitement un travail **humain**. La politique d'usage d'Amazon Mechanical Turk interdit « d'utiliser des bots, des scripts ou d'autres méthodes automatisées pour compléter des HITs ». Amazon a par ailleurs fermé MTurk aux nouveaux clients le 30 juillet 2026, après avoir constaté que 33 à 46 % des travailleurs recouraient à l'IA. Automatiser ce canal, c'est se faire payer pour un travail dont on déclare faussement l'origine. Blocage **inconditionnel**. |
| **Affiliation généraliste** (places de marché) | Le contrat Amazon Partenaires interdit l'usage des liens de suivi « en lien avec l'IA générative » (clause de mars 2024), et sa politique d'avril 2026 exige un commentaire ou une analyse à valeur ajoutée humaine. Sanction : fermeture du compte et perte des commissions non versées. Blocage **levé** si `FULLY_AUTOMATED_CONTENT=false`, c'est-à-dire si un humain relit réellement les contenus. |

### Canaux ouverts, par ordre d'accessibilité

Dons et soutien récurrent · Licence du jeu de données · Affiliation en direct
auprès d'éditeurs · Sponsoring. Chacun expose ses seuils (audience, volume de
corpus), ses étapes humaines irréductibles, ce que le pipeline prend en charge,
et ses contraintes légales.

### Le verdict actuel du système

> Aucun canal ne produira d'euro tant que le site n'a pas d'audience mesurée.
> La priorité n'est pas d'ouvrir un compte : c'est d'obtenir des lecteurs, puis
> de les compter.

C'est volontaire : le score d'opportunité pondère l'audience à 40 %. Un
classement qui l'ignorerait mentirait sur la faisabilité.

### Revenus constatés

La table `revenue` n'est alimentée que par un connecteur rattaché à un compte
réel. **Le système n'invente jamais un montant** : une table vide affiche
sincèrement 0,00 €.

### Obligations dès le premier euro

- Des revenus commerciaux réguliers imposent un statut ; la micro-entreprise est
  la voie usuelle pour démarrer.
- Seuils micro-entreprise 2026 : **83 600 €** en prestations de services,
  **203 100 €** en activités de vente.
- La déclaration de chiffre d'affaires à l'URSSAF est obligatoire à chaque
  échéance, **y compris à zéro euro**.

> Ce module n'est pas un conseil juridique ou fiscal. Les CGU de chaque
> programme et les obligations déclaratives engagent la personne qui ouvre le
> compte.

## Limites connues

À avoir en tête avant d'en attendre des résultats commerciaux :

- **La fitness mesure des proxys, pas de l'audience.** Elle optimise ce qu'elle
  sait mesurer (lisibilité, conformité, forme du titre), ce qui n'est pas la
  même chose que la valeur réelle pour un lecteur. Pour une boucle vraiment
  utile, il faudrait réinjecter des signaux d'audience réels (statistiques
  GitHub Pages, clics sortants) — ce que ce système ne fait pas.
- **L'affiliation ne génère rien par elle-même.** Sans trafic, un lien affilié
  ne convertit pas. Ce projet produit et publie du contenu conforme ; il
  n'apporte pas d'audience.
- **La dérive génétique est réelle.** Sur de nombreuses générations, les prompts
  peuvent converger vers une forme qui optimise le score sans gagner en qualité.
  Le seuil de diversité et la pression de mutation adaptative limitent le
  phénomène sans l'éliminer.
- **Vérifiez les CGU de chaque flux ajouté.** Les trois flux par défaut sont
  publics et destinés à la syndication ; cela ne présume pas de ceux que vous
  ajouterez, ni de vos obligations éditoriales locales en tant qu'éditeur du
  site publié.
