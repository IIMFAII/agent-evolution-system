"""Recherche autonome de pistes de monétisation **légales**.

Ce module répond à une demande précise : « que les agents trouvent seuls des
méthodes légales pour gagner de l'argent ». Il le fait, avec une limite qui est
un fait et non un choix de conception :

    Un agent ne peut pas encaisser d'argent.

Tout canal d'encaissement (programme d'affiliation, Stripe, PayPal, banque)
impose un KYC : identité vérifiée, justificatifs, données fiscales, acceptation
de CGU par une personne physique ou morale. Un job cron ne peut pas ouvrir ces
comptes ; il ne peut qu'utiliser les identifiants d'un compte déjà ouvert par un
humain. En revanche, une fois le compte ouvert, **le versement est déjà
automatique** : les réseaux paient seuls sur le RIB enregistré, selon leur
seuil et leur échéance. La « collecte » ne demande donc pas de code.

Ce que ce module automatise réellement :
  1. évaluer l'état monétisable du site (thèmes, volume, audience, nature du
     contenu) ;
  2. confronter cet état à un catalogue de mécanismes légaux ;
  3. **bloquer** ceux dont les CGU sont incompatibles avec un contenu
     entièrement généré ;
  4. produire un plan d'action classé, en séparant ce que le pipeline peut
     faire de ce qui exige une action humaine irréductible.

⚠️ Ce module ne constitue pas un conseil juridique ou fiscal. Les CGU de chaque
programme et les obligations déclaratives doivent être vérifiées par l'humain
qui ouvre le compte — c'est lui, et lui seul, qui est engagé.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibilité CGU avec un contenu entièrement automatisé
# ---------------------------------------------------------------------------

#: Le mécanisme accepte un contenu généré sans intervention humaine.
COMPATIBLE = "compatible"
#: Toléré sous conditions strictes, à vérifier programme par programme.
RESTREINT = "restreint"
#: Interdit par les CGU pour un contenu entièrement généré. Jamais recommandé.
INCOMPATIBLE = "incompatible"

POLICY_LABEL = {
    COMPATIBLE: "Compatible contenu automatisé",
    RESTREINT: "À vérifier au cas par cas",
    INCOMPATIBLE: "Interdit par les CGU",
}


@dataclass
class Mechanism:
    """Un mécanisme de revenu légal, avec ses contraintes réelles."""

    id: str
    name: str
    summary: str
    #: Compatibilité des CGU avec un contenu 100 % généré.
    ai_policy: str
    #: Audience mensuelle en dessous de laquelle le canal ne produit rien.
    audience_floor: int
    #: Effort humain d'ouverture, de 0 (trivial) à 1 (lourd).
    setup_effort: float
    #: Délai réaliste avant le premier euro, en jours.
    days_to_first_euro: int
    #: Étapes que seul un humain peut accomplir (KYC, CGU, fiscalité).
    human_steps: List[str]
    #: Ce que le pipeline peut prendre en charge une fois le compte ouvert.
    automatable: List[str]
    #: Contraintes légales françaises à connaître.
    legal_notes: List[str]
    #: Volume de publications en dessous duquel le canal n'a rien à vendre.
    #: Distinct de l'audience : une licence de données ne dépend pas des
    #: lecteurs du site, mais elle exige un corpus substantiel.
    corpus_floor: int = 0
    #: Source de la contrainte, quand elle vient d'une politique publiée.
    policy_source: Optional[str] = None
    #: Motif de blocage **inconditionnel**. Renseigné quand le canal est fermé
    #: quoi qu'il arrive, et pas seulement parce que ce site-ci est automatisé.
    blocking_note: Optional[str] = None


#: Catalogue de mécanismes. Volontairement décrit par **catégories** plutôt que
#: par offres commerciales : les taux et conditions d'un programme donné
#: changent, et les inventer serait pire que de ne rien dire.
MECHANISMS: List[Mechanism] = [
    Mechanism(
        id="dons",
        name="Dons et soutien récurrent",
        summary=(
            "Une page de soutien (GitHub Sponsors, Liberapay, Ko-fi…). Le donateur "
            "soutient le projet, pas un produit : aucune promesse commerciale n'est "
            "faite, donc aucune CGU n'encadre l'origine du contenu."
        ),
        ai_policy=COMPATIBLE,
        audience_floor=200,
        setup_effort=0.2,
        days_to_first_euro=7,
        human_steps=[
            "Créer le compte de collecte à votre nom (KYC : pièce d'identité, RIB).",
            "Renseigner vos informations fiscales sur la plateforme.",
        ],
        automatable=[
            "Publier et maintenir la page de soutien sur le site.",
            "Afficher le total collecté dans le tableau de bord (via l'API de la plateforme).",
        ],
        legal_notes=[
            "Les dons perçus régulièrement en contrepartie d'une activité sont des "
            "revenus imposables : ils doivent être déclarés.",
            "Indiquer clairement que le contenu est automatisé, comme le fait déjà le site.",
        ],
    ),
    Mechanism(
        id="licence_donnees",
        name="Licence du jeu de données",
        summary=(
            "Vendre l'accès à la donnée agrégée que le pipeline produit déjà "
            "(historique de veille structuré, séries de fitness, corpus horodaté) "
            "plutôt qu'au texte. La valeur porte sur l'agrégation et la fraîcheur, "
            "pas sur la rédaction — donc aucune CGU sur l'origine du texte ne s'applique."
        ),
        ai_policy=COMPATIBLE,
        audience_floor=0,
        corpus_floor=150,
        setup_effort=0.6,
        days_to_first_euro=60,
        human_steps=[
            "Créer un compte d'encaissement professionnel (KYC + statut d'entreprise).",
            "Rédiger et publier des conditions de licence.",
            "Démarcher les premiers clients : aucun automatisme ne remplace cette étape.",
        ],
        automatable=[
            "Produire l'export structuré à chaque cycle (déjà en place : data.json).",
            "Versionner et horodater le corpus pour en garantir la traçabilité.",
        ],
        legal_notes=[
            "Ne redistribuer que ce que vous produisez : les métadonnées et vos "
            "synthèses, jamais le texte intégral des sources.",
            "Le droit sui generis des bases de données protège l'investissement, "
            "à condition que la base soit substantielle.",
        ],
    ),
    Mechanism(
        id="affiliation_directe",
        name="Affiliation en direct auprès d'éditeurs",
        summary=(
            "Programmes gérés directement par un éditeur (outils, SaaS, hébergeurs) "
            "plutôt que par une place de marché généraliste. Les conditions varient "
            "fortement : certains acceptent un contenu assisté par IA s'il est "
            "divulgué, d'autres l'interdisent."
        ),
        ai_policy=RESTREINT,
        audience_floor=1000,
        setup_effort=0.5,
        days_to_first_euro=45,
        human_steps=[
            "Candidater au programme et faire valider le site (revue humaine fréquente).",
            "Lire la clause « contenu généré par IA » des CGU AVANT d'insérer un lien.",
            "Fournir identité et coordonnées bancaires (KYC).",
        ],
        automatable=[
            "Injecter le tag d'affiliation et sa divulgation (déjà en place).",
            "Suivre les commissions via l'API du programme.",
        ],
        legal_notes=[
            "La divulgation du lien affilié est obligatoire et doit être visible "
            "avant le lien, pas seulement en pied de page.",
            "Un lien affilié inséré en violation des CGU expose à la fermeture du "
            "compte et à la perte des commissions déjà acquises.",
        ],
    ),
    Mechanism(
        id="sponsoring",
        name="Sponsoring d'une rubrique ou d'une newsletter",
        summary=(
            "Un annonceur paie pour un emplacement identifié. Le revenu ne dépend "
            "pas d'un achat, seulement de l'exposition — mais il exige une audience "
            "mesurable et vérifiable par l'annonceur."
        ),
        ai_policy=RESTREINT,
        audience_floor=3000,
        setup_effort=0.7,
        days_to_first_euro=90,
        human_steps=[
            "Constituer un dossier d'audience vérifiable (statistiques publiques).",
            "Négocier et contractualiser avec l'annonceur.",
            "Facturer : cela suppose un statut d'entreprise.",
        ],
        automatable=[
            "Insérer l'emplacement sponsorisé et son marquage à chaque publication.",
            "Produire les statistiques d'affichage remises à l'annonceur.",
        ],
        legal_notes=[
            "Tout contenu sponsorisé doit être identifié comme tel de manière "
            "explicite : l'absence de marquage est une pratique commerciale trompeuse.",
            "Informer l'annonceur que le contenu est automatisé : le lui cacher "
            "vicierait le consentement contractuel.",
        ],
    ),
    Mechanism(
        id="micro_taches",
        name="Micro-tâches rémunérées (annotation, cartographie de données)",
        summary=(
            "Les plateformes de travail à la tâche paient l'annotation, la "
            "catégorisation et la cartographie de données. Le canal paraît taillé "
            "pour des agents — c'est exactement l'inverse : ces plateformes vendent "
            "du jugement HUMAIN, et n'existent que parce qu'une machine ne l'a pas fait."
        ),
        ai_policy=INCOMPATIBLE,
        audience_floor=0,
        setup_effort=0.3,
        days_to_first_euro=14,
        human_steps=[
            "Aucun : ce canal ne peut pas être automatisé sans fraude envers le "
            "donneur d'ordre, qui paie pour un travail humain.",
        ],
        automatable=[],
        legal_notes=[
            "La politique d'usage d'Amazon Mechanical Turk interdit explicitement "
            "« d'utiliser des bots, des scripts ou d'autres méthodes automatisées "
            "pour compléter des HITs ».",
            "Amazon a fermé Mechanical Turk aux nouveaux clients le 30 juillet 2026, "
            "après avoir constaté que 33 à 46 % des travailleurs recouraient à l'IA, "
            "ce qui a ruiné la valeur du service.",
            "Les plateformes équivalentes (annotation, transcription, panels d'étude) "
            "posent la même exigence d'exécution humaine : automatiser revient à se "
            "faire payer pour un travail qu'on déclare faussement humain.",
            "La voie légale pour monétiser de la donnée traitée par machine est de "
            "VENDRE le jeu de données produit, pas de se faire passer pour un "
            "travailleur humain — voir « Licence du jeu de données ».",
        ],
        policy_source="https://www.mturk.com/acceptable-use-policy",
        blocking_note=(
            "Ces plateformes rémunèrent explicitement un travail humain : les faire "
            "exécuter par un agent revient à se faire payer pour un travail dont on "
            "déclare faussement l'origine. Le canal reste fermé même si le site "
            "publiait un jour du contenu écrit par un humain."
        ),
    ),
    Mechanism(
        id="affiliation_generaliste",
        name="Affiliation généraliste (places de marché)",
        summary=(
            "Les grandes places de marché d'affiliation. Le canal le plus évident "
            "est ici le plus risqué : il est fermé à un site entièrement automatisé."
        ),
        ai_policy=INCOMPATIBLE,
        audience_floor=1000,
        setup_effort=0.4,
        days_to_first_euro=30,
        human_steps=[
            "Ce canal exige une rédaction humaine à valeur ajoutée — il ne peut pas "
            "être ouvert en l'état pour ce site.",
        ],
        automatable=[],
        legal_notes=[
            "Le contrat Amazon Partenaires interdit l'usage des liens de suivi « en "
            "lien avec l'IA générative » (clause de mars 2024), et sa politique "
            "d'avril 2026 exige un commentaire, une analyse ou une transformation "
            "apportant une valeur ajoutée humaine.",
            "Les pages de comparaison générées sans intervention humaine sont "
            "explicitement hors politique ; la sanction est la fermeture du compte "
            "et la perte des commissions non versées.",
        ],
        policy_source="https://affiliate-program.amazon.com/help/operating/policies",
    ),
]


#: Obligations qui s'appliquent dès qu'un revenu est effectivement perçu.
FISCAL_NOTES: List[str] = [
    "Percevoir des revenus commerciaux de façon régulière impose un statut : la "
    "micro-entreprise est la voie usuelle pour démarrer.",
    "Seuils micro-entreprise 2026 : 83 600 € pour les prestations de services, "
    "203 100 € pour les activités de vente.",
    "La déclaration de chiffre d'affaires à l'URSSAF est obligatoire à chaque "
    "échéance, y compris lorsque le chiffre d'affaires est nul.",
    "Les commissions d'affiliation et les dons liés à l'activité sont imposables "
    "et doivent figurer sur la déclaration de revenus.",
]


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

#: Motifs interdits dans une proposition générée par le modèle.
FORBIDDEN_ANGLE = re.compile(
    r"\b("
    r"achet(?:er|ez)\s+des\s+(?:clics|vues|abonnés)"
    r"|faux\s+avis|avis\s+fictifs?"
    r"|cloak(?:ing|er)?|dissimul(?:er|ation)\s+(?:le|la|les)\s+(?:lien|nature)"
    r"|contourn(?:er|ement)\s+(?:les\s+)?(?:cgu|conditions|règles)"
    r"|spam|cold\s*email(?:ing)?|scrap(?:er|ing)"
    r"|dropshipping\s+sans\s+stock"
    r"|sans\s+déclar(?:er|ation)|non\s+déclaré"
    r"|crypto|token|nft|parrainage\s+pyramidal|mlm"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class SiteState:
    """Photographie de l'état monétisable du site à un instant donné."""

    publications: int = 0
    generations: int = 0
    #: Audience mensuelle. `None` = non mesurée, ce qui n'est pas 0 mais est
    #: traité comme tel : on ne peut pas monétiser ce qu'on ne sait pas mesurer.
    monthly_audience: Optional[int] = None
    #: True tant que la rédaction n'a aucune relecture humaine.
    fully_automated: bool = True
    themes: List[str] = field(default_factory=list)

    @property
    def effective_audience(self) -> int:
        return int(self.monthly_audience or 0)


@dataclass
class Opportunity:
    """Un mécanisme évalué face à l'état réel du site."""

    mechanism: Mechanism
    score: float
    status: str            # "recommandé" | "à préparer" | "bloqué"
    blocked_reason: Optional[str]
    rationale: str
    angle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        m = self.mechanism
        return {
            "id": m.id,
            "name": m.name,
            "summary": m.summary,
            "score": round(self.score, 4),
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "rationale": self.rationale,
            "angle": self.angle,
            "ai_policy": m.ai_policy,
            "ai_policy_label": POLICY_LABEL[m.ai_policy],
            "audience_floor": m.audience_floor,
            "days_to_first_euro": m.days_to_first_euro,
            "human_steps": list(m.human_steps),
            "automatable": list(m.automatable),
            "legal_notes": list(m.legal_notes),
            "policy_source": m.policy_source,
        }


class Strategist:
    """Classe les mécanismes de revenu face à l'état réel du site.

    Le classement est **rule-based**, pas génératif : un LLM peut proposer un
    angle éditorial, jamais décider qu'un canal est légal. Le verrou CGU est
    dans le code, comme la charte légale des agents rédacteurs.
    """

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    # -- Verrous ------------------------------------------------------------

    @staticmethod
    def _blocked_reason(mech: Mechanism, state: SiteState) -> Optional[str]:
        """Motif de blocage dur, ou `None` si le canal est ouvrable.

        Deux blocages distincts : celui qui tient à ce site-ci (contenu non
        relu par un humain, donc contraire aux CGU du canal) et celui qui tient
        au canal lui-même, fermé quelles que soient les circonstances.
        """
        if mech.blocking_note:
            return mech.blocking_note
        if mech.ai_policy == INCOMPATIBLE and state.fully_automated:
            return (
                "Les CGU de ce canal exigent une valeur ajoutée humaine. "
                "Le site publie actuellement du contenu entièrement automatisé : "
                "l'ouvrir exposerait à la fermeture du compte."
            )
        return None

    # -- Score --------------------------------------------------------------

    @staticmethod
    def _score(mech: Mechanism, state: SiteState) -> float:
        """Score d'opportunité [0,1] : accessible, rapide, peu coûteux.

        L'audience domine volontairement. Sans lecteurs, aucun canal ne produit
        de revenu — un score qui l'ignorerait mentirait sur la faisabilité.
        """
        if mech.audience_floor <= 0:
            audience_fit = 1.0
        else:
            audience_fit = min(1.0, state.effective_audience / mech.audience_floor)
        if mech.corpus_floor > 0:
            audience_fit = min(audience_fit, state.publications / mech.corpus_floor)

        effort_fit = 1.0 - mech.setup_effort
        speed_fit = max(0.0, 1.0 - mech.days_to_first_euro / 120)
        policy_fit = {COMPATIBLE: 1.0, RESTREINT: 0.55, INCOMPATIBLE: 0.0}[mech.ai_policy]

        return round(
            0.40 * audience_fit + 0.25 * policy_fit + 0.20 * effort_fit + 0.15 * speed_fit,
            4,
        )

    # -- Angle proposé par le modèle ---------------------------------------

    def _angle(self, mech: Mechanism, state: SiteState) -> Optional[str]:
        """Angle concret proposé par le LLM. `None` si indisponible ou refusé."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return None
        if not state.themes:
            return None

        raw = self.llm.complete(
            system_prompt=(
                "Tu conseilles un éditeur de site de veille technologique sur la "
                "monétisation. On te donne un mécanisme de revenu et les thèmes "
                "réellement traités par le site. Propose UN angle concret et "
                "applicable, en une phrase de 25 mots maximum.\n"
                "Interdits absolus : toute pratique trompeuse, l'achat d'audience, "
                "les faux avis, la dissimulation de la nature publicitaire d'un "
                "lien, le contournement de CGU, toute activité non déclarée. "
                "Réponds uniquement par la phrase, sans préambule."
            ),
            user_prompt=(
                f"Mécanisme : {mech.name} — {mech.summary}\n"
                f"Thèmes du site : {', '.join(state.themes[:6])}\n"
                f"Publications à ce jour : {state.publications}"
            ),
            temperature=0.6,
            max_tokens=90,
        )
        if not raw:
            return None
        angle = raw.strip().strip('"').split("\n")[0][:260]
        if len(angle) < 15 or FORBIDDEN_ANGLE.search(angle):
            logger.warning("Angle de monétisation rejeté par la validation.")
            return None
        return angle

    # -- API publique -------------------------------------------------------

    def evaluate(self, state: SiteState) -> List[Opportunity]:
        """Classe tous les mécanismes, du plus actionnable au plus bloqué."""
        out: List[Opportunity] = []

        for mech in MECHANISMS:
            blocked = self._blocked_reason(mech, state)
            score = 0.0 if blocked else self._score(mech, state)

            if blocked:
                status, rationale = "bloqué", blocked
            elif state.publications < mech.corpus_floor:
                status = "à préparer"
                rationale = (
                    f"Ne dépend pas de l'audience, mais exige un corpus vendable : "
                    f"~{mech.corpus_floor} publications contre {state.publications} "
                    f"aujourd'hui. Le pipeline y arrive seul en continuant de tourner."
                )
            elif state.effective_audience < mech.audience_floor:
                status = "à préparer"
                rationale = (
                    f"Ouvrable dès maintenant, mais ne produira rien avant "
                    f"~{mech.audience_floor} visiteurs/mois "
                    f"(audience actuelle : {state.monthly_audience if state.monthly_audience is not None else 'non mesurée'})."
                )
            else:
                status = "recommandé"
                rationale = (
                    f"Seuil d'audience atteint et CGU compatibles. Premier revenu "
                    f"réaliste sous ~{mech.days_to_first_euro} jours."
                )

            opp = Opportunity(mech, score, status, blocked, rationale)
            if status != "bloqué":
                opp.angle = self._angle(mech, state)
            out.append(opp)

        out.sort(key=lambda o: (o.status == "bloqué", -o.score))
        return out

    @staticmethod
    def verdict(opportunities: Sequence[Opportunity], state: SiteState) -> str:
        """Conclusion honnête en une phrase, affichée en tête du plan."""
        ready = [o for o in opportunities if o.status == "recommandé"]
        if state.effective_audience == 0:
            if ready:
                names = ", ".join(o.mechanism.name for o in ready)
                return (
                    "Sans audience mesurée, tous les canaux qui en dépendent sont hors "
                    f"d'atteinte. Reste ce qui n'en dépend pas : {names}."
                )
            return (
                "Aucun canal ne produira d'euro tant que le site n'a pas d'audience "
                "mesurée. La priorité n'est pas d'ouvrir un compte : c'est d'obtenir "
                "des lecteurs, puis de les compter."
            )
        if not ready:
            return (
                "L'audience actuelle reste sous le seuil de tous les canaux ouverts. "
                "Continuer à publier et mesurer avant d'ouvrir un compte."
            )
        return f"{len(ready)} canal/canaux atteignent leur seuil : {', '.join(o.mechanism.name for o in ready)}."


def build_state(
    publications: int,
    generations: int,
    themes: Sequence[str],
    monthly_audience: Optional[int] = None,
    fully_automated: bool = True,
) -> SiteState:
    """Fabrique l'état du site à partir des données déjà collectées par le pipeline."""
    return SiteState(
        publications=publications,
        generations=generations,
        monthly_audience=monthly_audience,
        fully_automated=fully_automated,
        themes=[t for t in themes][:8],
    )


__all__ = [
    "Strategist",
    "SiteState",
    "Opportunity",
    "Mechanism",
    "MECHANISMS",
    "FISCAL_NOTES",
    "build_state",
    "COMPATIBLE",
    "RESTREINT",
    "INCOMPATIBLE",
]
