"""Client LLM unifié (Groq — offre gratuite) avec disjoncteur *fail-safe*.

Ce module est le **seul** point d'accès au réseau LLM du projet. Il garantit :

  * **Aucune clé en dur** : la clé vient de `settings.groq_api_key` (env/secret).
  * **Aucune boucle infinie** : nombre de tentatives borné, backoff exponentiel
    plafonné, et un *disjoncteur* qui coupe définitivement les appels pour le
    reste du cycle après trop d'échecs (ou dès le premier dépassement de quota).
  * **Dégradation propre** : si le LLM est indisponible, `complete()` renvoie
    `None` et les appelants basculent sur une stratégie locale déterministe.
    Le pipeline n'échoue jamais à cause du LLM.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

try:  # Le SDK est optionnel : son absence ne doit pas casser l'import du projet.
    from groq import Groq

    _GROQ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - dépendance absente
    Groq = None  # type: ignore[assignment]
    _GROQ_IMPORT_ERROR = exc


#: Nombre d'échecs consécutifs au-delà duquel on coupe les appels du cycle.
MAX_CONSECUTIVE_FAILURES = 3

#: Plafond de backoff (s) — on ne bloque jamais un job CI très longtemps.
MAX_BACKOFF_SECONDS = 8.0


class LLMClient:
    """Enveloppe minimaliste et défensive autour de l'API Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.model = model or settings.groq_model
        self.timeout = timeout if timeout is not None else settings.llm_timeout
        self.max_retries = (
            max_retries if max_retries is not None else settings.llm_max_retries
        )
        self._api_key = api_key if api_key is not None else settings.groq_api_key
        self._failures = 0
        self._circuit_open = False
        self._client = self._build_client()
        #: Compteur d'appels réussis — utile pour les logs et les tests.
        self.calls_made = 0

    # -- Construction --------------------------------------------------------

    def _build_client(self):
        if not self._api_key:
            logger.warning(
                "GROQ_API_KEY absente — mode dégradé hors-ligne "
                "(génération et mutations déterministes locales)."
            )
            return None
        if Groq is None:
            logger.warning(
                "SDK groq indisponible (%s) — mode dégradé hors-ligne.",
                _GROQ_IMPORT_ERROR,
            )
            return None
        try:
            # `max_retries=0` : la logique de retry est gérée ici, pour garder
            # le contrôle du backoff et du disjoncteur.
            return Groq(api_key=self._api_key, timeout=self.timeout, max_retries=0)
        except Exception as exc:  # pragma: no cover - init SDK défaillante
            logger.warning("Initialisation du client Groq impossible : %s", exc)
            return None

    # -- État ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True si un appel réseau peut encore être tenté durant ce cycle."""
        return self._client is not None and not self._circuit_open

    def _trip_circuit(self, reason: str) -> None:
        """Ouvre le disjoncteur : plus aucun appel LLM jusqu'à la fin du cycle."""
        if not self._circuit_open:
            self._circuit_open = True
            logger.warning("Disjoncteur LLM ouvert (%s) — bascule hors-ligne.", reason)

    # -- Appel ---------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
    ) -> Optional[str]:
        """Renvoie la complétion du modèle, ou `None` si le LLM est indisponible.

        Ne lève jamais d'exception : c'est un contrat volontaire, l'appelant
        doit pouvoir enchaîner sans try/except.
        """
        if not self.available:
            return None

        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=max(0.0, min(2.0, float(temperature))),
                    max_tokens=max_tokens,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("réponse vide du modèle")
                self._failures = 0
                self.calls_made += 1
                return content

            except Exception as exc:
                name = type(exc).__name__
                # Quota / rate limit : on n'insiste pas, c'est le respect des
                # limites d'usage de l'API (obligation contractuelle CGU).
                if "RateLimit" in name or "429" in str(exc):
                    self._trip_circuit("limite de requêtes atteinte (429)")
                    return None
                # Authentification invalide : réessayer est inutile.
                if "Authentication" in name or "401" in str(exc):
                    self._trip_circuit("clé API refusée (401)")
                    return None

                self._failures += 1
                logger.warning(
                    "Appel LLM en échec (%s : %s) — tentative %d/%d",
                    name,
                    exc,
                    attempt + 1,
                    attempts,
                )
                if self._failures >= MAX_CONSECUTIVE_FAILURES:
                    self._trip_circuit("trop d'échecs consécutifs")
                    return None
                if attempt < attempts - 1:
                    time.sleep(min(MAX_BACKOFF_SECONDS, 1.5 * (2**attempt)))

        return None


#: Client partagé, construit paresseusement pour rester testable.
_shared_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Retourne le client LLM partagé du processus."""
    global _shared_client
    if _shared_client is None:
        _shared_client = LLMClient()
    return _shared_client


def reset_llm_client() -> None:
    """Réinitialise le client partagé (utilisé par les tests)."""
    global _shared_client
    _shared_client = None
