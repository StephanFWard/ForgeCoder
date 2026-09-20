"""POST /v1/decide — System One decisions, Jev-shaped, free by default.

The endpoint accepts exactly what the hosted Jev API accepts (``state`` plus
typed ``questions``) and answers in the same shape (``answers``, ``usage``),
with two ForgeCoder additions: ``backend`` and ``free``, so a caller can see
whether the numbers came from offline arithmetic or from the billed model.

Backends are chosen per request (``{"backend": "local"}``) or by configuration
(``FORGECODER_SYSTEM_ONE_BACKEND``). The paid Jev backend additionally needs
``FORGECODER_SYSTEM_ONE_ALLOW_PAID=1`` and a key, so a ForgeCoder install can
never start billing by accident.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from core.system_one.backends import DEFAULT_BACKEND, BackendUnavailable, backend_catalog
from core.system_one.decide import Decider
from core.system_one.primitives import DecisionRequest
from forge_server.main import AppState, get_state
from forge_server.security import check_permission

router = APIRouter(prefix="/v1", tags=["decide"])

_ALLOW_PAID_ENV = "FORGECODER_SYSTEM_ONE_ALLOW_PAID"
_TRUTHY = {"1", "true", "yes", "on"}


def _allow_paid() -> bool:
    """Read the paid-backend opt-in at request time (env changes stay testable)."""
    return os.environ.get(_ALLOW_PAID_ENV, "").strip().lower() in _TRUTHY


@router.get("/decide/backends")
async def backends() -> dict:
    """List the decision backends, their cost, and what each one needs."""
    return {
        "ok": True,
        "default": DEFAULT_BACKEND,
        "backends": backend_catalog(),
        "allow_paid": _allow_paid(),
    }


@router.post("/decide")
async def decide(req: DecisionRequest, state: AppState = Depends(get_state)) -> dict:
    """Answer typed questions about a state; local and free unless asked otherwise."""
    check_permission("SEARCH")
    decider = Decider(backend=req.backend, client=state.inference, allow_paid=_allow_paid())
    try:
        decision = await decider.adecide(req.state, req.questions)
    except BackendUnavailable as exc:
        return {"ok": False, "error": str(exc), "backends": backend_catalog()}
    return {"ok": True, **decision.model_dump()}
