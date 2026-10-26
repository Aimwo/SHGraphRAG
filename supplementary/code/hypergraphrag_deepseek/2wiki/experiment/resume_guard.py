"""No-credential guard copied into the new c100 experiment packages."""
from __future__ import annotations

import hashlib
import json


def require_conditional_release(coordination):
    folder = coordination / "conditional_resume_20261025"
    authorization = json.loads((folder / "authorization.json").read_text(encoding="utf-8"))
    if authorization.get("status") not in (
            "activated_after_verified_condition", "activated_by_current_user"):
        raise RuntimeError("Other-chat evaluation condition has not been activated")
    release = folder / "release.json"
    if not release.is_file():
        raise RuntimeError("Other-chat completion evidence is absent")
    digest = hashlib.sha256(release.read_bytes()).hexdigest()
    plan = json.loads((coordination / "campaign.json").read_text(encoding="utf-8"))
    evidence = json.loads(release.read_text(encoding="utf-8"))
    if (authorization.get("status") == "activated_by_current_user"
            and (authorization.get("mode") != "immediate_user_resume"
                 or evidence.get("resume_basis") != "explicit_current_user_immediate_resume")):
        raise RuntimeError("Immediate resume lacks its explicit current user authorization")
    if (digest != plan.get("conditional_release_sha256")
            or evidence.get("condition_completed") is not True
            or evidence.get("reference_chat") != "01a16c84-bb09-78b3-adde-be6601a96137"
            or evidence.get("questions") != 1000
            or authorization.get("max_combined_requests") != 100
            or plan.get("max_combined_requests") != 100):
        raise RuntimeError("Conditional c100 release evidence differs")
