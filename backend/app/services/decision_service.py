from ..constants import DECISION_ACCESS_GRANTED, DECISION_ACCESS_DENIED

class DecisionService:
    def decide(self, liveness: dict, identity: dict | None = None) -> dict:
        if not liveness.get("is_live", False):
            return {"access_granted": False, "decision": DECISION_ACCESS_DENIED}
        if identity is not None and not identity.get("verified", False):
            return {"access_granted": False, "decision": DECISION_ACCESS_DENIED}
        return {"access_granted": True, "decision": DECISION_ACCESS_GRANTED}
