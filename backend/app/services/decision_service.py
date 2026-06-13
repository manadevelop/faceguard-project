from ..constants import DECISION_ACCESS_GRANTED, DECISION_ACCESS_DENIED


# Servicio que combina liveness e identidad para emitir la decisión final.
class DecisionService:
    def decide(self, liveness: dict, identity: dict | None = None) -> dict:
        # Deniega el acceso si la muestra no supera la prueba de vida.
        if not liveness.get("is_live", False):
            return {"access_granted": False, "decision": DECISION_ACCESS_DENIED}

        # Deniega el acceso si se evaluó identidad y no fue verificada.
        if identity is not None and not identity.get("verified", False):
            return {"access_granted": False, "decision": DECISION_ACCESS_DENIED}

        # Concede el acceso si la muestra es viva y la identidad es válida o no requerida.
        return {"access_granted": True, "decision": DECISION_ACCESS_GRANTED}