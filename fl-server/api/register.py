from flask import Blueprint, request, jsonify, current_app

bp_register = Blueprint("register", __name__, url_prefix="/api")


@bp_register.route("/register", methods=["POST"])
def register():
    payload = request.get_json(force=True, silent=True) or {}
    agent_id = payload.get("agent_id")
    if not agent_id:
        return jsonify({"error": "agent_id required"}), 400

    registry = current_app.config.setdefault("REGISTRY", {})
    registry[agent_id] = {
        "capabilities": payload.get("capabilities", {}),
        "last_seen": current_app.config.get("_now", 0)
    }
    vm = current_app.config["VERSION_MANAGER"]
    return jsonify({
        "status": "registered",
        "agent_id": agent_id,
        "model_version": vm.versions.model_version,
        "signature_version": vm.versions.signature_version
    }), 200
