from flask import Blueprint, request, jsonify, current_app

bp_broadcast = Blueprint("broadcast", __name__, url_prefix="/api")


@bp_broadcast.route("/broadcast/model", methods=["GET"]) 
def broadcast_model():
    weights = current_app.config.get("GLOBAL_WEIGHTS")
    vm = current_app.config["VERSION_MANAGER"]
    if not weights:
        return jsonify({"error": "no model available yet", "model_version": vm.versions.model_version}), 404
    # Encode weights for transport
    encoded = current_app.config["AGGREGATOR"].encode_weights(weights)
    return jsonify({
        "model_version": vm.versions.model_version,
        "weights": encoded
    }), 200


@bp_broadcast.route("/broadcast/signatures", methods=["GET"]) 
def broadcast_signatures():
    ss = current_app.config["SIGNATURE_STORE"]
    since = int(request.args.get("since", -1))
    if since >= 0:
        return jsonify(ss.get_updates_since(since)), 200
    return jsonify(ss.all()), 200
