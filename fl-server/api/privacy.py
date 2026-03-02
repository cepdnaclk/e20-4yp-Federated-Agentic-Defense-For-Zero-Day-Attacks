"""
Privacy Metrics API Blueprint

Provides REST API endpoints for accessing privacy metrics and visualizations.
"""

import io
import os
import base64
from flask import Blueprint, jsonify, current_app, send_file

bp_privacy = Blueprint("privacy", __name__, url_prefix="/api/privacy")


@bp_privacy.route("/summary", methods=["GET"])
def get_privacy_summary():
    """Get summary of privacy metrics across all rounds"""
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    summary = collector.get_privacy_summary()
    return jsonify(summary), 200


@bp_privacy.route("/rounds", methods=["GET"])
def get_all_rounds():
    """Get metrics for all federation rounds"""
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    metrics = collector.get_all_metrics()
    return jsonify({
        "total_rounds": len(metrics),
        "rounds": metrics
    }), 200


@bp_privacy.route("/rounds/<int:round_id>", methods=["GET"])
def get_round_metrics(round_id: int):
    """Get metrics for a specific round"""
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    for metrics in collector.get_all_metrics():
        if metrics.get("round_id") == round_id:
            return jsonify(metrics), 200
    
    return jsonify({"error": f"Round {round_id} not found"}), 404


@bp_privacy.route("/budget", methods=["GET"])
def get_privacy_budget():
    """Get current privacy budget status"""
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    return jsonify({
        "cumulative_epsilon": collector._cumulative_epsilon,
        "target_epsilon": collector.target_epsilon,
        "delta": collector.target_delta,
        "budget_consumed_percent": (collector._cumulative_epsilon / collector.target_epsilon) * 100,
        "noise_multiplier": collector.noise_multiplier,
        "clip_norm": collector.clip_norm
    }), 200


@bp_privacy.route("/report", methods=["GET"])
def get_privacy_report():
    """Generate and return a text-based privacy report"""
    from privacy.privacy_analyzer import PrivacyAnalyzer
    
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    analyzer = PrivacyAnalyzer(log_path=collector.log_path)
    analyzer.load_metrics(collector.get_all_metrics())
    
    report = analyzer.generate_report()
    return report, 200, {"Content-Type": "text/plain; charset=utf-8"}


@bp_privacy.route("/visualize/<viz_type>", methods=["GET"])
def get_visualization(viz_type: str):
    """
    Generate and return a privacy visualization as PNG.
    
    viz_type options:
    - budget: Privacy budget consumption
    - leakage: Data leakage risk
    - health: Federation health metrics
    - heatmap: Agent privacy heatmap
    - all: Generate all visualizations (returns JSON with paths)
    """
    from privacy.privacy_analyzer import PrivacyAnalyzer
    
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return jsonify({"error": "matplotlib not installed"}), 500
    
    analyzer = PrivacyAnalyzer(log_path=collector.log_path)
    analyzer.load_metrics(collector.get_all_metrics())
    
    if not analyzer.rounds_data:
        return jsonify({"error": "No metrics data available"}), 404
    
    viz_functions = {
        "budget": analyzer.plot_privacy_budget,
        "leakage": analyzer.plot_leakage_risk,
        "health": analyzer.plot_federation_health,
        "heatmap": analyzer.plot_agent_privacy_heatmap,
    }
    
    if viz_type == "all":
        # Generate all visualizations and return paths
        for name, func in viz_functions.items():
            func(save=True, show=False)
        
        return jsonify({
            "status": "success",
            "output_path": analyzer.output_path,
            "visualizations": list(viz_functions.keys())
        }), 200
    
    if viz_type not in viz_functions:
        return jsonify({
            "error": f"Unknown visualization type: {viz_type}",
            "available": list(viz_functions.keys()) + ["all"]
        }), 400
    
    # Generate visualization
    fig = viz_functions[viz_type](save=False, show=False)
    
    if fig is None:
        return jsonify({"error": "Failed to generate visualization"}), 500
    
    # Convert to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return send_file(
        buf,
        mimetype='image/png',
        as_attachment=False,
        download_name=f"privacy_{viz_type}.png"
    )


@bp_privacy.route("/export/csv", methods=["GET"])
def export_csv():
    """Export all metrics to CSV"""
    from privacy.privacy_analyzer import PrivacyAnalyzer
    
    collector = current_app.config.get("PRIVACY_COLLECTOR")
    if not collector:
        return jsonify({"error": "Privacy collector not configured"}), 500
    
    analyzer = PrivacyAnalyzer(log_path=collector.log_path)
    analyzer.load_metrics(collector.get_all_metrics())
    
    filepath = analyzer.export_to_csv()
    
    if os.path.exists(filepath):
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name='privacy_metrics.csv'
        )
    
    return jsonify({"error": "Failed to export CSV"}), 500
