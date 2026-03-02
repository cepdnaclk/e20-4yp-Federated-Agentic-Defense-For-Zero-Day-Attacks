"""
Privacy Analyzer and Visualizer for Federated Learning

Provides analysis and visualization of privacy metrics including:
- Privacy budget consumption over time
- Data leakage risk heatmaps
- Agent participation and privacy scores
- Model convergence with privacy guarantees
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Plotting imports with fallback
try:
    import matplotlib
    # Use Agg backend for headless environments (no GUI required)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class PrivacyAnalyzer:
    """
    Analyzes and visualizes privacy metrics from federation rounds.
    
    Usage:
        analyzer = PrivacyAnalyzer("./privacy_logs")
        analyzer.load_metrics()
        analyzer.plot_privacy_dashboard()
        analyzer.generate_report()
    """
    
    def __init__(self, log_path: str = "./privacy_logs"):
        self.log_path = log_path
        self.rounds_data: List[Dict] = []
        self.output_path = os.path.join(log_path, "visualizations")
        os.makedirs(self.output_path, exist_ok=True)
    
    def load_metrics(self, metrics_list: Optional[List[Dict]] = None) -> int:
        """
        Load metrics from files or from provided list.
        
        Returns:
            Number of rounds loaded
        """
        if metrics_list is not None:
            self.rounds_data = metrics_list
            return len(metrics_list)
        
        self.rounds_data = []
        if not os.path.exists(self.log_path):
            return 0
        
        for filename in sorted(os.listdir(self.log_path)):
            if filename.startswith("privacy_round_") and filename.endswith(".json"):
                filepath = os.path.join(self.log_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    self.rounds_data.append(json.load(f))
        
        return len(self.rounds_data)
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for visualization. Install with: pip install matplotlib")
            return False
        return True
    
    def plot_privacy_budget(self, save: bool = True, show: bool = True) -> Optional[Figure]:
        """
        Plot privacy budget consumption over federation rounds.
        Shows epsilon accumulation and remaining budget.
        """
        if not self._check_matplotlib() or not self.rounds_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Privacy Budget Analysis", fontsize=14, fontweight='bold')
        
        rounds = [r["round_id"] for r in self.rounds_data]
        epsilons = [r["epsilon"] for r in self.rounds_data]
        cumulative = [r["cumulative_epsilon"] for r in self.rounds_data]
        noise_scales = [r["noise_scale"] for r in self.rounds_data]
        
        # Get target epsilon from first round or default
        target_eps = self.rounds_data[-1].get("cumulative_epsilon", 10) * 1.5 if self.rounds_data else 10
        
        # Plot 1: Epsilon per round
        ax1 = axes[0, 0]
        ax1.bar(rounds, epsilons, color='steelblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel("Federation Round")
        ax1.set_ylabel("Epsilon (ε)")
        ax1.set_title("Privacy Budget Spent Per Round")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative epsilon with budget line
        ax2 = axes[0, 1]
        ax2.fill_between(rounds, cumulative, alpha=0.3, color='coral')
        ax2.plot(rounds, cumulative, 'o-', color='crimson', linewidth=2, markersize=8)
        ax2.axhline(y=target_eps, color='darkred', linestyle='--', linewidth=2, label=f'Target Budget: {target_eps:.1f}')
        ax2.set_xlabel("Federation Round")
        ax2.set_ylabel("Cumulative Epsilon")
        ax2.set_title("Privacy Budget Consumption Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Noise scale applied
        ax3 = axes[1, 0]
        ax3.plot(rounds, noise_scales, 's-', color='forestgreen', linewidth=2, markersize=8)
        ax3.fill_between(rounds, noise_scales, alpha=0.2, color='green')
        ax3.set_xlabel("Federation Round")
        ax3.set_ylabel("Noise Scale (σ)")
        ax3.set_title("Noise Multiplier Applied Per Round")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Budget remaining (gauge-style)
        ax4 = axes[1, 1]
        budget_used = cumulative[-1] if cumulative else 0
        budget_remaining = max(0, target_eps - budget_used)
        
        colors = ['#ff6b6b', '#4ecdc4']
        sizes = [budget_used, budget_remaining]
        labels = [f'Used: {budget_used:.2f}', f'Remaining: {budget_remaining:.2f}']
        
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0)
        )
        ax4.set_title(f"Privacy Budget Status\n(Target ε = {target_eps:.1f})")
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_path, "privacy_budget.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_leakage_risk(self, save: bool = True, show: bool = True) -> Optional[Figure]:
        """
        Plot data leakage risk metrics over time.
        Shows gradient similarity, exposure risk, and weight update patterns.
        """
        if not self._check_matplotlib() or not self.rounds_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Data Leakage Risk Analysis", fontsize=14, fontweight='bold')
        
        rounds = [r["round_id"] for r in self.rounds_data]
        grad_sim = [r["gradient_similarity"] for r in self.rounds_data]
        exposure = [r["information_exposure_risk"] for r in self.rounds_data]
        magnitude = [r["weight_update_magnitude"] for r in self.rounds_data]
        sparsity = [r["weight_update_sparsity"] for r in self.rounds_data]
        
        # Plot 1: Information Exposure Risk
        ax1 = axes[0, 0]
        colors = ['green' if e < 0.3 else 'orange' if e < 0.6 else 'red' for e in exposure]
        ax1.bar(rounds, exposure, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Low Risk')
        ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High Risk')
        ax1.set_xlabel("Federation Round")
        ax1.set_ylabel("Exposure Risk Score")
        ax1.set_title("Information Exposure Risk")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient Similarity (indicator of memorization)
        ax2 = axes[0, 1]
        ax2.plot(rounds, grad_sim, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.fill_between(rounds, grad_sim, alpha=0.2, color='purple')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High Similarity Warning')
        ax2.set_xlabel("Federation Round")
        ax2.set_ylabel("Gradient Similarity")
        ax2.set_title("Cross-Agent Gradient Similarity\n(Higher = potential data memorization)")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Weight Update Magnitude
        ax3 = axes[1, 0]
        ax3.bar(rounds, magnitude, color='teal', alpha=0.7, edgecolor='darkcyan')
        ax3.set_xlabel("Federation Round")
        ax3.set_ylabel("L2 Norm")
        ax3.set_title("Weight Update Magnitude")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Update Sparsity
        ax4 = axes[1, 1]
        ax4.plot(rounds, sparsity, 's-', color='darkorange', linewidth=2, markersize=8)
        ax4.fill_between(rounds, sparsity, alpha=0.2, color='orange')
        ax4.set_xlabel("Federation Round")
        ax4.set_ylabel("Sparsity (fraction non-zero)")
        ax4.set_title("Weight Update Sparsity\n(Lower = better privacy)")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_path, "leakage_risk.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_federation_health(self, save: bool = True, show: bool = True) -> Optional[Figure]:
        """
        Plot federation health metrics including participation and convergence.
        """
        if not self._check_matplotlib() or not self.rounds_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Federation Health & Privacy Metrics", fontsize=14, fontweight='bold')
        
        rounds = [r["round_id"] for r in self.rounds_data]
        participants = [r["participating_agents"] for r in self.rounds_data]
        samples = [r["total_samples"] for r in self.rounds_data]
        convergence = [r["model_convergence_delta"] for r in self.rounds_data]
        signatures = [r["signature_count"] for r in self.rounds_data]
        zero_days = [r["zero_day_candidates_found"] for r in self.rounds_data]
        bytes_tx = [r["bytes_transmitted"] / 1024 for r in self.rounds_data]  # KB
        
        # Plot 1: Participation
        ax1 = axes[0, 0]
        ax1.bar(rounds, participants, color='royalblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel("Federation Round")
        ax1.set_ylabel("Number of Agents")
        ax1.set_title("Agent Participation Per Round")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model Convergence
        ax2 = axes[0, 1]
        ax2.plot(rounds, convergence, 'o-', color='darkgreen', linewidth=2, markersize=8)
        ax2.fill_between(rounds, convergence, alpha=0.2, color='green')
        ax2.set_xlabel("Federation Round")
        ax2.set_ylabel("Weight Change (L2)")
        ax2.set_title("Model Convergence Progress")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Signatures and Zero-Days
        ax3 = axes[1, 0]
        x = np.arange(len(rounds))
        width = 0.35
        ax3.bar(x - width/2, signatures, width, label='Signatures Shared', color='steelblue', alpha=0.7)
        ax3.bar(x + width/2, zero_days, width, label='Zero-Day Candidates', color='crimson', alpha=0.7)
        ax3.set_xlabel("Federation Round")
        ax3.set_ylabel("Count")
        ax3.set_title("Threat Intelligence Sharing")
        ax3.set_xticks(x)
        ax3.set_xticklabels(rounds)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data Transmission
        ax4 = axes[1, 1]
        ax4.fill_between(rounds, bytes_tx, alpha=0.3, color='coral')
        ax4.plot(rounds, bytes_tx, 'o-', color='crimson', linewidth=2, markersize=8)
        ax4.set_xlabel("Federation Round")
        ax4.set_ylabel("Data Transmitted (KB)")
        ax4.set_title("Communication Volume")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_path, "federation_health.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_agent_privacy_heatmap(self, save: bool = True, show: bool = True) -> Optional[Figure]:
        """
        Plot heatmap of per-agent privacy scores across rounds.
        """
        if not self._check_matplotlib() or not self.rounds_data:
            return None
        
        # Collect all unique agents
        all_agents = set()
        for r in self.rounds_data:
            all_agents.update(r.get("agent_privacy_scores", {}).keys())
        
        if not all_agents:
            print("No agent privacy scores available")
            return None
        
        agents = sorted(all_agents)
        rounds = [r["round_id"] for r in self.rounds_data]
        
        # Build matrix
        matrix = np.zeros((len(agents), len(rounds)))
        for j, r in enumerate(self.rounds_data):
            scores = r.get("agent_privacy_scores", {})
            for i, agent in enumerate(agents):
                matrix[i, j] = scores.get(agent, np.nan)
        
        fig, ax = plt.subplots(figsize=(max(10, len(rounds) * 0.8), max(6, len(agents) * 0.5)))
        
        # Use masked array for NaN values
        masked_matrix = np.ma.masked_invalid(matrix)
        
        im = ax.imshow(masked_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        ax.set_xlabel("Federation Round")
        ax.set_ylabel("Agent")
        ax.set_title("Agent Privacy Score Heatmap\n(Green = Better Privacy, Red = Higher Exposure Risk)")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Privacy Score (lower is better)")
        
        # Add text annotations
        for i in range(len(agents)):
            for j in range(len(rounds)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                   ha="center", va="center", 
                                   color="white" if matrix[i, j] > 0.5 else "black",
                                   fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_path, "agent_privacy_heatmap.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_privacy_dashboard(self, save: bool = True, show: bool = True) -> None:
        """
        Generate all privacy visualization plots.
        """
        if not self.rounds_data:
            print("No metrics data loaded. Call load_metrics() first.")
            return
        
        print(f"\n{'='*60}")
        print("PRIVACY METRICS VISUALIZATION DASHBOARD")
        print(f"{'='*60}")
        print(f"Analyzing {len(self.rounds_data)} federation rounds...")
        print()
        
        self.plot_privacy_budget(save=save, show=show)
        self.plot_leakage_risk(save=save, show=show)
        self.plot_federation_health(save=save, show=show)
        self.plot_agent_privacy_heatmap(save=save, show=show)
        
        print(f"\nAll visualizations saved to: {self.output_path}")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a text-based privacy analysis report.
        """
        if not self.rounds_data:
            return "No metrics data available."
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("FEDERATED LEARNING PRIVACY METRICS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Summary Statistics
        report_lines.append("1. SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        cumulative_eps = self.rounds_data[-1]["cumulative_epsilon"]
        avg_exposure = np.mean([r["information_exposure_risk"] for r in self.rounds_data])
        avg_similarity = np.mean([r["gradient_similarity"] for r in self.rounds_data])
        total_sigs = sum(r["signature_count"] for r in self.rounds_data)
        total_bytes = sum(r["bytes_transmitted"] for r in self.rounds_data)
        total_zero_days = sum(r["zero_day_candidates_found"] for r in self.rounds_data)
        
        report_lines.append(f"  Total Federation Rounds: {len(self.rounds_data)}")
        report_lines.append(f"  Cumulative Privacy Budget (ε): {cumulative_eps:.4f}")
        report_lines.append(f"  Average Exposure Risk: {avg_exposure:.4f}")
        report_lines.append(f"  Average Gradient Similarity: {avg_similarity:.4f}")
        report_lines.append(f"  Total Signatures Shared: {total_sigs}")
        report_lines.append(f"  Total Data Transmitted: {total_bytes / 1024:.2f} KB")
        report_lines.append(f"  Zero-Day Candidates Found: {total_zero_days}")
        report_lines.append("")
        
        # Privacy Budget Analysis
        report_lines.append("2. PRIVACY BUDGET ANALYSIS")
        report_lines.append("-" * 40)
        eps_values = [r["epsilon"] for r in self.rounds_data]
        report_lines.append(f"  Min ε per round: {min(eps_values):.4f}")
        report_lines.append(f"  Max ε per round: {max(eps_values):.4f}")
        report_lines.append(f"  Avg ε per round: {np.mean(eps_values):.4f}")
        report_lines.append(f"  Std ε per round: {np.std(eps_values):.4f}")
        
        # Estimate rounds remaining
        avg_eps = np.mean(eps_values)
        target_eps = cumulative_eps * 1.5  # Assume target is 1.5x current
        if avg_eps > 0:
            rounds_remaining = int((target_eps - cumulative_eps) / avg_eps)
            report_lines.append(f"  Estimated rounds until budget exhaustion: {rounds_remaining}")
        report_lines.append("")
        
        # Risk Assessment
        report_lines.append("3. PRIVACY RISK ASSESSMENT")
        report_lines.append("-" * 40)
        
        # Categorize risk levels
        high_risk_rounds = [r for r in self.rounds_data if r["information_exposure_risk"] > 0.6]
        medium_risk_rounds = [r for r in self.rounds_data if 0.3 <= r["information_exposure_risk"] <= 0.6]
        low_risk_rounds = [r for r in self.rounds_data if r["information_exposure_risk"] < 0.3]
        
        report_lines.append(f"  High Risk Rounds: {len(high_risk_rounds)} ({len(high_risk_rounds)/len(self.rounds_data)*100:.1f}%)")
        report_lines.append(f"  Medium Risk Rounds: {len(medium_risk_rounds)} ({len(medium_risk_rounds)/len(self.rounds_data)*100:.1f}%)")
        report_lines.append(f"  Low Risk Rounds: {len(low_risk_rounds)} ({len(low_risk_rounds)/len(self.rounds_data)*100:.1f}%)")
        
        # Identify concerning patterns
        if avg_similarity > 0.7:
            report_lines.append("  ⚠ WARNING: High gradient similarity detected - potential data memorization")
        if avg_exposure > 0.5:
            report_lines.append("  ⚠ WARNING: High exposure risk - consider increasing noise multiplier")
        report_lines.append("")
        
        # Per-Round Details
        report_lines.append("4. PER-ROUND DETAILS")
        report_lines.append("-" * 40)
        report_lines.append(f"  {'Round':<8} {'ε':<10} {'Exposure':<10} {'Agents':<8} {'Sigs':<8}")
        report_lines.append("  " + "-" * 44)
        for r in self.rounds_data[-10:]:  # Last 10 rounds
            report_lines.append(
                f"  {r['round_id']:<8} {r['epsilon']:<10.4f} {r['information_exposure_risk']:<10.4f} "
                f"{r['participating_agents']:<8} {r['signature_count']:<8}"
            )
        if len(self.rounds_data) > 10:
            report_lines.append(f"  ... (showing last 10 of {len(self.rounds_data)} rounds)")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("5. RECOMMENDATIONS")
        report_lines.append("-" * 40)
        recommendations = []
        
        if avg_exposure > 0.5:
            recommendations.append("• Increase noise_multiplier to reduce information exposure")
        if avg_similarity > 0.7:
            recommendations.append("• Consider local DP at the client level before aggregation")
        if cumulative_eps > target_eps * 0.8:
            recommendations.append("• Privacy budget running low - consider reducing training rounds")
        if np.mean([r["weight_update_sparsity"] for r in self.rounds_data]) > 0.9:
            recommendations.append("• High weight density - consider gradient sparsification")
        if total_bytes / len(self.rounds_data) > 100 * 1024:  # > 100KB average
            recommendations.append("• High communication volume - consider compression techniques")
        
        if not recommendations:
            recommendations.append("• Privacy metrics are within acceptable ranges")
            recommendations.append("• Continue monitoring for drift in metrics")
        
        for rec in recommendations:
            report_lines.append(f"  {rec}")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def export_to_csv(self, output_file: Optional[str] = None) -> str:
        """Export metrics to CSV format"""
        if not self.rounds_data:
            return ""
        
        output_file = output_file or os.path.join(self.output_path, "privacy_metrics.csv")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(self.rounds_data)
            df.to_csv(output_file, index=False)
        else:
            # Manual CSV export
            if not self.rounds_data:
                return output_file
            
            headers = list(self.rounds_data[0].keys())
            # Remove nested dict columns for simple CSV
            headers = [h for h in headers if h != "agent_privacy_scores"]
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for r in self.rounds_data:
                    values = [str(r.get(h, "")) for h in headers]
                    f.write(",".join(values) + "\n")
        
        print(f"Metrics exported to: {output_file}")
        return output_file
