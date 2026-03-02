"""
Privacy Visualizations for Federated IDS.

This module provides comprehensive visualization tools for
privacy metrics, attack simulations, and privacy-utility trade-offs.

Visualization Categories:
    1. Privacy Budget Evolution
    2. Attack Success Rate Analysis
    3. Privacy-Utility Trade-off Curves
    4. Gradient Leakage Maps
    5. Secure Aggregation Flow
    6. Comparative Analysis Dashboards

Output Formats:
    - Matplotlib figures (PNG, SVG, PDF)
    - Interactive Plotly dashboards
    - ASCII diagrams for terminal output
    - HTML reports with embedded charts

Research Visualizations:
    - Novel adaptive clipping dynamics
    - Threat-aware privacy budget allocation
    - Multi-organization privacy boundaries
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

# Plotting imports - handle missing dependencies gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for privacy visualizations."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: List[str] = None
    save_format: str = "png"
    interactive: bool = True
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                "#2ecc71",  # Green - safe
                "#f39c12",  # Orange - warning
                "#e74c3c",  # Red - danger
                "#3498db",  # Blue - info
                "#9b59b6",  # Purple - highlight
                "#1abc9c",  # Teal - secondary
            ]


class PrivacyVisualizer:
    """
    Comprehensive Privacy Visualization Suite.
    
    Provides publication-quality visualizations for privacy research.
    
    Features:
        - Privacy budget evolution over training
        - Attack success rate comparisons
        - Privacy-utility Pareto frontiers
        - Method comparison heatmaps
        - Interactive dashboards
    
    Example:
        >>> visualizer = PrivacyVisualizer(output_dir="./visualizations")
        >>> 
        >>> # Plot privacy budget over time
        >>> visualizer.plot_epsilon_evolution(budget_history)
        >>> 
        >>> # Create comparison dashboard
        >>> visualizer.create_privacy_dashboard(metrics_dict)
    """
    
    def __init__(
        self,
        output_dir: str = "./privacy_visualizations",
        config: Optional[VisualizationConfig] = None,
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures.
            config: Visualization configuration.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style if available
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(self.config.style)
            except Exception:
                plt.style.use('default')
        
        logger.info(f"PrivacyVisualizer initialized: output_dir={output_dir}")
    
    def plot_epsilon_evolution(
        self,
        round_history: List[Dict[str, float]],
        title: str = "Privacy Budget Evolution",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Plot privacy budget (epsilon) evolution over training rounds.
        
        Args:
            round_history: List of dicts with 'round', 'epsilon_used', 'cumulative_epsilon'.
            title: Plot title.
            save_path: Optional save path.
        
        Returns:
            Figure object if matplotlib available.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available for plotting")
            return self._ascii_epsilon_plot(round_history)
        
        rounds = [r.get("round", i) for i, r in enumerate(round_history)]
        eps_per_round = [r.get("epsilon_used", 0) for r in round_history]
        cumulative_eps = [r.get("cumulative_epsilon", 0) for r in round_history]
        remaining = [r.get("remaining_epsilon", 0) for r in round_history]
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # Plot 1: Epsilon per round
        ax1 = axes[0, 0]
        ax1.bar(rounds, eps_per_round, color=self.config.color_palette[3], alpha=0.7)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("ε per Round")
        ax1.set_title("Privacy Budget Spent per Round")
        ax1.axhline(y=np.mean(eps_per_round), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(eps_per_round):.4f}')
        ax1.legend()
        
        # Plot 2: Cumulative epsilon
        ax2 = axes[0, 1]
        ax2.fill_between(rounds, cumulative_eps, alpha=0.3, color=self.config.color_palette[2])
        ax2.plot(rounds, cumulative_eps, color=self.config.color_palette[2], linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cumulative ε")
        ax2.set_title("Cumulative Privacy Loss")
        if remaining and remaining[0] > 0:
            total = cumulative_eps[-1] + remaining[-1]
            ax2.axhline(y=total, color='gray', linestyle=':', label=f'Total Budget: {total:.2f}')
            ax2.legend()
        
        # Plot 3: Remaining budget
        ax3 = axes[1, 0]
        if remaining:
            ax3.fill_between(rounds, remaining, alpha=0.3, color=self.config.color_palette[0])
            ax3.plot(rounds, remaining, color=self.config.color_palette[0], linewidth=2)
            ax3.set_xlabel("Round")
            ax3.set_ylabel("Remaining ε")
            ax3.set_title("Remaining Privacy Budget")
            
            # Add warning zone
            if remaining[0] > 0:
                warning_threshold = remaining[0] * 0.2
                ax3.axhline(y=warning_threshold, color='orange', linestyle='--',
                           label=f'Warning (20%): {warning_threshold:.2f}')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No remaining budget data", ha='center', va='center')
        
        # Plot 4: Budget utilization pie
        ax4 = axes[1, 1]
        if round_history and cumulative_eps:
            spent = max(0, cumulative_eps[-1])  # Ensure non-negative
            remain = max(0, remaining[-1] if remaining else 0)  # Ensure non-negative
            total = spent + remain
            
            if total > 0 and spent >= 0 and remain >= 0:
                sizes = [spent, remain]
                labels = [f'Spent\n(ε={spent:.2f})', f'Remaining\n(ε={remain:.2f})']
                colors = [self.config.color_palette[2], self.config.color_palette[0]]
                
                ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, explode=(0.05, 0))
                ax4.set_title("Budget Utilization")
            else:
                ax4.text(0.5, 0.5, "Invalid budget values", ha='center', va='center')
                ax4.set_title("Budget Utilization (Error)")
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved epsilon evolution plot to {save_path}")
        else:
            # Save to default location
            default_path = self.output_dir / f"epsilon_evolution.{self.config.save_format}"
            plt.savefig(default_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_privacy_utility_tradeoff(
        self,
        epsilons: List[float],
        accuracies: List[float],
        method_names: Optional[List[str]] = None,
        title: str = "Privacy-Utility Trade-off",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Plot privacy-utility trade-off curve.
        
        Args:
            epsilons: Privacy budget values.
            accuracies: Corresponding model accuracies.
            method_names: Optional method labels.
            title: Plot title.
            save_path: Optional save path.
        
        Returns:
            Figure object.
        """
        if not HAS_MATPLOTLIB:
            return self._ascii_tradeoff_plot(epsilons, accuracies)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sort by epsilon for smooth curve
        sorted_pairs = sorted(zip(epsilons, accuracies))
        eps_sorted = [e for e, _ in sorted_pairs]
        acc_sorted = [a for _, a in sorted_pairs]
        
        # Plot 1: Trade-off curve
        ax1 = axes[0]
        ax1.plot(eps_sorted, acc_sorted, 'o-', color=self.config.color_palette[3],
                linewidth=2, markersize=8, label='Observed')
        
        # Add trend line
        z = np.polyfit(epsilons, accuracies, 2)
        p = np.poly1d(z)
        eps_smooth = np.linspace(min(epsilons), max(epsilons), 100)
        ax1.plot(eps_smooth, p(eps_smooth), '--', color=self.config.color_palette[4],
                alpha=0.7, label='Trend')
        
        # Annotations for specific points
        if method_names:
            for eps, acc, name in zip(epsilons, accuracies, method_names):
                ax1.annotate(name, (eps, acc), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)
        
        # Privacy regions
        ax1.axvspan(0, 1, alpha=0.1, color='green', label='High Privacy (ε≤1)')
        ax1.axvspan(1, 5, alpha=0.1, color='orange', label='Medium Privacy (1<ε≤5)')
        ax1.axvspan(5, max(epsilons) * 1.1, alpha=0.1, color='red', label='Low Privacy (ε>5)')
        
        ax1.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax1.set_ylabel("Model Accuracy", fontsize=12)
        ax1.set_title("Privacy-Utility Trade-off Curve")
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Utility loss vs privacy cost
        ax2 = axes[1]
        max_acc = max(accuracies)
        utility_loss = [(max_acc - a) * 100 for a in acc_sorted]
        
        ax2.bar(range(len(eps_sorted)), utility_loss, 
               color=self.config.color_palette[1], alpha=0.7)
        ax2.set_xticks(range(len(eps_sorted)))
        ax2.set_xticklabels([f'ε={e:.1f}' for e in eps_sorted], rotation=45)
        ax2.set_xlabel("Privacy Budget", fontsize=12)
        ax2.set_ylabel("Accuracy Drop (%)", fontsize=12)
        ax2.set_title("Utility Cost of Privacy")
        
        # Add efficiency ratio
        for i, (ul, eps) in enumerate(zip(utility_loss, eps_sorted)):
            if eps > 0:
                ratio = ul / np.log(eps + 1)  # Privacy efficiency
                ax2.annotate(f'{ratio:.1f}', (i, ul), textcoords="offset points",
                           xytext=(0, 5), ha='center', fontsize=7, color='gray')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_path or self.output_dir / f"privacy_utility_tradeoff.{self.config.save_format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved trade-off plot to {save_path}")
        
        return fig
    
    def plot_membership_inference_results(
        self,
        attack_results: Dict[str, Any],
        title: str = "Membership Inference Attack Analysis",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Visualize membership inference attack results.
        
        Args:
            attack_results: Results from MembershipInferenceAttack.evaluate().
            title: Plot title.
            save_path: Optional save path.
        
        Returns:
            Figure object.
        """
        if not HAS_MATPLOTLIB:
            return self._ascii_mia_plot(attack_results)
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # Plot 1: Score distributions
        ax1 = axes[0, 0]
        stats = attack_results.get("statistics", {})
        
        # Generate synthetic distributions for visualization
        member_mean = stats.get("member_score_mean", 0.6)
        member_std = stats.get("member_score_std", 0.1)
        non_member_mean = stats.get("non_member_score_mean", 0.4)
        non_member_std = stats.get("non_member_score_std", 0.1)
        
        x = np.linspace(0, 1, 1000)
        from scipy.stats import norm
        member_dist = norm.pdf(x, member_mean, max(0.01, member_std))
        non_member_dist = norm.pdf(x, non_member_mean, max(0.01, non_member_std))
        
        ax1.fill_between(x, member_dist, alpha=0.5, color=self.config.color_palette[0],
                        label='Members (Training)')
        ax1.fill_between(x, non_member_dist, alpha=0.5, color=self.config.color_palette[2],
                        label='Non-members (Test)')
        ax1.set_xlabel("Membership Score")
        ax1.set_ylabel("Density")
        ax1.set_title("Score Distributions")
        ax1.legend()
        
        # Plot 2: Attack success metrics
        ax2 = axes[0, 1]
        threshold_auc = attack_results.get("threshold_attack", {}).get("auc", 0.5)
        classifier_auc = attack_results.get("classifier_attack", {}).get("auc", 0.5)
        
        metrics = ['Threshold\nAttack', 'Classifier\nAttack', 'Random\nGuess']
        aucs = [threshold_auc, classifier_auc, 0.5]
        colors = [self.config.color_palette[2] if a > 0.6 else self.config.color_palette[0] 
                 for a in aucs]
        
        bars = ax2.bar(metrics, aucs, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline')
        ax2.axhline(y=0.7, color='orange', linestyle=':', label='Concern Threshold')
        ax2.set_ylabel("AUC Score")
        ax2.set_title("Attack Success Rates")
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Privacy risk gauge
        ax3 = axes[1, 0]
        best_auc = attack_results.get("combined", {}).get("best_auc", 0.5)
        concern = attack_results.get("combined", {}).get("privacy_concern", "UNKNOWN")
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax3.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
        
        # Colored sections
        colors_gauge = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
        for i in range(5):
            start = i * np.pi / 5
            end = (i + 1) * np.pi / 5
            t = np.linspace(start, end, 20)
            ax3.fill_between(r * np.cos(t), 0, r * np.sin(t), 
                            alpha=0.3, color=colors_gauge[i])
        
        # Needle
        needle_angle = np.pi * (1 - (best_auc - 0.5) * 2)  # Map 0.5-1.0 to pi-0
        needle_angle = max(0, min(np.pi, needle_angle))
        ax3.plot([0, 0.8 * np.cos(needle_angle)], [0, 0.8 * np.sin(needle_angle)],
                'k-', linewidth=3)
        ax3.plot(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle), 
                'ko', markersize=10)
        
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-0.1, 1.2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title(f"Privacy Risk: {concern}\n(AUC: {best_auc:.3f})")
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ["Metric", "Value", "Status"],
            ["Best Attack AUC", f"{best_auc:.3f}", 
             "⚠️" if best_auc > 0.6 else "✓"],
            ["Score Separation", f"{stats.get('score_separation', 0):.3f}", 
             "⚠️" if stats.get('score_separation', 0) > 0.1 else "✓"],
            ["Members Analyzed", str(attack_results.get('num_members', 0)), "—"],
            ["Non-members Analyzed", str(attack_results.get('num_non_members', 0)), "—"],
            ["Privacy Concern", concern, 
             "🔴" if concern in ["HIGH", "CRITICAL"] else "🟢"],
        ]
        
        table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         loc='center', cellLoc='center',
                         colColours=['lightgray']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_path or self.output_dir / f"mia_analysis.{self.config.save_format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved MIA analysis plot to {save_path}")
        
        return fig
    
    def plot_method_comparison(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Privacy Method Comparison",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Compare multiple privacy methods across metrics.
        
        Args:
            comparison_data: Dict mapping method_name -> {metric: value}.
            metrics: List of metrics to compare.
            title: Plot title.
            save_path: Optional save path.
        
        Returns:
            Figure object.
        """
        if not HAS_MATPLOTLIB:
            return self._ascii_comparison(comparison_data)
        
        methods = list(comparison_data.keys())
        if metrics is None:
            metrics = list(comparison_data[methods[0]].keys())
        
        # Create radar chart and bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart comparison
        ax1 = axes[0]
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            values = [comparison_data[method].get(m, 0) for m in metrics]
            offset = (i - len(methods)/2 + 0.5) * width
            bars = ax1.bar(x + offset, values, width, 
                          label=method, alpha=0.8)
        
        ax1.set_xlabel("Metric")
        ax1.set_ylabel("Value")
        ax1.set_title("Method Comparison by Metric")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Radar chart
        ax2 = axes[1]
        ax2 = plt.subplot(122, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, method in enumerate(methods):
            values = [comparison_data[method].get(m, 0) for m in metrics]
            # Normalize to 0-1 for radar
            max_vals = [max(comparison_data[m].get(metric, 1) 
                           for m in methods) for metric in metrics]
            normalized = [v/mv if mv > 0 else 0 for v, mv in zip(values, max_vals)]
            normalized += normalized[:1]
            
            ax2.plot(angles, normalized, 'o-', linewidth=2, 
                    label=method, alpha=0.7)
            ax2.fill(angles, normalized, alpha=0.1)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, fontsize=8)
        ax2.set_title("Normalized Comparison")
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_path or self.output_dir / f"method_comparison.{self.config.save_format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved method comparison plot to {save_path}")
        
        return fig
    
    def plot_secure_aggregation_flow(
        self,
        num_clients: int = 5,
        protocol: str = "pairwise_masking",
        title: str = "Secure Aggregation Protocol Flow",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Visualize secure aggregation protocol flow.
        
        Args:
            num_clients: Number of clients in diagram.
            protocol: Protocol type.
            title: Plot title.
            save_path: Optional save path.
        
        Returns:
            Figure object.
        """
        if not HAS_MATPLOTLIB:
            return self._ascii_aggregation_flow(num_clients, protocol)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Layout parameters
        client_y = 0.8
        server_y = 0.2
        step_height = 0.15
        
        # Draw clients
        client_positions = np.linspace(0.1, 0.9, num_clients)
        for i, x in enumerate(client_positions):
            circle = mpatches.Circle((x, client_y), 0.04, 
                                     facecolor=self.config.color_palette[3],
                                     edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, client_y + 0.08, f'Org {i+1}', ha='center', fontsize=10, fontweight='bold')
        
        # Draw server
        server_rect = mpatches.FancyBboxPatch((0.4, server_y - 0.05), 0.2, 0.1,
                                              boxstyle="round,pad=0.02",
                                              facecolor=self.config.color_palette[4],
                                              edgecolor='black', linewidth=2)
        ax.add_patch(server_rect)
        ax.text(0.5, server_y, 'Aggregation\nServer', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Draw protocol steps
        if protocol == "pairwise_masking":
            # Step 1: Pairwise secret exchange
            for i, x1 in enumerate(client_positions):
                for j, x2 in enumerate(client_positions):
                    if i < j:
                        ax.annotate("", xy=(x2, client_y - 0.06), 
                                   xytext=(x1, client_y - 0.06),
                                   arrowprops=dict(arrowstyle="<->", color='gray', 
                                                  linestyle=':', linewidth=1))
            ax.text(0.02, client_y - 0.02, "1. Secret\nExchange", fontsize=8, 
                   style='italic', color='gray')
            
            # Step 2: Masked updates to server
            for x in client_positions:
                ax.annotate("", xy=(0.5, server_y + 0.05), 
                           xytext=(x, client_y - 0.04),
                           arrowprops=dict(arrowstyle="->", color='green', linewidth=2))
            ax.text(0.02, 0.5, "2. Masked\nUpdates", fontsize=8, 
                   style='italic', color='green')
            
            # Step 3: Aggregated result
            ax.annotate("", xy=(0.5, server_y + 0.12), 
                       xytext=(0.5, server_y + 0.25),
                       arrowprops=dict(arrowstyle="->", color='purple', 
                                      linewidth=3, linestyle='--'))
            ax.text(0.55, server_y + 0.18, "3. Sum =\nTrue Aggregate", fontsize=8,
                   style='italic', color='purple')
        
        # Add legend
        legend_elements = [
            mpatches.Circle((0, 0), 0.05, facecolor=self.config.color_palette[3], label='Organization'),
            mpatches.Rectangle((0, 0), 0.1, 0.05, facecolor=self.config.color_palette[4], label='Server'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
        
        # Add protocol description
        desc = {
            "pairwise_masking": (
                "Pairwise Masking Protocol:\n"
                "• Clients agree on shared secrets pairwise\n"
                "• Each client masks update: x + Σmasks - Σopposite_masks\n"
                "• Server sums all masked updates\n"
                "• Masks cancel, revealing only the sum"
            ),
            "secret_sharing": (
                "Secret Sharing Protocol:\n"
                "• Each client splits update into n shares\n"
                "• Shares distributed to other clients\n"
                "• Server collects and reconstructs average\n"
                "• No individual update revealed"
            ),
        }
        ax.text(0.5, 0.02, desc.get(protocol, "Protocol visualization"),
               ha='center', va='bottom', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Save
        save_path = save_path or self.output_dir / f"secure_aggregation_flow.{self.config.save_format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved aggregation flow to {save_path}")
        
        return fig
    
    # ASCII fallbacks for environments without matplotlib
    def _ascii_epsilon_plot(self, round_history: List[Dict]) -> str:
        """Generate ASCII epsilon plot."""
        lines = ["=" * 50, "PRIVACY BUDGET EVOLUTION", "=" * 50]
        
        for r in round_history[-10:]:  # Last 10 rounds
            round_num = r.get("round", 0)
            eps = r.get("epsilon_used", 0)
            cum = r.get("cumulative_epsilon", 0)
            bar_len = int(eps * 100)
            bar = "█" * min(bar_len, 40)
            lines.append(f"R{round_num:3d}: {bar} ε={eps:.4f} (cum: {cum:.4f})")
        
        return "\n".join(lines)
    
    def _ascii_tradeoff_plot(self, epsilons: List[float], accuracies: List[float]) -> str:
        """Generate ASCII trade-off display."""
        lines = ["=" * 50, "PRIVACY-UTILITY TRADE-OFF", "=" * 50]
        lines.append(f"{'Epsilon':>10} | {'Accuracy':>10} | Visualization")
        lines.append("-" * 50)
        
        for eps, acc in sorted(zip(epsilons, accuracies)):
            bar = "█" * int(acc * 20)
            lines.append(f"{eps:>10.2f} | {acc:>10.4f} | {bar}")
        
        return "\n".join(lines)
    
    def _ascii_mia_plot(self, results: Dict) -> str:
        """Generate ASCII MIA results."""
        auc = results.get("combined", {}).get("best_auc", 0.5)
        concern = results.get("combined", {}).get("privacy_concern", "UNKNOWN")
        
        lines = [
            "=" * 50,
            "MEMBERSHIP INFERENCE ATTACK RESULTS",
            "=" * 50,
            f"Best Attack AUC: {auc:.4f}",
            f"Privacy Concern: {concern}",
            "",
            "Risk Level: " + "█" * int(auc * 20) + f" ({auc*100:.1f}%)",
        ]
        
        return "\n".join(lines)
    
    def _ascii_comparison(self, data: Dict) -> str:
        """Generate ASCII comparison table."""
        lines = ["=" * 60, "METHOD COMPARISON", "=" * 60]
        
        methods = list(data.keys())
        metrics = list(data[methods[0]].keys()) if methods else []
        
        # Header
        header = f"{'Metric':>20} | " + " | ".join(f"{m:>12}" for m in methods)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        for metric in metrics:
            values = [f"{data[m].get(metric, 0):>12.4f}" for m in methods]
            lines.append(f"{metric:>20} | " + " | ".join(values))
        
        return "\n".join(lines)
    
    def _ascii_aggregation_flow(self, num_clients: int, protocol: str) -> str:
        """Generate ASCII aggregation flow."""
        lines = [
            "=" * 50,
            f"SECURE AGGREGATION: {protocol.upper()}",
            "=" * 50,
            "",
            "  " + " ".join([f"[Org{i+1}]" for i in range(min(num_clients, 5))]),
            "     " + "  \\  |  /  " * (min(num_clients, 5) // 2),
            "        \\  |  /",
            "     [Aggregation Server]",
            "",
            "Protocol Steps:",
            "  1. Pairwise secret exchange",
            "  2. Masked updates sent to server",
            "  3. Server computes sum (masks cancel)",
            "  4. Only aggregate revealed",
        ]
        
        return "\n".join(lines)


def create_privacy_dashboard(
    metrics_data: Dict[str, Any],
    output_path: str = "privacy_dashboard.html",
) -> str:
    """
    Create interactive HTML privacy dashboard.
    
    Args:
        metrics_data: Comprehensive metrics dictionary.
        output_path: Output HTML file path.
    
    Returns:
        Path to generated dashboard.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available. Generating simple HTML report.")
        return _generate_simple_html_report(metrics_data, output_path)
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Privacy Budget Over Time",
            "Attack Success Rates",
            "Privacy-Utility Trade-off",
            "Method Comparison"
        ),
        specs=[[{}, {}], [{}, {"type": "polar"}]]
    )
    
    # Plot 1: Budget evolution
    if "budget_history" in metrics_data:
        rounds = [r.get("round", i) for i, r in enumerate(metrics_data["budget_history"])]
        epsilons = [r.get("cumulative_epsilon", 0) for r in metrics_data["budget_history"]]
        
        fig.add_trace(
            go.Scatter(x=rounds, y=epsilons, mode='lines+markers', name='ε spent'),
            row=1, col=1
        )
    
    # Plot 2: Attack rates
    if "attack_results" in metrics_data:
        attacks = metrics_data["attack_results"]
        fig.add_trace(
            go.Bar(
                x=['Threshold', 'Classifier', 'Baseline'],
                y=[attacks.get("threshold_auc", 0.5), 
                   attacks.get("classifier_auc", 0.5), 0.5],
                name='AUC'
            ),
            row=1, col=2
        )
    
    # Plot 3: Trade-off curve
    if "tradeoff" in metrics_data:
        tradeoff = metrics_data["tradeoff"]
        fig.add_trace(
            go.Scatter(
                x=tradeoff.get("epsilons", []),
                y=tradeoff.get("accuracies", []),
                mode='lines+markers',
                name='Trade-off'
            ),
            row=2, col=1
        )
    
    # Plot 4: Radar comparison
    if "comparison" in metrics_data:
        comparison = metrics_data["comparison"]
        for method, values in comparison.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=list(values.values()),
                    theta=list(values.keys()),
                    fill='toself',
                    name=method
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title_text="Privacy-Preserving Federated Learning Dashboard",
        showlegend=True,
        height=800
    )
    
    # Save to HTML
    fig.write_html(output_path)
    logger.info(f"Created interactive dashboard: {output_path}")
    
    return output_path


def _generate_simple_html_report(metrics_data: Dict, output_path: str) -> str:
    """Generate simple HTML report without Plotly."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .critical {{ background: #ffcdd2; }}
            .warning {{ background: #fff9c4; }}
            .good {{ background: #c8e6c9; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Privacy-Preserving Federated Learning Report</h1>
        
        <h2>Privacy Budget</h2>
        <div class="metric">
            <pre>{json.dumps(metrics_data.get('budget', {}), indent=2)}</pre>
        </div>
        
        <h2>Attack Results</h2>
        <div class="metric">
            <pre>{json.dumps(metrics_data.get('attack_results', {}), indent=2)}</pre>
        </div>
        
        <h2>Full Metrics</h2>
        <div class="metric">
            <pre>{json.dumps(metrics_data, indent=2, default=str)}</pre>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def plot_privacy_utility_tradeoff(
    results: List[Dict[str, float]],
    output_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Convenience function to plot privacy-utility trade-off.
    
    Args:
        results: List of dicts with 'epsilon' and 'accuracy' keys.
        output_path: Optional output path.
    
    Returns:
        Figure object.
    """
    visualizer = PrivacyVisualizer()
    epsilons = [r['epsilon'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    return visualizer.plot_privacy_utility_tradeoff(epsilons, accuracies, save_path=output_path)


def plot_epsilon_evolution(
    budget_tracker,
    output_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Convenience function to plot epsilon evolution.
    
    Args:
        budget_tracker: PrivacyBudgetTracker instance.
        output_path: Optional output path.
    
    Returns:
        Figure object.
    """
    visualizer = PrivacyVisualizer()
    return visualizer.plot_epsilon_evolution(
        budget_tracker.round_history, 
        save_path=output_path
    )
