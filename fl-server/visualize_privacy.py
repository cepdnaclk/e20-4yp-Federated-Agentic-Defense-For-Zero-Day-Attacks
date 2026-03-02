"""
Privacy Metrics Visualization Script

Run this script to visualize privacy metrics from federation rounds.

Usage:
    python visualize_privacy.py [--log-path ./privacy_logs] [--no-show]

Examples:
    # Visualize from default logs directory
    python visualize_privacy.py

    # Visualize from custom directory
    python visualize_privacy.py --log-path ./custom_logs

    # Save only (no display)
    python visualize_privacy.py --no-show
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.privacy_analyzer import PrivacyAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize privacy metrics from federated learning"
    )
    parser.add_argument(
        "--log-path", "-l",
        default="./privacy_logs",
        help="Path to privacy logs directory (default: ./privacy_logs)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save visualizations without displaying them"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate text report"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export metrics to CSV"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("FEDERATED LEARNING PRIVACY METRICS VISUALIZER")
    print(f"{'='*60}")
    print(f"Log path: {args.log_path}")
    print()

    # Initialize analyzer
    analyzer = PrivacyAnalyzer(log_path=args.log_path)
    
    # Load metrics
    num_rounds = analyzer.load_metrics()
    
    if num_rounds == 0:
        print("No privacy metrics found!")
        print("\nTo generate sample data, run:")
        print("  python demo_privacy_metrics.py")
        print("\nOr start the FL server and run some federation rounds.")
        return 1
    
    print(f"Loaded {num_rounds} federation rounds\n")

    # Generate visualizations
    analyzer.plot_privacy_dashboard(save=True, show=not args.no_show)

    # Generate report if requested
    if args.report:
        report = analyzer.generate_report(
            output_file=os.path.join(args.log_path, "visualizations", "privacy_report.txt")
        )
        print("\n" + report)

    # Export to CSV if requested
    if args.csv:
        analyzer.export_to_csv()

    print(f"\nVisualization output: {analyzer.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
