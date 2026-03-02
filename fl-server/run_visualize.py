"""Quick visualization runner"""
import traceback
import matplotlib
matplotlib.use('Agg')

from privacy.privacy_analyzer import PrivacyAnalyzer

analyzer = PrivacyAnalyzer('./privacy_logs')
num = analyzer.load_metrics()
print(f"Loaded {num} rounds")

if num > 0:
    print("\nGenerating visualizations...")
    try:
        print("1. Privacy budget...")
        analyzer.plot_privacy_budget(save=True, show=False)
        print("   Done!")
    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
    
    try:
        print("2. Leakage risk...")
        analyzer.plot_leakage_risk(save=True, show=False)
        print("   Done!")
    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
    
    try:
        print("3. Federation health...")
        analyzer.plot_federation_health(save=True, show=False)
        print("   Done!")
    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
    
    try:
        print("4. Agent privacy heatmap...")
        analyzer.plot_agent_privacy_heatmap(save=True, show=False)
        print("   Done!")
    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
    
    print(f"\nVisualization files saved to: {analyzer.output_path}")
    
    print("\n" + "="*60)
    print(analyzer.generate_report())
else:
    print("No metrics found. Run demo_privacy_metrics.py first.")
