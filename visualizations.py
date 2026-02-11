"""Visualization module for stress test metrics."""

import os
from typing import List, Dict
from datetime import datetime
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config
from stress_test import StressTestResult


class MetricsVisualizer:
    """Generate visualizations for stress test results."""

    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_latency_comparison(
        self,
        results: List[StressTestResult],
        filename: str = "latency_comparison.png"
    ) -> str:
        """Create latency comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        test_names = [r.test_name for r in results]
        avg_latencies = [r.avg_latency * 1000 for r in results]
        p95_latencies = [r.p95_latency * 1000 for r in results]
        p99_latencies = [r.p99_latency * 1000 for r in results]

        x = np.arange(len(test_names))
        width = 0.25

        bars1 = ax.bar(x - width, avg_latencies, width, label='Avg Latency', color='#3498db')
        bars2 = ax.bar(x, p95_latencies, width, label='P95 Latency', color='#f39c12')
        bars3 = ax.bar(x + width, p99_latencies, width, label='P99 Latency', color='#e74c3c')

        ax.set_xlabel('Test Name')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency Comparison Across Tests')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def plot_throughput_chart(
        self,
        results: List[StressTestResult],
        filename: str = "throughput_chart.png"
    ) -> str:
        """Create throughput comparison chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        test_names = [r.test_name for r in results]
        throughputs = [r.throughput for r in results]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(test_names)))

        bars = ax.barh(test_names, throughputs, color=colors)

        ax.set_xlabel('Throughput (Queries/Second)')
        ax.set_title('Query Throughput by Test')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, throughputs):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f} QPS', va='center', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def plot_latency_distribution(
        self,
        results: List[StressTestResult],
        filename: str = "latency_distribution.png"
    ) -> str:
        """Create latency distribution histogram."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot up to 4 results
        for i, result in enumerate(results[:4]):
            ax = axes[i]
            latencies_ms = [l * 1000 for l in result.latencies]

            if latencies_ms:
                ax.hist(latencies_ms, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
                ax.axvline(result.avg_latency * 1000, color='red', linestyle='--',
                          label=f'Avg: {result.avg_latency * 1000:.2f}ms')
                ax.axvline(result.p95_latency * 1000, color='orange', linestyle='--',
                          label=f'P95: {result.p95_latency * 1000:.2f}ms')

            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{result.test_name}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Hide unused axes
        for i in range(len(results), 4):
            axes[i].set_visible(False)

        plt.suptitle('Latency Distribution by Test', fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def plot_concurrent_scaling(
        self,
        results: List[StressTestResult],
        filename: str = "concurrent_scaling.png"
    ) -> str:
        """Plot how latency scales with concurrency."""
        # Filter for concurrent query tests
        concurrent_results = [r for r in results if r.test_name.startswith("concurrent")]

        if not concurrent_results:
            print("No concurrent query results found.")
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Extract concurrency levels from test names
        concurrencies = []
        avg_latencies = []
        throughputs = []

        for r in concurrent_results:
            try:
                conc = int(r.test_name.split("_")[-1])
                concurrencies.append(conc)
                avg_latencies.append(r.avg_latency * 1000)
                throughputs.append(r.throughput)
            except:
                pass

        # Sort by concurrency
        sorted_data = sorted(zip(concurrencies, avg_latencies, throughputs))
        concurrencies, avg_latencies, throughputs = zip(*sorted_data) if sorted_data else ([], [], [])

        # Latency vs Concurrency
        ax1.plot(concurrencies, avg_latencies, 'o-', color='#3498db', linewidth=2, markersize=10)
        ax1.set_xlabel('Concurrent Queries')
        ax1.set_ylabel('Average Latency (ms)')
        ax1.set_title('Latency vs Concurrency')
        ax1.grid(alpha=0.3)

        # Throughput vs Concurrency
        ax2.plot(concurrencies, throughputs, 's-', color='#2ecc71', linewidth=2, markersize=10)
        ax2.set_xlabel('Concurrent Queries')
        ax2.set_ylabel('Throughput (QPS)')
        ax2.set_title('Throughput vs Concurrency')
        ax2.grid(alpha=0.3)

        plt.suptitle('Concurrent Query Scaling Analysis', fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def plot_k_value_comparison(
        self,
        results: List[StressTestResult],
        filename: str = "k_value_comparison.png"
    ) -> str:
        """Plot latency comparison for different k values."""
        # Filter for top_k tests
        k_results = [r for r in results if r.test_name.startswith("top_k")]

        if not k_results:
            print("No k-value test results found.")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        k_values = []
        avg_latencies = []
        p95_latencies = []

        for r in k_results:
            try:
                k = int(r.test_name.split("_")[-1])
                k_values.append(k)
                avg_latencies.append(r.avg_latency * 1000)
                p95_latencies.append(r.p95_latency * 1000)
            except:
                pass

        # Sort by k value
        sorted_data = sorted(zip(k_values, avg_latencies, p95_latencies))
        k_values, avg_latencies, p95_latencies = zip(*sorted_data) if sorted_data else ([], [], [])

        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width/2, avg_latencies, width, label='Avg Latency', color='#3498db')
        ax.bar(x + width/2, p95_latencies, width, label='P95 Latency', color='#f39c12')

        ax.set_xlabel('Top-K Value')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency by Top-K Value')
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filepath}")
        return filepath

    def create_interactive_dashboard(
        self,
        results: List[StressTestResult],
        filename: str = "dashboard.html"
    ) -> str:
        """Create an interactive Plotly dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latency Comparison',
                'Throughput by Test',
                'Error Rates',
                'Request Summary'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )

        test_names = [r.test_name for r in results]
        avg_latencies = [r.avg_latency * 1000 for r in results]
        p95_latencies = [r.p95_latency * 1000 for r in results]
        throughputs = [r.throughput for r in results]
        error_rates = [r.error_rate * 100 for r in results]
        total_success = sum(r.successful_requests for r in results)
        total_failed = sum(r.failed_requests for r in results)

        # Latency comparison
        fig.add_trace(
            go.Bar(name='Avg Latency', x=test_names, y=avg_latencies, marker_color='#3498db'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='P95 Latency', x=test_names, y=p95_latencies, marker_color='#f39c12'),
            row=1, col=1
        )

        # Throughput
        fig.add_trace(
            go.Bar(name='QPS', x=test_names, y=throughputs, marker_color='#2ecc71'),
            row=1, col=2
        )

        # Error rates
        fig.add_trace(
            go.Bar(name='Error Rate %', x=test_names, y=error_rates, marker_color='#e74c3c'),
            row=2, col=1
        )

        # Request summary pie
        fig.add_trace(
            go.Pie(
                labels=['Successful', 'Failed'],
                values=[total_success, total_failed],
                marker_colors=['#2ecc71', '#e74c3c']
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text='RAG Vector Benchmark Dashboard',
            height=800,
            showlegend=True,
            barmode='group'
        )

        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        print(f"Saved: {filepath}")
        return filepath

    def generate_summary_report(
        self,
        results: List[StressTestResult],
        filename: str = "summary_report.txt"
    ) -> str:
        """Generate a text summary report."""
        lines = [
            "=" * 60,
            "RAG VECTOR BENCHMARK - SUMMARY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "TEST RESULTS OVERVIEW",
            "-" * 40,
        ]

        for result in results:
            lines.extend([
                f"\n{result.test_name}:",
                f"  Total Requests: {result.total_requests}",
                f"  Success Rate: {(1 - result.error_rate) * 100:.2f}%",
                f"  Average Latency: {result.avg_latency * 1000:.2f}ms",
                f"  P50 Latency: {result.p50_latency * 1000:.2f}ms",
                f"  P95 Latency: {result.p95_latency * 1000:.2f}ms",
                f"  P99 Latency: {result.p99_latency * 1000:.2f}ms",
                f"  Throughput: {result.throughput:.2f} QPS",
            ])

        # Overall statistics
        all_latencies = []
        for r in results:
            all_latencies.extend([l * 1000 for l in r.latencies])

        if all_latencies:
            lines.extend([
                "",
                "OVERALL STATISTICS",
                "-" * 40,
                f"Total Tests Run: {len(results)}",
                f"Total Requests: {sum(r.total_requests for r in results)}",
                f"Overall Success Rate: {sum(r.successful_requests for r in results) / max(sum(r.total_requests for r in results), 1) * 100:.2f}%",
                f"Overall Average Latency: {np.mean(all_latencies):.2f}ms",
                f"Overall P95 Latency: {np.percentile(all_latencies, 95):.2f}ms",
            ])

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        report = "\n".join(lines)

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)

        print(f"Saved: {filepath}")
        print("\n" + report)
        return filepath

    def save_results_json(
        self,
        results: List[StressTestResult],
        filename: str = "results.json"
    ) -> str:
        """Save results as JSON for further analysis."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results]
        }

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved: {filepath}")
        return filepath

    def generate_all_visualizations(self, results: List[StressTestResult]) -> List[str]:
        """Generate all visualizations."""
        print("\n" + "=" * 50)
        print("Generating Visualizations")
        print("=" * 50)

        files = []

        files.append(self.plot_latency_comparison(results))
        files.append(self.plot_throughput_chart(results))
        files.append(self.plot_latency_distribution(results))
        files.append(self.plot_concurrent_scaling(results))
        files.append(self.plot_k_value_comparison(results))
        files.append(self.create_interactive_dashboard(results))
        files.append(self.generate_summary_report(results))
        files.append(self.save_results_json(results))

        print(f"\nGenerated {len([f for f in files if f])} visualization files in {self.output_dir}/")
        return files


if __name__ == "__main__":
    # Demo with mock results
    mock_results = [
        StressTestResult(
            test_name="concurrent_10",
            total_requests=50,
            total_time=5.0,
            successful_requests=50,
            failed_requests=0,
            latencies=[0.05 + np.random.random() * 0.02 for _ in range(50)]
        ),
        StressTestResult(
            test_name="concurrent_25",
            total_requests=125,
            total_time=10.0,
            successful_requests=123,
            failed_requests=2,
            latencies=[0.08 + np.random.random() * 0.03 for _ in range(123)]
        ),
        StressTestResult(
            test_name="top_k_5",
            total_requests=50,
            total_time=3.0,
            successful_requests=50,
            failed_requests=0,
            latencies=[0.04 + np.random.random() * 0.01 for _ in range(50)]
        ),
        StressTestResult(
            test_name="top_k_10",
            total_requests=50,
            total_time=3.5,
            successful_requests=50,
            failed_requests=0,
            latencies=[0.05 + np.random.random() * 0.015 for _ in range(50)]
        ),
    ]

    visualizer = MetricsVisualizer()
    visualizer.generate_all_visualizations(mock_results)
