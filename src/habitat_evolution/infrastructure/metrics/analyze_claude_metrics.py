"""
Analyze Claude API metrics.

This script analyzes the metrics collected from Claude API usage,
providing insights into API performance, token consumption, and costs.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[4]))

from src.habitat_evolution.infrastructure.metrics.claude_api_metrics import ClaudeAPIMetrics


def load_metrics(metrics_file: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from the metrics file.
    
    Args:
        metrics_file: Path to the metrics file
        
    Returns:
        List of metrics entries
    """
    metrics = []
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return metrics
    
    with open(metrics_file, "r") as f:
        for line in f:
            try:
                metrics.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return metrics


def analyze_query_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze query metrics.
    
    Args:
        metrics: List of metrics entries
        
    Returns:
        Dictionary containing analysis results
    """
    query_metrics = [m for m in metrics if m.get("type") == "query"]
    
    if not query_metrics:
        return {"status": "No query metrics found"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(query_metrics)
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate statistics
    total_queries = len(df)
    total_tokens = df["tokens_used"].sum()
    avg_tokens_per_query = df["tokens_used"].mean()
    avg_response_time = df["response_time_ms"].mean()
    
    # Calculate token rate over time
    df = df.sort_values("timestamp")
    df["cumulative_tokens"] = df["tokens_used"].cumsum()
    
    # Calculate cost (assuming $15 per million tokens for Claude 3 Opus)
    cost_per_million = 15.0
    total_cost = (total_tokens / 1_000_000) * cost_per_million
    
    return {
        "total_queries": total_queries,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": avg_tokens_per_query,
        "avg_response_time_ms": avg_response_time,
        "total_cost_usd": total_cost,
        "dataframe": df
    }


def analyze_document_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze document processing metrics.
    
    Args:
        metrics: List of metrics entries
        
    Returns:
        Dictionary containing analysis results
    """
    document_metrics = [m for m in metrics if m.get("type") == "document"]
    
    if not document_metrics:
        return {"status": "No document metrics found"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(document_metrics)
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate statistics
    total_documents = len(df)
    total_tokens = df["tokens_used"].sum()
    avg_tokens_per_document = df["tokens_used"].mean()
    avg_response_time = df["response_time_ms"].mean()
    avg_patterns_per_document = df["pattern_count"].mean()
    
    # Calculate token rate over time
    df = df.sort_values("timestamp")
    df["cumulative_tokens"] = df["tokens_used"].cumsum()
    
    # Calculate cost (assuming $15 per million tokens for Claude 3 Opus)
    cost_per_million = 15.0
    total_cost = (total_tokens / 1_000_000) * cost_per_million
    
    return {
        "total_documents": total_documents,
        "total_tokens": total_tokens,
        "avg_tokens_per_document": avg_tokens_per_document,
        "avg_response_time_ms": avg_response_time,
        "avg_patterns_per_document": avg_patterns_per_document,
        "total_cost_usd": total_cost,
        "dataframe": df
    }


def analyze_error_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze error metrics.
    
    Args:
        metrics: List of metrics entries
        
    Returns:
        Dictionary containing analysis results
    """
    error_metrics = [m for m in metrics if m.get("type") == "error"]
    
    if not error_metrics:
        return {"status": "No error metrics found"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(error_metrics)
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate statistics
    total_errors = len(df)
    errors_by_operation = df["operation_type"].value_counts().to_dict()
    
    # Get most common error messages
    error_counts = df["error_message"].value_counts().head(5).to_dict()
    
    return {
        "total_errors": total_errors,
        "errors_by_operation": errors_by_operation,
        "common_errors": error_counts,
        "dataframe": df
    }


def plot_token_usage_over_time(query_df: pd.DataFrame, document_df: pd.DataFrame, output_dir: Path):
    """
    Plot token usage over time.
    
    Args:
        query_df: DataFrame containing query metrics
        document_df: DataFrame containing document metrics
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot query token usage
    if not query_df.empty:
        plt.plot(query_df["timestamp"], query_df["cumulative_tokens"], label="Queries")
    
    # Plot document token usage
    if not document_df.empty:
        plt.plot(document_df["timestamp"], document_df["cumulative_tokens"], label="Documents")
    
    plt.title("Cumulative Token Usage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Tokens")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = output_dir / "token_usage_over_time.png"
    plt.savefig(output_file)
    plt.close()
    
    print(f"Token usage plot saved to: {output_file}")


def plot_response_time_distribution(query_df: pd.DataFrame, document_df: pd.DataFrame, output_dir: Path):
    """
    Plot response time distribution.
    
    Args:
        query_df: DataFrame containing query metrics
        document_df: DataFrame containing document metrics
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot query response time distribution
    if not query_df.empty:
        plt.hist(query_df["response_time_ms"], bins=20, alpha=0.5, label="Queries")
    
    # Plot document response time distribution
    if not document_df.empty:
        plt.hist(document_df["response_time_ms"], bins=20, alpha=0.5, label="Documents")
    
    plt.title("Response Time Distribution")
    plt.xlabel("Response Time (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = output_dir / "response_time_distribution.png"
    plt.savefig(output_file)
    plt.close()
    
    print(f"Response time plot saved to: {output_file}")


def plot_pattern_count_distribution(document_df: pd.DataFrame, output_dir: Path):
    """
    Plot pattern count distribution.
    
    Args:
        document_df: DataFrame containing document metrics
        output_dir: Directory to save the plot
    """
    if document_df.empty:
        return
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(document_df["pattern_count"], bins=10, alpha=0.7)
    
    plt.title("Pattern Count Distribution")
    plt.xlabel("Number of Patterns")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Save the plot
    output_file = output_dir / "pattern_count_distribution.png"
    plt.savefig(output_file)
    plt.close()
    
    print(f"Pattern count plot saved to: {output_file}")


def generate_report(query_analysis: Dict[str, Any], document_analysis: Dict[str, Any], 
                   error_analysis: Dict[str, Any], output_dir: Path):
    """
    Generate a report of the metrics analysis.
    
    Args:
        query_analysis: Results of query metrics analysis
        document_analysis: Results of document metrics analysis
        error_analysis: Results of error metrics analysis
        output_dir: Directory to save the report
    """
    report = []
    
    # Add report header
    report.append("# Claude API Metrics Analysis")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Add query metrics
    report.append("## Query Metrics")
    if query_analysis.get("status"):
        report.append(query_analysis["status"])
    else:
        report.append(f"Total Queries: {query_analysis['total_queries']}")
        report.append(f"Total Tokens: {query_analysis['total_tokens']:,}")
        report.append(f"Average Tokens per Query: {query_analysis['avg_tokens_per_query']:.2f}")
        report.append(f"Average Response Time: {query_analysis['avg_response_time_ms']:.2f} ms")
        report.append(f"Estimated Cost: ${query_analysis['total_cost_usd']:.4f}")
    report.append("")
    
    # Add document metrics
    report.append("## Document Metrics")
    if document_analysis.get("status"):
        report.append(document_analysis["status"])
    else:
        report.append(f"Total Documents: {document_analysis['total_documents']}")
        report.append(f"Total Tokens: {document_analysis['total_tokens']:,}")
        report.append(f"Average Tokens per Document: {document_analysis['avg_tokens_per_document']:.2f}")
        report.append(f"Average Response Time: {document_analysis['avg_response_time_ms']:.2f} ms")
        report.append(f"Average Patterns per Document: {document_analysis['avg_patterns_per_document']:.2f}")
        report.append(f"Estimated Cost: ${document_analysis['total_cost_usd']:.4f}")
    report.append("")
    
    # Add error metrics
    report.append("## Error Metrics")
    if error_analysis.get("status"):
        report.append(error_analysis["status"])
    else:
        report.append(f"Total Errors: {error_analysis['total_errors']}")
        report.append("")
        report.append("### Errors by Operation Type")
        for op_type, count in error_analysis["errors_by_operation"].items():
            report.append(f"- {op_type}: {count}")
        report.append("")
        report.append("### Common Error Messages")
        for error_msg, count in error_analysis["common_errors"].items():
            report.append(f"- ({count}) {error_msg}")
    report.append("")
    
    # Add total cost
    total_cost = 0
    if not query_analysis.get("status"):
        total_cost += query_analysis["total_cost_usd"]
    if not document_analysis.get("status"):
        total_cost += document_analysis["total_cost_usd"]
    
    report.append("## Total Cost")
    report.append(f"Estimated Total Cost: ${total_cost:.4f}")
    report.append("")
    
    # Add plots section
    report.append("## Visualizations")
    report.append("The following visualizations have been generated:")
    report.append("")
    report.append("1. Token Usage Over Time: `token_usage_over_time.png`")
    report.append("2. Response Time Distribution: `response_time_distribution.png`")
    report.append("3. Pattern Count Distribution: `pattern_count_distribution.png`")
    report.append("")
    
    # Write the report
    report_file = output_dir / "claude_metrics_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    
    print(f"Report saved to: {report_file}")


def main():
    """Main function."""
    # Get the metrics file
    metrics_dir = Path(__file__).parents[3] / "data" / "metrics"
    metrics_file = metrics_dir / "claude_api_metrics.jsonl"
    
    # Create output directory for visualizations
    output_dir = metrics_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(metrics_file)
    
    if not metrics:
        print("No metrics found. Run some queries or document processing first.")
        return
    
    print(f"Loaded {len(metrics)} metrics entries")
    
    # Analyze metrics
    query_analysis = analyze_query_metrics(metrics)
    document_analysis = analyze_document_metrics(metrics)
    error_analysis = analyze_error_metrics(metrics)
    
    # Generate visualizations
    query_df = query_analysis.get("dataframe", pd.DataFrame())
    document_df = document_analysis.get("dataframe", pd.DataFrame())
    
    plot_token_usage_over_time(query_df, document_df, output_dir)
    plot_response_time_distribution(query_df, document_df, output_dir)
    plot_pattern_count_distribution(document_df, output_dir)
    
    # Generate report
    generate_report(query_analysis, document_analysis, error_analysis, output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
