#!/usr/bin/env python3
"""
Example script demonstrating probability visualization functionality.

This script shows how to use the LogprobVisualizer to plot both log probabilities
and probabilities (exponentiated logprobs) for comparing different LLMs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.logprob_visualizer import LogprobVisualizer
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main example function."""
    
    # Initialize the visualizer with the merged logprobs file
    parquet_path = 'output/merged/merged_results_logprobs.parquet'
    visualizer = LogprobVisualizer(parquet_path)
    
    # Get some sample indices
    available_indices = visualizer.get_available_indices()
    print(f"Total available texts: {len(available_indices)}")
    print(f"First 10 indices: {available_indices[:10]}")
    
    # Select a sample index for demonstration
    sample_index = available_indices[0]
    print(f"\nUsing sample index: {sample_index}")
    
    # Example 1: Plot log probabilities (default behavior)
    print("\n1. Plotting log probabilities...")
    fig1 = visualizer.plot_logprobs_for_text(
        original_index=sample_index,
        figsize=(12, 6),
        title=f"Log Probability Sequences (Index: {sample_index})"
    )

    fig1.savefig('output/plots/logprob_visualization_08_25.png', dpi=300, bbox_inches='tight')

    # # Example 2: Plot probabilities (exponentiated logprobs)
    # print("2. Plotting probabilities...")
    # fig2 = visualizer.plot_probabilities_for_text(
    #     original_index=sample_index,
    #     models=['Llama-3.1-8B-Instruct', 'Phi-4', 'Qwen3-0.6B'],  # Same models for comparison
    #     figsize=(12, 6),
    #     title=f"Probability Sequences (Index: {sample_index})"
    # )
    
    # # Example 3: Using the general method with use_probabilities=True
    # print("3. Using general method with use_probabilities=True...")
    # fig3 = visualizer.plot_logprobs_for_text(
    #     original_index=sample_index,
    #     models=['DeepSeek-7B-Chat', 'Yi-1.5-6B-Chat'],
    #     figsize=(12, 6),
    #     use_probabilities=True,  # This is the key parameter
    #     title=f"Probability Sequences - Alternative Models (Index: {sample_index})"
    # )
    
    # # Example 4: Multiple texts comparison with probabilities
    # if len(available_indices) >= 3:
    #     print("4. Plotting multiple texts with probabilities...")
    #     sample_indices = available_indices[:3]  # Take first 3 texts
    #     fig4 = visualizer.plot_probabilities_multiple_texts(
    #         original_indices=sample_indices,
    #         models=['Llama-3.1-8B-Instruct', 'Phi-4'],  # Limit to 2 models for clarity
    #         figsize=(15, 8)
    #     )
    
    # # Show statistics for the sample text
    # print(f"\nStatistics for text {sample_index}:")
    # stats_df = visualizer.compare_models_summary(sample_index)
    # print(stats_df[['Model', 'Num Tokens', 'Mean LogProb']].round(4))
    
    # # Save each figure to a file
    # fig1.savefig('output/plots/logprob_visualization.png', dpi=300, bbox_inches='tight')
    # fig2.savefig('output/plots/probability_visualization.png', dpi=300, bbox_inches='tight')
    # fig3.savefig('output/plots/probability_visualization_alternative.png', dpi=300, bbox_inches='tight')
    # fig4.savefig('output/plots/probability_visualization_multiple.png', dpi=300, bbox_inches='tight')
    
    # print("\nExample completed! You now have access to both log probability and probability visualizations.")


if __name__ == "__main__":
    main()
