"""
Logprob Sequence Visualizer

This module provides functionality to visualize log probability sequences
from different LLMs for the same text, with normalized x-axis (0-1) to
account for different tokenization across models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import warnings

# Set up matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")

class LogprobVisualizer:
    """
    A class to visualize log probability sequences from different LLMs.
    """
    
    def __init__(self, parquet_file: str):
        """
        Initialize the visualizer with a parquet file containing logprob data.
        
        Args:
            parquet_file (str): Path to the parquet file with merged logprob data
        """
        self.parquet_file = parquet_file
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load the parquet data."""
        try:
            self.df = pd.read_parquet(self.parquet_file)
            print(f"Loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Get model names (all columns except 'original_index')
            self.model_names = [col for col in self.df.columns if col != 'original_index']
            print(f"Available models: {self.model_names}")
            
        except Exception as e:
            raise ValueError(f"Error loading parquet file: {e}")
    
    def get_available_indices(self) -> List[int]:
        """
        Get list of available original_index values.
        
        Returns:
            List[int]: List of available original indices
        """
        return sorted(self.df['original_index'].tolist())
    
    def plot_logprobs_for_text(self, 
                              original_index: int, 
                              models: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (12, 8),
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              show_legend: bool = True,
                              alpha: float = 0.7,
                              linewidth: float = 1.5,
                              use_probabilities: bool = False) -> plt.Figure:
        """
        Plot logprob sequences for a specific text across different models.
        
        Args:
            original_index (int): The original index of the text to visualize
            models (Optional[List[str]]): List of model names to include. If None, includes all models.
            figsize (Tuple[int, int]): Figure size (width, height)
            title (Optional[str]): Custom title for the plot
            save_path (Optional[str]): Path to save the figure
            show_legend (bool): Whether to show the legend
            alpha (float): Transparency of the lines
            linewidth (float): Width of the lines
            use_probabilities (bool): If True, plot probabilities (exp(logprobs)) instead of raw logprobs
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        
        # Check if the index exists
        if original_index not in self.df['original_index'].values:
            raise ValueError(f"Original index {original_index} not found in data")
        
        # Get the row for this text
        row = self.df[self.df['original_index'] == original_index].iloc[0]
        
        # Use all models if none specified
        if models is None:
            models = self.model_names
        else:
            # Validate model names
            invalid_models = [m for m in models if m not in self.model_names]
            if invalid_models:
                raise ValueError(f"Invalid model names: {invalid_models}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each model's logprob sequence
        for i, model in enumerate(models):
            logprobs = row[model]
            
            # Skip if no data for this model
            if logprobs is None or len(logprobs) == 0:
                print(f"Warning: No logprob data for model {model}")
                continue
            
            # Create normalized x-axis (0 to 1)
            x_normalized = np.linspace(0, 1, len(logprobs))
            
            # Convert to probabilities if requested
            if use_probabilities:
                y_values = np.exp(logprobs)
                y_label = "Probability"
                plot_title_prefix = "Probability"
            else:
                y_values = logprobs
                y_label = "Log Probability"
                plot_title_prefix = "Log Probability"
            
            # Plot the sequence
            ax.plot(x_normalized, y_values, 
                   label=f"{model} ({len(logprobs)} tokens)",
                   alpha=alpha, 
                   linewidth=linewidth)
        
        # Customize the plot
        ax.set_xlabel('Normalized Token Position (0-1)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        if title is None:
            title = f'{plot_title_prefix} Sequences for Text (Original Index: {original_index})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Tight layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_multiple_texts(self, 
                           original_indices: List[int],
                           models: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None,
                           use_probabilities: bool = False) -> plt.Figure:
        """
        Plot logprob sequences for multiple texts in subplots.
        
        Args:
            original_indices (List[int]): List of original indices to plot
            models (Optional[List[str]]): List of model names to include
            figsize (Tuple[int, int]): Figure size (width, height)
            save_path (Optional[str]): Path to save the figure
            use_probabilities (bool): If True, plot probabilities (exp(logprobs)) instead of raw logprobs
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        
        n_texts = len(original_indices)
        n_cols = min(2, n_texts)
        n_rows = (n_texts + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_texts == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, original_index in enumerate(original_indices):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            
            # Get the data for this text
            if original_index not in self.df['original_index'].values:
                ax.text(0.5, 0.5, f'Index {original_index}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Original Index: {original_index}')
                continue
            
            row_data = self.df[self.df['original_index'] == original_index].iloc[0]
            
            # Use all models if none specified
            plot_models = models if models is not None else self.model_names
            
            # Determine y-axis label and values
            y_label = "Probability" if use_probabilities else "Log Probability"
            
            # Plot each model's sequence
            for model in plot_models:
                logprobs = row_data[model]
                if logprobs is not None and len(logprobs) > 0:
                    x_normalized = np.linspace(0, 1, len(logprobs))
                    
                    # Convert to probabilities if requested
                    y_values = np.exp(logprobs) if use_probabilities else logprobs
                    
                    ax.plot(x_normalized, y_values, 
                           label=f"{model} ({len(logprobs)})",
                           alpha=0.7, linewidth=1.2)
            
            ax.set_xlabel('Normalized Position')
            ax.set_ylabel(y_label)
            ax.set_title(f'Index: {original_index}')
            ax.grid(True, alpha=0.3)
            
            # Add legend for first subplot only
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Hide empty subplots
        for i in range(n_texts, n_rows * n_cols):
            if n_rows > 1:
                row_idx = i // n_cols
                col_idx = i % n_cols
                axes[row_idx, col_idx].set_visible(False)
            elif n_cols > 1 and i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def get_text_stats(self, original_index: int) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about logprob sequences for a specific text.
        
        Args:
            original_index (int): The original index of the text
            
        Returns:
            Dict[str, Dict[str, float]]: Statistics for each model
        """
        if original_index not in self.df['original_index'].values:
            raise ValueError(f"Original index {original_index} not found in data")
        
        row = self.df[self.df['original_index'] == original_index].iloc[0]
        stats = {}
        
        for model in self.model_names:
            logprobs = row[model]
            if logprobs is not None and len(logprobs) > 0:
                stats[model] = {
                    'num_tokens': len(logprobs),
                    'mean_logprob': np.mean(logprobs),
                    'std_logprob': np.std(logprobs),
                    'min_logprob': np.min(logprobs),
                    'max_logprob': np.max(logprobs),
                    'median_logprob': np.median(logprobs)
                }
            else:
                stats[model] = None
        
        return stats
    
    def compare_models_summary(self, original_index: int) -> pd.DataFrame:
        """
        Create a summary comparison of models for a specific text.
        
        Args:
            original_index (int): The original index of the text
            
        Returns:
            pd.DataFrame: Summary statistics for all models
        """
        stats = self.get_text_stats(original_index)
        
        summary_data = []
        for model, model_stats in stats.items():
            if model_stats is not None:
                summary_data.append({
                    'Model': model,
                    'Num Tokens': model_stats['num_tokens'],
                    'Mean LogProb': model_stats['mean_logprob'],
                    'Std LogProb': model_stats['std_logprob'],
                    'Min LogProb': model_stats['min_logprob'],
                    'Max LogProb': model_stats['max_logprob'],
                    'Median LogProb': model_stats['median_logprob']
                })
        
        return pd.DataFrame(summary_data)
    
    def plot_probabilities_for_text(self, 
                                   original_index: int, 
                                   models: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (12, 8),
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None,
                                   show_legend: bool = True,
                                   alpha: float = 0.7,
                                   linewidth: float = 1.5) -> plt.Figure:
        """
        Convenience method to plot probabilities (exponentiated logprobs) for a specific text.
        
        Args:
            original_index (int): The original index of the text to visualize
            models (Optional[List[str]]): List of model names to include. If None, includes all models.
            figsize (Tuple[int, int]): Figure size (width, height)
            title (Optional[str]): Custom title for the plot
            save_path (Optional[str]): Path to save the figure
            show_legend (bool): Whether to show the legend
            alpha (float): Transparency of the lines
            linewidth (float): Width of the lines
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        return self.plot_logprobs_for_text(
            original_index=original_index,
            models=models,
            figsize=figsize,
            title=title,
            save_path=save_path,
            show_legend=show_legend,
            alpha=alpha,
            linewidth=linewidth,
            use_probabilities=True
        )
    
    def plot_probabilities_multiple_texts(self, 
                                         original_indices: List[int],
                                         models: Optional[List[str]] = None,
                                         figsize: Tuple[int, int] = (15, 10),
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Convenience method to plot probabilities for multiple texts in subplots.
        
        Args:
            original_indices (List[int]): List of original indices to plot
            models (Optional[List[str]]): List of model names to include
            figsize (Tuple[int, int]): Figure size (width, height)
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        return self.plot_multiple_texts(
            original_indices=original_indices,
            models=models,
            figsize=figsize,
            save_path=save_path,
            use_probabilities=True
        )


def demo_usage():
    """Demonstrate usage of the LogprobVisualizer."""
    
    # Initialize the visualizer
    visualizer = LogprobVisualizer('output/merged/merged_results_logprobs.parquet')
    
    # Get available indices
    available_indices = visualizer.get_available_indices()
    print(f"Available indices: {available_indices[:10]}... (showing first 10)")
    
    # Plot logprobs for the first available text
    first_index = available_indices[0]
    print(f"\nPlotting logprobs for text with original_index={first_index}")
    
    # Create the logprob plot
    fig1 = visualizer.plot_logprobs_for_text(
        original_index=first_index,
        models=None,  # Use all available models
        figsize=(14, 8),
        save_path=f'./logprob_visualization_{first_index}.png'
    )
    
    # Create the probability plot
    print(f"\nPlotting probabilities for text with original_index={first_index}")
    fig2 = visualizer.plot_probabilities_for_text(
        original_index=first_index,
        models=None,  # Use all available models
        figsize=(14, 8),
        save_path=f'./probability_visualization_{first_index}.png'
    )
    
    # Show the plots
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary statistics for text {first_index}:")
    summary_df = visualizer.compare_models_summary(first_index)
    print(summary_df.round(3))


if __name__ == "__main__":
    demo_usage()
