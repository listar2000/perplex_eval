from regression import *
from loader import load_merged_data, load_llm_responses, load_simple_original_text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_correlation_distribution(correlations: pd.Series, title: str):
    """
    Plot the distribution of the correlations.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.savefig(f"{title}.png", dpi=300)


def plot_rank_versus_correlation(correlations: pd.Series, title: str):
    """
    Sort the correlations by rank and plot the correlation versus the rank.
    """
    plt.figure(figsize=(10, 6))
    sorted_correlations = correlations.sort_values()
    # Use range(len()) to get proper rank numbers from 1 to N
    ranks = range(1, len(sorted_correlations) + 1)
    plt.scatter(ranks, sorted_correlations.values)
    plt.title(title)
    plt.xlabel('Rank')
    plt.ylabel('Correlation')
    plt.savefig(f"{title}.png", dpi=300)


def test_full_pipeline():
    covariate_path = "data/merged_results_avg_logprob.parquet"
    response_path = "data/gsm8k.json"

    covariate_df = load_merged_data(covariate_path)
    response_dict = load_llm_responses(response_path)

    og_df = load_simple_original_text("data/redpajama-subset-50k.parquet")

    preselect_correlations = get_preselect_rank_correlations(
        covariate_df, response_dict)

    # plot_correlation_distribution(
    #     preselect_correlations, "preselect_gsm8k")

    # Check 1: see the documents that the lowest preselect correlation corresponds to:
    lowest_10_preselect_idx = preselect_correlations.sort_values().index[:10]
    for idx in lowest_10_preselect_idx:
        print(f"Document {idx}: domain = {og_df.loc[idx, 'domain']}")
        print(f"Preselect correlation: {preselect_correlations.loc[idx]}")
        print(f"Text: {og_df.loc[idx, 'text'][:100]}")
        print("-" * 100)
    print("*" * 100)

    # filter the dataframe by the thrush correlations
    filtered_df, filtered_preselect_correlations = filter_df_by_correlation(
        covariate_df, preselect_correlations, 0.1, absolute=True)

    # Run two times and check overlapping coefficient proportion
    selected_covariates = penalized_regression(
        filtered_df, response_dict, penalty="lasso", verbose=True, n_cv_folds=5, random_state=123)

    # selected_covariates = forward_selection(
    #     filtered_df, response_dict, stop_criterion="adj_r2", verbose=True)

    # Check 2: see the documents that selected covariates correspond to:
    for idx in selected_covariates.index:
        print(f"Document {idx}: domain = {og_df.loc[idx, 'domain']}")
        print(
            f"Preselect correlation: {preselect_correlations.loc[idx]}, Regression coefficient: {selected_covariates.loc[idx]}")
        print(f"Text: {og_df.loc[idx, 'text'][:100]}")
        print("-" * 100)


def test_dummy_data():
    # we create a [30000, 10] dataframe filled with Unif[0, 1] random numbers
    covariate_df = pd.DataFrame(np.random.rand(30000, 10))

    col_names = [f"model_{i}" for i in range(10)]
    covariate_df.columns = col_names

    response_dict = {k: np.random.rand() for k in col_names}

    thrush_correlations = get_thrush_rank_correlations(
        covariate_df, response_dict)
    preselect_correlations = get_preselect_rank_correlations(
        covariate_df, response_dict)

    plot_correlation_distribution(
        thrush_correlations, "dummy_thrush_correlation_distribution")
    plot_correlation_distribution(
        preselect_correlations, "dummy_preselect_correlation_distribution")


if __name__ == "__main__":
    test_full_pipeline()