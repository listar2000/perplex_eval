"""
All of the filtering/regression algorithms go here.

An overall pipeline looks like this:

1. Get the rank correlations (Thrush et al. 2024 / Shum et al. 2025)
2. Filter the dataframe by correlation (absolute or relative)
3. Forward selection
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

import logging
from tqdm import tqdm

# Configure logger to output to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if it doesn't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

"""
PIPELINE 1: Get the rank correlations
"""


def get_thrush_rank_correlations(covariate_df: pd.DataFrame, response_dict: dict, verbose: bool = False) -> pd.Series:
    """
    Calculates the [Thrush et al. 2024] rank correlation for each covariate (document).

    This method computes a U-statistic that measures the alignment between the ranking of
    models by performance and their ranking by perplexity on a given document.

    Arguments:
        covariate_df: pd.DataFrame
            The covariates dataframe where each row is a document/chunk (D), and each column
            is an LLM (N). Shape: (D, N).
        response_dict: dict
            A dictionary of <LLM_name, response_score> pairs. Higher scores are better.
        verbose: bool
            Whether to print verbose output.

    Returns:
        pd.Series
            A series containing the Thrush rank correlation score for each document,
            indexed by the document IDs from the input DataFrame.
    """
    # --- 1. Align Inputs ---
    # Ensure we only use LLMs present in both inputs
    # Sort to ensure deterministic ordering across runs
    common_llms = sorted(list(set(covariate_df.columns)
                         & set(response_dict.keys())))
    if len(common_llms) < 2:
        raise ValueError(
            "Need at least 2 common LLMs between the dataframe and the response dictionary.")

    if verbose:
        logger.info("LLMs (columns):", common_llms)
        logger.info("# of documents (rows):", len(covariate_df))

    # Filter and align the data. The order of llms will be consistent from now on.
    X = covariate_df[common_llms]
    y = pd.Series(response_dict)[common_llms]

    # --- 2. Pre-calculate Ranks and Signs ---
    # The paper's formula is easier to compute if models are rows. Transpose X.
    # Now X_T has shape (N, D)
    X_T = X.T

    # For each document (column), rank the models' perplexities.
    # Lower perplexity should get a lower rank.
    # Using pct=True gives percentile ranks, which is robust and normalized.
    # Use method='first' to ensure deterministic ranking when there are ties
    X_ranks = X_T.rank(axis=0, pct=True, method='first')

    # Pre-compute the sign of performance differences for all model pairs (k, l).
    # This creates an (N, N) matrix where sign_y[k, l] = sign(y_k - y_l).
    y_values = y.values
    sign_y = np.sign(np.subtract.outer(y_values, y_values))

    # --- 3. Vectorized Correlation Calculation ---
    # This function will be applied to each document's rank vector (each column of X_ranks).
    def calculate_gamma_j(rank_vec_j: pd.Series) -> float:
        # Create an (N, N) matrix of rank differences for the current document j.
        # rank_diffs[k, l] = rank_k - rank_l
        rank_diffs = np.subtract.outer(rank_vec_j.values, rank_vec_j.values)

        # Element-wise multiply the sign matrix and the rank difference matrix,
        # and sum all elements to get the final score for document j.
        gamma_j = np.sum(sign_y * rank_diffs)
        return gamma_j

    # Apply the calculation to every document (column) in the ranked dataframe.
    thrush_correlations = X_ranks.apply(calculate_gamma_j, axis=0)
    thrush_correlations.name = "thrush_correlation"

    return thrush_correlations


def get_preselect_rank_correlations(covariate_df: pd.DataFrame, response_dict: dict, verbose: bool = False) -> pd.Series:
    """
    Calculates the [Shum et al. 2025] PRESELECT predictive strength score for each covariate.

    This method counts, for a given document, the number of model pairs where the
    model with lower performance rank also has a higher perplexity.

    Arguments:
        covariate_df: pd.DataFrame
            The covariates dataframe where each row is a document/chunk (D), and each column
            is an LLM (N). Shape: (D, N).
        response_dict: dict
            A dictionary of <LLM_name, response_score> pairs. Higher scores are better.
        verbose: bool
            Whether to print verbose output.

    Returns:
        pd.Series
            A series containing the PRESELECT score for each document, indexed by the
            document IDs from the input DataFrame.
    """
    # --- 1. Align and Sort Inputs ---
    # Sort to ensure deterministic ordering across runs
    common_llms = sorted(list(set(covariate_df.columns)
                         & set(response_dict.keys())))
    if len(common_llms) < 2:
        raise ValueError(
            "Need at least 2 common LLMs between the dataframe and the response dictionary.")

    if verbose:
        logger.info("LLMs (columns):", common_llms)
        logger.info("# of documents (rows):", len(covariate_df))

    # Create a Series of responses and sort it to get the performance ranking of LLMs.
    y = pd.Series(response_dict)[common_llms]
    sorted_llms = y.sort_values().index.tolist()
    n_models = len(sorted_llms)

    # Reorder the covariate dataframe columns according to the performance ranking.
    # Now, column `i` corresponds to the model with the i-th best performance.
    X_sorted = covariate_df[sorted_llms]

    # --- 2. Vectorized Score Calculation ---
    # This function will be applied to each document (row of X_sorted).
    def calculate_s_d(perplexity_row: pd.Series) -> float:
        # `c_d` is the vector of perplexities for one document, sorted by model performance.
        c_d = perplexity_row.values

        # Create an (N, N) boolean matrix `G` where G[i, j] is True if c_d[i] > c_d[j].
        # This checks if a worse-performing model `i` has a higher perplexity than a
        # better-performing model `j`.
        G = np.greater.outer(c_d, c_d)

        # The formula sums over pairs where performance_rank(i) < performance_rank(j).
        # In our sorted data, this corresponds to i < j.
        # This is equivalent to summing the upper triangle of the matrix G.
        # np.triu(G, k=1) zeros out the lower triangle and diagonal.
        inversion_count = np.triu(G, k=1).sum()
        return inversion_count

    # Apply the calculation to every document (row)
    preselect_scores = X_sorted.apply(calculate_s_d, axis=1)

    # --- 3. Normalize the Score ---
    # The normalization factor Z is the total number of pairs, N * (N-1) / 2.
    num_pairs = n_models * (n_models - 1) / 2
    if num_pairs > 0:
        preselect_scores /= num_pairs

    preselect_scores.name = "preselect_score"
    return preselect_scores


"""
PIPELINE 2: Filter the dataframe by correlation
"""


def filter_df_by_correlation(covariate_df: pd.DataFrame, correlation_series: pd.Series, value: float, absolute: bool = True) -> (pd.DataFrame, pd.Series):
    """
    Filter the dataframe by the correlation series. This function supports either absolute filtering or relative 
    (so value need to be a percentage) filtering.

    Arguments:
        covariate_df: pd.DataFrame
            The covariates dataframe where each row is a document/chunk (D), and each column
            is an LLM (N). Shape: (D, N).
        correlation_series: pd.Series
            A series containing the correlation score for each covariate, indexed by the
            covariate IDs from the input DataFrame.
        value: float
            The value to filter the dataframe by. If absolute is True, this is the value of the correlation score.
            If absolute is False, this is the relative value of the correlation score.
        absolute: bool
            Whether to filter the dataframe by the absolute value of the correlation score.

    Returns:
        pd.DataFrame
            The filtered dataframe.
        pd.Series
            The filtered correlation series.
    """
    if absolute:
        if correlation_series.name == "thrush_correlation":
            return covariate_df[correlation_series > value], correlation_series[correlation_series > value]
        elif correlation_series.name == "preselect_score":
            return covariate_df[correlation_series < value], correlation_series[correlation_series < value]
        else:
            raise ValueError(
                "Correlation series must be either 'thrush_correlation' or 'preselect_score'.")
    else:
        assert value > 0 and value < 1, "Value must be between 0 and 1 for relative filtering."
        # value is now a percentage and we will obtain the value% max/min correlations depending on the correlation type
        if correlation_series.name == "thrush_correlation":
            quantile = correlation_series.quantile(1 - value)
            return covariate_df[correlation_series > quantile], correlation_series[correlation_series > quantile]
        elif correlation_series.name == "preselect_score":
            quantile = correlation_series.quantile(value)
            return covariate_df[correlation_series < quantile], correlation_series[correlation_series < quantile]
        else:
            raise ValueError(
                "Correlation series must be either 'thrush_correlation' or 'preselect_score'.")


"""
PIPELINE 3: Regression algorithms
"""


def forward_selection(covariate_df: pd.DataFrame, response_dict: dict, stop_criterion: str = "AIC", verbose: bool = False) -> pd.Series:
    """
    Performs forward selection to build a sparse linear model.

    Starts with a null model (intercept only) and iteratively adds the covariate that
    most improves the model according to the specified stopping criterion (AIC or BIC),
    until no further improvement is possible.

    Arguments:
        covariate_df: pd.DataFrame
            The covariates dataframe where each row is a document/chunk (D), and each column
            is an LLM (N). Shape: (D, N).
        response_dict: dict
            A dictionary of <LLM_name, response_score> pairs. Higher scores are better.
        stop_criterion: str
            The criterion to stop the forward selection ("AIC", "BIC", or "adj_r2"). Default is "AIC".
            BIC tends to result in smaller, more parsimonious models.
        verbose: bool
            Whether to print verbose output.

    Returns:
        pd.Series
            A series where the index contains the names of the selected covariates
            (document IDs) and the values are their corresponding regression coefficients.
    """
    # --- 1. Align and Prepare Data ---
    # Ensure we only use LLMs present in both inputs
    # Sort to ensure deterministic ordering across runs
    common_llms = sorted(list(set(covariate_df.columns)
                         & set(response_dict.keys())))
    if len(common_llms) < 2:
        raise ValueError(
            "Need at least 2 common LLMs between the dataframe and the response dictionary.")

    # The statsmodels OLS function expects predictors (X) and response (y).
    # In our case, LLMs are the observations (n), and documents are the potential predictors (p).
    # So we need to transpose the covariate_df.
    # X_T has shape (N, D) - N observations, D potential features.
    X_T = covariate_df[common_llms].T
    y = pd.Series(response_dict)[common_llms]

    # Add a constant for the intercept term in the regression
    X_T_with_const = sm.add_constant(X_T, has_constant='add')

    n_obs, n_features = X_T.shape
    initial_covariates = list(X_T.columns)
    selected_covariates = []

    # --- 2. Initialize the Model ---
    # Start with a null model (intercept only)
    # The intercept column is always the first one after sm.add_constant
    X_current_const = X_T_with_const.iloc[:, [0]]
    best_model = sm.OLS(y, X_current_const).fit()

    if stop_criterion.lower() == 'aic':
        best_criterion_score = best_model.aic
    elif stop_criterion.lower() == 'bic':
        best_criterion_score = best_model.bic
    elif stop_criterion.lower() == 'adj_r2':
        best_criterion_score = best_model.rsquared_adj
    else:
        raise ValueError("stop_criterion must be 'AIC', 'BIC', or 'adj_r2'")

    if verbose:
        logger.info(
            f"Initial model (intercept only): {stop_criterion} = {best_criterion_score:.4f}")

    # --- 3. Iterative Selection Loop ---
    while True:
        remaining_covariates = [
            c for c in initial_covariates if c not in selected_covariates]
        if not remaining_covariates:
            break  # No more covariates to add

        candidate_scores = {}

        # Test adding each remaining covariate
        for cov in tqdm(remaining_covariates, desc="Testing covariates"):
            # Create a candidate feature set
            candidate_features = selected_covariates + [cov]
            X_candidate = X_T[candidate_features]
            X_candidate_const = sm.add_constant(
                X_candidate, has_constant='add')

            # Fit the model and get the criterion score
            model = sm.OLS(y, X_candidate_const).fit()
            if stop_criterion.lower() == 'aic':
                candidate_scores[cov] = model.aic
            else:  # BIC
                candidate_scores[cov] = model.bic

        # Find the best covariate to add in this step
        best_new_covariate = min(candidate_scores, key=candidate_scores.get)
        best_candidate_score = candidate_scores[best_new_covariate]

        # --- 4. Stopping Condition Check ---
        # If the best new model is not better than our current best, stop.
        if best_candidate_score >= best_criterion_score:
            logger.warning(
                "\nStopping: No further improvement in the criterion.")
            break

        # Otherwise, update our best model and continue
        best_criterion_score = best_candidate_score
        selected_covariates.append(best_new_covariate)

        if verbose:
            logger.info(
                f"Step {len(selected_covariates)}: Added '{best_new_covariate}', New {stop_criterion} = {best_criterion_score:.4f}")

    # --- 5. Final Model and Results ---
    if not selected_covariates:
        logger.warning("No covariates were selected. The null model was best.")
        return pd.Series(dtype=float)

    # Fit the final model with all selected covariates
    X_final = X_T[selected_covariates]
    X_final_const = sm.add_constant(X_final, has_constant='add')
    final_model = sm.OLS(y, X_final_const).fit()

    if verbose:
        logger.info("\n--- Final Model Summary ---")
        logger.info(final_model.summary())

    # Return the coefficients, excluding the intercept
    # The coefficients Series from statsmodels includes the 'const' term
    final_coefficients = final_model.params.drop('const')
    # final_coefficients = final_model.params
    final_coefficients.name = "coefficients"

    return final_coefficients


def penalized_regression(
    covariate_df: pd.DataFrame,
    response_dict: dict,
    penalty: str = 'lasso',
    n_cv_folds: int = 5,
    verbose: bool = False,
    random_state: int = 42
) -> pd.Series:
    """
    Performs penalized linear regression to find a sparse or regularized model.

    This function uses cross-validation to automatically select the best regularization
    strength (alpha) for Lasso, Ridge, or Elastic Net regression.

    Arguments:
        covariate_df: pd.DataFrame
            The covariates dataframe where each row is a document/chunk (D), and each column
            is an LLM (N). Shape: (D, N).
        response_dict: dict
            A dictionary of <LLM_name, response_score> pairs. Higher scores are better.
        penalty: str
            The type of penalized regression to perform. Options are 'lasso', 'ridge',
            or 'elasticnet'. Default is 'lasso'.
        n_cv_folds: int
            The number of folds to use for cross-validation when tuning the alpha
            hyperparameter. Default is 5.
        verbose: bool
            Whether to print verbose output.
        random_state: int
            The random state to use for the cross-validation. Default is 42.
    Returns:
        pd.Series
            A series where the index contains the names of the selected covariates
            (document IDs) and the values are their corresponding regression coefficients.
            For Lasso, many coefficients will be exactly zero.
    """
    # Set global random state for full reproducibility
    np.random.seed(random_state)
    # --- 1. Align and Prepare Data ---
    # Sort to ensure deterministic ordering across runs
    common_llms = sorted(list(set(covariate_df.columns)
                         & set(response_dict.keys())))
    if len(common_llms) < 2:
        raise ValueError(
            "Need at least 2 common LLMs between the dataframe and the response dictionary.")

    # Transpose the data: LLMs are observations, documents are features.
    X_T = covariate_df[common_llms].T
    y = pd.Series(response_dict)[common_llms]

    # --- 2. Standardize Features ---
    # It is standard practice and highly recommended to scale features before
    # fitting a penalized regression model. This ensures that the penalty is
    # applied fairly to all features, regardless of their original scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_T)

    # --- 3. Select and Fit Model with Cross-Validation ---
    penalty = penalty.lower()
    model = None

    if penalty == 'lasso':
        if verbose:
            logger.info(
                f"Fitting LassoCV with {n_cv_folds}-fold cross-validation...")
        # LassoCV finds the best alpha from a predefined path of values.
        model = LassoCV(cv=n_cv_folds, random_state=random_state,
                        n_jobs=-1).fit(X_scaled, y)
        logger.info(f"Optimal alpha found: {model.alpha_:.4f}")

    elif penalty == 'ridge':
        if verbose:
            logger.info(
                f"Fitting RidgeCV with {n_cv_folds}-fold cross-validation...")
        # RidgeCV is similar but for Ridge regression.
        # It's often good to provide a range of alphas to test.
        alphas_to_test = np.logspace(-6, 6, 100)
        model = RidgeCV(alphas=alphas_to_test, cv=n_cv_folds).fit(X_scaled, y)
        logger.info(f"Optimal alpha found: {model.alpha_:.4f}")

    elif penalty == 'elasticnet':
        if verbose:
            logger.info(
                f"Fitting ElasticNetCV with {n_cv_folds}-fold cross-validation...")
        # ElasticNetCV tunes both alpha and l1_ratio.
        # We'll test a few common l1_ratio values.
        l1_ratios_to_test = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        model = ElasticNetCV(
            l1_ratio=l1_ratios_to_test,
            cv=n_cv_folds,
            random_state=random_state,
            n_jobs=-1
        ).fit(X_scaled, y)
        logger.info(f"Optimal alpha found: {model.alpha_:.4f}")
        logger.info(f"Optimal l1_ratio found: {model.l1_ratio_:.2f}")

    else:
        raise ValueError("penalty must be 'lasso', 'ridge', or 'elasticnet'")

    # --- 4. Extract and Format Results ---
    # The coefficients are for the scaled data. While interpretable in terms of
    # relative importance, they aren't on the original scale. For a true "signature",
    # this is often what you want.
    coefficients = model.coef_

    # Create a Series with document names and their coefficients
    results = pd.Series(coefficients, index=X_T.columns)

    # For creating a sparse signature, it's useful to filter out the zero-coefficient features
    selected_covariates = results[results !=
                                  0].sort_values(key=abs, ascending=False)
    selected_covariates.name = "coefficients"

    return selected_covariates