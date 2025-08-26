import json
import pandas as pd


def load_merged_data(file_path: str) -> pd.DataFrame:
    """
    Load the merged data from the file path.

    Parameters
    ----------
    file_path : str
        Path to the CSV or Parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only valid rows (no NaN or empty strings).
    """
    file_type = file_path.split(".")[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Identify invalid rows
    # Strip strings before checking empties, so "   " counts as empty
    df_stripped = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    invalid_mask = df_stripped.isna() | (df_stripped == "")
    invalid_rows = invalid_mask.any(axis=1)

    # Print count of invalid rows
    print(f"Number of invalid rows: {invalid_rows.sum()}")

    # Return only valid rows
    return df[~invalid_rows]


def load_llm_responses(file_path: str) -> dict:
    """
    Load the LLM responses from the file path.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary containing the LLM responses.
    """
    assert file_path.endswith(".json"), "File must be a JSON file"

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def load_simple_original_text(parquet_path: str) -> pd.DataFrame:
    """
    Load the simple original text from the parquet file.
    """
    df = pd.read_parquet(parquet_path)
    # truncate the "text" column to 1000 characters
    df["text"] = df["text"].str[:1000]
    return df[["text", "domain"]]


if __name__ == "__main__":
    path = "data/merged_results_avg_logprob.csv"
    df = load_merged_data(path)

    # print the head and summary statistics of the dataframe
    print(df.head())
    print(df.describe())
    print(df.columns)

    response_path = "data/wikiqa.json"
    response_dict = load_llm_responses(response_path)
    print(response_dict)