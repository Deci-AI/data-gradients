import numpy as np


def get_top_values(df, id_col, split_col, val_col, mode):
    """
    Returns the top 5 rows of the DataFrame based on the mode.
    The DataFrame is expected to have three columns: id_col, split_col, val_col.

    Modes:
    'gap' - Returns rows with the biggest gap between 'train' and 'val' split values.
    'outliers' - Returns rows with the most extreme average split values.
    'max' - Returns rows with the highest average split values.
    'min' - Returns rows with the lowest average split values.
    """
    # Verify inputs
    for col in [id_col, split_col, val_col]:
        if col not in df.columns:
            raise ValueError(f"{col} is not a column in the DataFrame")
    print(id_col, split_col, val_col)

    # the mean of val_col for each id_col/split_col
    df_mean = df.groupby([id_col, split_col])[val_col].mean().reset_index()

    # Pivot DataFrame to have 'train' and 'val' as columns
    df_pivot = df_mean.pivot(index=id_col, columns=split_col, values=val_col)

    # Calculate the relative difference or average based on the mode
    if mode == "gap":
        df_pivot["metric"] = np.abs((df_pivot["train"] - df_pivot["val"]) / ((df_pivot["train"] + df_pivot["val"]) / 2))
    elif mode in ["outliers", "max", "min"]:
        df_pivot["metric"] = (df_pivot["train"] + df_pivot["val"]) / 2

    # Calculate the z-score if mode is 'outliers'
    if mode == "outliers":
        mean, std = df_pivot["metric"].mean(), df_pivot["metric"].std()
        df_pivot["metric"] = (df_pivot["metric"] - mean).abs() / std

    # Get the top 5 class_ids based on the metric
    if mode in ["gap", "outliers", "max"]:
        top_ids = df_pivot.nlargest(5, "metric").index
    elif mode == "min":
        top_ids = df_pivot.nsmallest(5, "metric").index
    else:
        raise ValueError("Invalid mode. Expected one of: gap, outliers, max, min")

    # Filter the original DataFrame to only include rows with the top class_ids
    return df[df[id_col].isin(top_ids)]
