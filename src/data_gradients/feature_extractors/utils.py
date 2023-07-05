import numpy as np
import pandas as pd


class MostImportantValuesSelector:
    def __init__(self, topk: int, prioritization_mode: str):
        """
        :param topk:    How many rows (per split) to return.
        :param prioritization_mode:    The prioritization_mode to get the top values for. One of:
                - 'train_val_diff':         Returns rows with the biggest train_val_diff between 'train' and 'val' split values.
                - 'outliers':    Returns rows with the most extreme average values.
                - 'max':         Returns rows with the highest average values.
                - 'min':         Returns rows with the lowest average values.
        """
        valid_modes = ("train_val_diff", "outliers", "max", "min")
        if prioritization_mode not in valid_modes:
            raise ValueError(f"Invalid `prioritization_mode={prioritization_mode}'. Must be one of: {valid_modes}.")
        self.topk = topk
        self.prioritization_mode = prioritization_mode

    def select(self, df: pd.DataFrame, id_col: str, split_col: str, value_col: str):
        """
        Returns the top 5 rows of the DataFrame based on the prioritization_mode.
        The DataFrame is expected to have three columns: id_col, split_col, val_col.

        :param df:          The DataFrame to get the top values from.
        :param id_col:      The name of the id column.
        :param split_col:   The name of the split column.
        :param value_col:   The name of the value column.
        """
        # Verify inputs
        for col in [id_col, split_col, value_col]:
            if col not in df.columns:
                raise ValueError(f"{col} is not a column in the DataFrame")

        # the mean of val_col for each id_col/split_col
        df_mean = df.groupby([id_col, split_col])[value_col].mean().reset_index()

        # Pivot DataFrame to have 'train' and 'val' as columns
        df_pivot = df_mean.pivot(index=id_col, columns=split_col, values=value_col)

        # Calculate the relative difference or average based on the prioritization_mode
        if self.prioritization_mode == "train_val_diff":
            df_pivot["metric"] = np.abs((df_pivot["train"] - df_pivot["val"]) / ((df_pivot["train"] + df_pivot["val"]) / 2))
        elif self.prioritization_mode in ["outliers", "max", "min"]:
            df_pivot["metric"] = (df_pivot["train"] + df_pivot["val"]) / 2

        if self.prioritization_mode == "outliers":
            mean, std = df_pivot["metric"].mean(), df_pivot["metric"].std()
            df_pivot["metric"] = (df_pivot["metric"] - mean).abs() / std

        # Only return the top k.
        if self.prioritization_mode in ["train_val_diff", "outliers", "max"]:
            top_ids = df_pivot.nlargest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        elif self.prioritization_mode == "min":
            top_ids = df_pivot.nsmallest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        else:
            raise NotImplementedError(f"Mode {self.prioritization_mode} is not implemented")
