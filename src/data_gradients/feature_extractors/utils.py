import numpy as np
import pandas as pd


class MostImportantValuesSelector:
    def __init__(self, topk: int, mode: str):
        """
        :param topk:    How many rows (per split) to return.
        :param mode:    The mode to get the top values for. One of:
                - 'gap':         Returns rows with the biggest gap between 'train' and 'val' split values.
                - 'outliers':    Returns rows with the most extreme average values.
                - 'max':         Returns rows with the highest average values.
                - 'min':         Returns rows with the lowest average values.
        """
        valid_modes = ("gap", "outliers", "max", "min")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}.")
        self.topk = topk
        self.mode = mode

    def select(self, df: pd.DataFrame, id_col: str, split_col: str, value_col: str):
        """
        Returns the top 5 rows of the DataFrame based on the mode.
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

        # Calculate the relative difference or average based on the mode
        if self.mode == "gap":
            df_pivot["metric"] = np.abs((df_pivot["train"] - df_pivot["val"]) / ((df_pivot["train"] + df_pivot["val"]) / 2))
        elif self.mode in ["outliers", "max", "min"]:
            df_pivot["metric"] = (df_pivot["train"] + df_pivot["val"]) / 2

        if self.mode == "outliers":
            mean, std = df_pivot["metric"].mean(), df_pivot["metric"].std()
            df_pivot["metric"] = (df_pivot["metric"] - mean).abs() / std

        # Only return the top k.
        if self.mode in ["gap", "outliers", "max"]:
            top_ids = df_pivot.nlargest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        elif self.mode == "min":
            top_ids = df_pivot.nsmallest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")
