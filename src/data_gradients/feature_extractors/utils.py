import pandas as pd


class MostImportantValuesSelector:
    def __init__(self, topk: int, prioritization_mode: str):
        """
        :param topk:    How many rows (per split) to return.
        :param prioritization_mode:    The prioritization_mode to get the top values for. One of:
                - 'train_val_diff': Returns the top k rows with the biggest train_val_diff between 'train' and 'val' split values.
                - 'outliers':       Returns the top k rows with the most extreme average values.
                - 'max':            Returns the top k rows with the highest average values.
                - 'min':            Returns the top k rows with the lowest average values.
                - 'min_max':        Returns the (top k)/2 rows with the biggest average values, and the (top k)/2 with the smallest average values.
        """
        valid_modes = ("train_val_diff", "outliers", "max", "min", "min_max")
        if prioritization_mode not in valid_modes:
            raise ValueError(f"Invalid `prioritization_mode={prioritization_mode}'. Must be one of: {valid_modes}.")
        self.topk = topk
        self.prioritization_mode = prioritization_mode

    def select(self, df: pd.DataFrame, id_col: str, split_col: str, value_col: str):
        """
        Returns the top k rows of the DataFrame based on the prioritization_mode.
        The DataFrame is expected to have at least three columns: id_col, split_col, val_col.

        :param df:          The DataFrame to get the top values from.
        :param id_col:      The name of the id column.
        :param split_col:   The name of the split column. (Usually 'split')
        :param value_col:   The name of column that will be used to calculate the metric.
        :return: The Dataframe with only the rows associated to the most important values.
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
            # `train_val_diff` only defined when working with 2 sets.
            if len(df_pivot.columns) != 2:
                raise ValueError(f'`prioritization_mode"train_val_diff"` is only supported when working with 2 sets. Found {len(df_pivot.columns)}.')
            delta = (df_pivot.iloc[:, 0] - df_pivot.iloc[:, 1]).abs()
            average = (df_pivot.iloc[:, 0] + df_pivot.iloc[:, 1]).abs() / 2
            df_pivot["metric"] = delta / (average + 1e-6)
        elif self.prioritization_mode in ["outliers", "max", "min", "min_max"]:
            df_pivot["metric"] = df_pivot.mean(1)

        if self.prioritization_mode == "outliers":
            mean, std = df_pivot["metric"].mean(), df_pivot["metric"].std()
            df_pivot["metric"] = (df_pivot["metric"] - mean).abs() / (std + 1e-6)

        # Only return the top k.
        if self.prioritization_mode in ["train_val_diff", "outliers", "max"]:
            top_ids = df_pivot.nlargest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        elif self.prioritization_mode == "min":
            top_ids = df_pivot.nsmallest(self.topk, "metric").index
            return df[df[id_col].isin(top_ids)]
        elif self.prioritization_mode == "min_max":
            n_max_results = self.topk // 2
            n_min_results = self.topk - n_max_results

            top_ids = df_pivot.nlargest(n_max_results, "metric").index

            n_rows_available = len(df_pivot) - len(top_ids)
            bottom_ids = df_pivot.nsmallest(min(n_min_results, n_rows_available), "metric").index
            return pd.concat([df[df[id_col].isin(top_ids)], df[df[id_col].isin(bottom_ids)]])
        else:
            raise NotImplementedError(f"Mode {self.prioritization_mode} is not implemented")
