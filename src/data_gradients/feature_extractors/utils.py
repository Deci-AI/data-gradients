import pandas as pd


def keep_most_frequent(df: pd.DataFrame, filtering_key: str, frequency_key: str, top_k: int) -> pd.DataFrame:
    bbox_cum_area_per_class = df.groupby(filtering_key)[frequency_key].sum()
    most_common_classes = bbox_cum_area_per_class.sort_values(ascending=False)[:top_k].index
    return df[df[filtering_key].isin(most_common_classes)]
