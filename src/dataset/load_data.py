import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from yaml.loader import SafeLoader


def get_path(path, check_dir=False):
    """Get the path of a file or directory."""
    dir_path = path.split("/")[:-1]
    dir_path = "/".join(dir_path)
    path_exists = os.path.exists(dir_path) if check_dir else os.path.exists(path)
    if not path_exists:
        from_parent_exists = (
            os.path.exists("../" + dir_path)
            if check_dir
            else os.path.exists("../" + path)
        )
        if from_parent_exists:
            new_path = "../" + path
        else:
            raise FileNotFoundError(f"Path {path} does not exist")
    else:
        new_path = path
    return new_path


class DataCleaner:
    """Clean the data by removing correlated features.

    Attributes:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name
        corr_threshold (float): correlation threshold

    Methods:
        compute_correlation_matrix: compute the correlation matrix
        plot_corr_matrix: plot the correlation matrix
        remove_correlated_features: remove correlated features from the
            dataframe with a threshold
        clean_data: clean the data

    Args:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name
        corr_threshold (float, optional): correlation threshold.
            Defaults to 0.7.
    """

    def __init__(self, df, target_col: str, corr_threshold: float = 0.8):
        self.df = df
        self.corr_threshold = corr_threshold
        self.target_col = target_col

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the correlation matrix.

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: correlation matrix
        """
        return df.corr(numeric_only=True).abs()

    def plot_corr_matrix(self, df: pd.DataFrame) -> None:
        """Plot the correlation matrix.

        Args:
            df (pd.DataFrame): dataframe
        """
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True)
        plt.show()

    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features from the dataframe with a threshold.

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: dataframe without correlated features
        """
        df_copy = df.drop(self.target_col, axis=1)
        corr_matrix = self.compute_correlation_matrix(df_copy)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.corr_threshold)
        ]
        df_copy = df_copy.drop(df[to_drop], axis=1)
        df_copy[self.target_col] = df[self.target_col]
        return df_copy

    def clean_data(self) -> pd.DataFrame:
        """Clean the data.

        Returns:
            pd.DataFrame: dataframe
        """
        df = self.remove_correlated_features(self.df)
        return df


def read_config(config_path: str) -> dict:
    """Read the config file.

    Args:
        config_path (str): path to the config file

    Returns:
        dict: config file
    """
    config_path = get_path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config


class DataLoader:
    """Load the data.

    Args:
        config_path (str): path to the config file

    Returns:
        pd.DataFrame: dataframe
    """

    def __init__(self, config_path: str):
        self.config = read_config(config_path)
        self.path = self.config["path"]
        self.target_col = self.config["target_col"]
        self.task = self.config["task"]
        self.all_target_cols = self.config["all_targets"]

    def load_csv(self) -> pd.DataFrame:
        """Load the csv file.

        Returns:
            pd.DataFrame: dataframe
        """
        return pd.read_csv(self.path, index_col=0)

    def load_data(self) -> list:
        """Provides a fast way to load data and preprocess it.

        Returns:
            list: a list containing the data, the target column name and the task
        """
        data = self.load_csv()

        return {
            "data": data,
            "target_col": self.target_col,
            "task": self.task,
            "all_target_cols": self.all_target_cols,
        }.values()

    def clean_correlated_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        corr_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Clean the data by removing correlated features.

        Args:
            data (pd.DataFrame): input dataframe
            target_col (str): target column name
            corr_threshold (float, optional): correlation threshold.
                Defaults to 0.7.

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        data_cleaner = DataCleaner(data, target_col, corr_threshold)
        data = data_cleaner.clean_data()
        return data
