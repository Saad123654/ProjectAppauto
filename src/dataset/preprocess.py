from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA


class Scaler:
    """Class for scaling the data.

    Attributes:
        df (pd.DataFrame): input dataframe
        self.all_target_cols (str): target column names
        categorical_cols (list): list of categorical columns
        x_num_scaler_name (str): scaler to use for x. Must be either
            None or one of standard, minmax, quantile_normal, quantile_uniform,
            maxabs, robust. Defaults to None.
        x_cat_encoder_name (str): scaler to use for x. Must be either None or
            one of labelencoder or onehotencoder. Defaults to None.
        y_scaler_name (str): scaler to use for y. Must be either None or one
            of standard, minmax, quantile_normal, quantile_uniform, maxabs,
            robust or labelencoder. Defaults to None.
        cat_not_to_onehot (List[str]): list of categorical columns not to one
            hot encode. Defaults to [].
        scalers (dict): dictionary of possible scalers

    Methods:
       encode_categorical: encode categorical columns in one hot or label encoding
       do_scaling: process data from categorical features first to numerical features scaling

    Args:
        df (pd.DataFrame): input dataframe
        self.all_target_cols (str, optional): target column names. Defaults to None.
        x_num_scaler_name (Optional[str], optional): scaler to use for x.
            Must be either None or one of standard, minmax, quantile_normal,
            quantile_uniform, maxabs, robust. Defaults to None.
        x_cat_encoder_name (Optional[str], optional): scaler to use for x.
            Must be either None or one of labelencoder or onehotencoder. Defaults to None.
        y_scaler_name (Optional[str], optional): scaler to use for y.
            Must be either None or one of standard, minmax, quantile_normal,
            quantile_uniform, maxabs, robust or labelencoder. Defaults to None.
        cat_not_to_onehot (Optional[List[str]], optional): list of
            categorical columns not to one hot encode. Defaults to [].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        all_target_cols: Optional[str] = None,
        x_num_scaler_name: Optional[str] = None,
        x_cat_encoder_name: Optional[str] = None,
        y_scaler_name: Optional[str] = None,
        cat_not_to_onehot: Optional[List[str]] = [],
    ):
        self.df = df
        self.all_target_cols = all_target_cols
        self.categorical_cols = list(set(df.select_dtypes(include=["object"]).columns))
        self.x_num_scaler_name = x_num_scaler_name
        self.x_cat_encoder_name = x_cat_encoder_name
        self.y_scaler_name = y_scaler_name
        self.cat_not_to_onehot = cat_not_to_onehot
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "quantile_normal": QuantileTransformer(output_distribution="normal"),
            "quantile_uniform": QuantileTransformer(output_distribution="uniform"),
            "maxabs": MaxAbsScaler(),
            "robust": RobustScaler(),
            "labelencoder": LabelEncoder(),
            "ordinalencoder": OrdinalEncoder(),
            "onehot": OneHotEncoder(),
        }
        assert (
            self.x_num_scaler_name is None or self.x_num_scaler_name in self.scalers
        ), "x_num_scaler_name must be None or \
         one of standard, minmax, quantile_normal, quantile_uniform, maxabs, robust or ordinalencoder"
        assert (
            self.x_cat_encoder_name is None
            or self.x_cat_encoder_name
            in [
                "labelencoder",
                "onehot",
                "ordinalencoder",
            ]
        ), "x_cat_encoder_name must be None or one of labelencoder or onehotencoder or ordinalencoder"
        assert (
            self.y_scaler_name is None or self.y_scaler_name in self.scalers
        ), "y_scaler_name must be None or \
         one of standard, minmax, quantile_normal, quantile_uniform, maxabs, robust or labelencoder"

    def __cat_cols__(self, x: pd.DataFrame) -> List[str]:
        """Check if an array represents categorical data."""
        cat_cols_names = []
        for col in x.columns:
            if (
                x[col].dtype in [np.int64, np.int32, np.int16, np.int8]
                and np.max(np.abs(x[col])) < 2
            ):
                cat_cols_names.append(col)
        return cat_cols_names

    def complete_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete missing values in the dataframe using KNNImputer.

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: dataframe with completed missing values
            sklearn.impute.KNNImputer: imputer
        """
        imputer = KNNImputer(n_neighbors=5)
        df_numerical = df.select_dtypes(include=np.number)
        df_copy = df.copy()
        df_copy[df_numerical.columns] = imputer.fit_transform(df_numerical)
        df_copy = df_copy.reset_index(drop=True)
        df_copy = df_copy.astype(df.dtypes.to_dict())
        return df_copy, imputer

    def apply_pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Apply PCA to the dataframe.

        Args:
            df (pd.DataFrame): dataframe
            n_components (int): number of components

        Returns:
            pd.DataFrame: dataframe with PCA applied
        """
        pca = PCA(n_components=n_components)
        df_copy = df.copy()
        df_copy = pca.fit_transform(df_copy)
        columns = [f"component_{i}" for i in range(n_components)]
        print("Explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
        df_copy = pd.DataFrame(df_copy, columns=columns, index=df.index)
        return df_copy, pca

    def encode_categorical(
        self,
        x_train: pd.DataFrame,
        x_test: Optional[pd.DataFrame],
        x_cat_encoder: Optional[str] = "labelencoder",
        cat_not_to_onehot: Optional[List[str]] = [],
    ) -> pd.DataFrame:
        """Encode categorical columns with LabelEncoder or OneHotEncoder.

        Args:
            x_train (pd.DataFrame): train dataframe with categorical columns
            x_test (pd.DataFrame): test dataframe with categorical columns
            x_cat_encoder (Optional[str], optional): encoding type for categorical columns
            cat_not_to_onehot (Optional[List[str]], optional): list of
                categorical columns not to one hot encode. For example if the
                dimensionality is too high. Defaults to [].

        Returns:
            pd.DataFrame: dataframe with encoded categorical columns
        """
        categorical_cols = list(set(self.categorical_cols) - set(self.all_target_cols))
        x_traincp, x_testcp = x_train.copy(), x_test.copy()
        for col in categorical_cols:
            if col not in cat_not_to_onehot and x_cat_encoder == "onehot":
                one = OneHotEncoder()
                one_hot_train = one.fit_transform(
                    x_traincp[col].values.reshape(-1, 1)
                ).toarray()
                x_traincp = x_traincp.drop(col, axis=1)
                for i in range(one_hot_train.shape[1]):
                    x_traincp[col + "_" + str(i)] = one_hot_train[:, i]
                one_hot_test = one.transform(
                    x_testcp[col].values.reshape(-1, 1)
                ).toarray()
                x_testcp = x_testcp.drop(col, axis=1)
                for i in range(one_hot_test.shape[1]):
                    x_testcp[col + "_" + str(i)] = one_hot_test[:, i]
            else:
                le = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                x_traincp[col] = le.fit_transform(x_traincp[col].values.reshape(-1, 1))
                x_testcp[col] = le.transform(x_testcp[col].values.reshape(-1, 1))
        return x_traincp, x_testcp

    def do_scaling(
        self,
        x_train: pd.DataFrame,
        x_test: Optional[pd.DataFrame],
        y_train: pd.Series,
        y_test: Optional[pd.Series],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits.

        Args:
            x_train (pd.DataFrame): train dataframe
            x_test (pd.DataFrame): test dataframe
            y_train (pd.Series): train targets
            y_test (pd.Series): test targets

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """

        x_num_scaler = (
            self.scalers[self.x_num_scaler_name]
            if self.x_num_scaler_name is not None
            else None
        )
        y_scaler = (
            self.scalers[self.y_scaler_name] if self.y_scaler_name is not None else None
        )

        if self.x_cat_encoder_name is not None:
            x_traincp, x_testcp = self.encode_categorical(
                x_train, x_test, self.x_cat_encoder_name, self.cat_not_to_onehot
            )
        else:
            x_traincp, x_testcp = x_train, x_test
        y_traincp, y_testcp = y_train, y_test

        self.categorical_cols += self.__cat_cols__(x_traincp)
        x_num_cols = x_traincp.columns[~x_traincp.columns.isin(self.categorical_cols)]
        x_train_num = x_traincp.loc[:, ~x_traincp.columns.isin(self.categorical_cols)]
        x_train_num = x_train_num.astype(np.float32)
        x_test_num = x_testcp.loc[:, ~x_testcp.columns.isin(self.categorical_cols)]
        x_test_num = x_test_num.astype(np.float32)
        if self.x_num_scaler_name is not None and x_train_num.shape[1] > 0:
            x_train_num = x_num_scaler.fit_transform(x_train_num)
            x_train_num = pd.DataFrame(x_train_num, columns=x_num_cols)
            x_test_num = x_num_scaler.transform(x_test_num)
            x_test_num = pd.DataFrame(x_test_num, columns=x_num_cols)
        if self.y_scaler_name is not None:
            if self.y_scaler_name == "labelencoder" and y_traincp.dtype in [
                np.int64,
                np.int32,
                str,
            ]:
                y_traincp = y_scaler.fit_transform(y_traincp)
                y_traincp = pd.DataFrame(y_traincp, columns=self.all_target_cols)
                y_traincp = y_traincp - y_traincp.min()
                y_testcp = y_scaler.transform(y_testcp)
                y_testcp = pd.DataFrame(y_testcp, columns=self.all_target_cols)
                y_testcp = y_testcp - y_testcp.min()
            else:
                y_traincp = y_scaler.fit_transform(y_traincp.values.reshape(-1, 1))
                y_traincp = pd.DataFrame(
                    y_traincp.flatten(), columns=self.all_target_cols
                )
                y_testcp = y_scaler.transform(y_testcp.values.reshape(-1, 1))
                y_testcp = pd.DataFrame(
                    y_testcp.flatten(), columns=self.all_target_cols
                )
        x_train_cat = x_traincp.loc[:, x_traincp.columns.isin(self.categorical_cols)]
        x_train_cat = x_train_cat.reset_index(drop=True)
        x_train_num = x_train_num.reset_index(drop=True)
        x_traincp = x_train_num.join(x_train_cat)
        y_traincp = y_traincp.reset_index(drop=True)
        x_test_cat = x_testcp.loc[:, x_testcp.columns.isin(self.categorical_cols)]
        x_test_cat = x_test_cat.reset_index(drop=True)
        x_test_num = x_test_num.reset_index(drop=True)
        x_testcp = x_test_num.join(x_test_cat)
        y_testcp = y_testcp.reset_index(drop=True)
        return x_traincp, x_testcp, y_traincp, y_testcp


class Dataset:
    """Dataset class to create train/test splits.

    Attributes:
        target_name (str): name of the target column
        all_targets_name (Optional[List[str]]): list of all target columns

    Methods:
        get_train_test: create train/test splits
        get_classes_num: get the number of classes for the task

    Args:
        target_name (str): name of the target column
        all_targets_name (Optional[List[str]]): list of all target columns
    """

    def __init__(
        self,
        target_name: str,
        all_targets_name: Optional[List[str]] = None,
    ):
        self.target_name = target_name
        self.all_targets_name = all_targets_name

    def get_train_test(
        self,
        data,
        test_size: float = 0.2,
        scaler_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits.

        Args:
            test_size (float, optional): test size. Defaults to 0.2.
            scaler_params (Optional[Dict[str, str]], optional): scaling parameters.
                Defaults to None.

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        x, y = data.drop(self.all_targets_name, axis=1), data[self.all_targets_name]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )
        if scaler_params is not None:
            scaler = Scaler(x_train, self.target_name, **scaler_params)
            x_train, x_test, y_train, y_test = scaler.do_scaling(
                x_train, x_test, y_train, y_test
            )
        x_train, x_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        return x_train, x_test, y_train, y_test

    def get_train_test_by_target(
        self, x, y, target_name: str, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/test splits by target.

        Args:
            target_name (str): name of the target column
            test_size (float, optional): test size. Defaults to 0.2.

        Returns:
            tuple: x_train, y_train, x_test, y_test
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        x_train, x_test, y_train, y_test = train_test_split(
            x, y[target_name], test_size=test_size, random_state=42
        )
        x_train, x_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        return x_train, y_train, x_test, y_test

    def get_no_nan_values(self, x, y):
        """Keep only non-NaN values from y."""
        nan_idx = np.where(np.isnan(y))[0]
        x_nonan = x.drop(nan_idx)
        y_no_nan = y.drop(nan_idx)
        x_nonan = x_nonan.reset_index(drop=True)
        y_no_nan = y_no_nan.reset_index(drop=True)
        return x_nonan, y_no_nan

    def get_train_test_lists_bytarget(self, x, y, test_size: float = 0.2):
        X_train_list_full = []
        y_train_list_full = []
        X_test_list_full = []
        y_test_list_full = []

        X_train_list_no_nan = []
        y_train_list_no_nan = []
        X_test_list_no_nan = []
        y_test_list_no_nan = []
        for target in self.all_targets_name:
            X_train_i, y_train_i, X_test_i, y_test_i = self.get_train_test_by_target(
                x, y, target, test_size=test_size
            )
            X_train_list_full.append(X_train_i)
            y_train_list_full.append(y_train_i)
            X_test_list_full.append(X_test_i)
            y_test_list_full.append(y_test_i)
            X_train_i_no_nan, y_train_i_no_nan = self.get_no_nan_values(
                X_train_i, y_train_i
            )
            X_train_list_no_nan.append(X_train_i_no_nan)
            y_train_list_no_nan.append(y_train_i_no_nan)
            X_test_i_no_nan, y_test_i_no_nan = self.get_no_nan_values(
                X_test_i, y_test_i
            )
            X_test_list_no_nan.append(X_test_i_no_nan)
            y_test_list_no_nan.append(y_test_i_no_nan)
        return (
            X_train_list_full,
            y_train_list_full,
            X_test_list_full,
            y_test_list_full,
            X_train_list_no_nan,
            y_train_list_no_nan,
            X_test_list_no_nan,
            y_test_list_no_nan,
        )
