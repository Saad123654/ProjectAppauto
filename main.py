from src.dataset.load_data import load_data
from src.dataset.preprocess import Dataset
from src.train.trainer import Trainer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DATA_NAME = "base_expe"
MODEL_NAME = "XGBRegressor"
CONFIG_PATH = f"configs/{DATA_NAME}.yaml"

df, target_col, task, _, all_target_cols = load_data(
    from_cleaned=False, config_path=CONFIG_PATH, keep_corr_features=False
)
data = Dataset(df, target_col)
X_train, X_test, y_train, y_test = data.get_train_test(
    test_size=0.2,
    scaler_params={"x_num_scaler_name": "standard", "x_cat_encoder_name": "onehot"},
)
trainer = Trainer(
    MODEL_NAME, task, {"input_dim": X_train.shape[1], "output_dim": 1}, device="cuda"
)
nan_indexes_train = y_train[y_train.isna()].index
X_train = X_train.drop(nan_indexes_train)
y_train = y_train.drop(nan_indexes_train)
nan_indexes_test = y_test[y_test.isna()].index
X_test = X_test.drop(nan_indexes_test)
y_test = y_test.drop(nan_indexes_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
trainer.train(X_train, y_train)
metrics = trainer.get_metrics(X_test, y_test)
print(metrics)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
X_train["Cluster"] = kmeans.labels_
X_train["TSNE1"], X_train["TSNE2"] = (
    TSNE(n_components=2).fit_transform(X_train)[:, 0],
    TSNE(n_components=2).fit_transform(X_train)[:, 1],
)
plt.scatter(
    X_train["TSNE1"],
    X_train["TSNE2"],
    c=X_train["Cluster"],
    cmap="viridis",
    marker="o",
    edgecolor="k",
    s=100,
)
plt.show()
