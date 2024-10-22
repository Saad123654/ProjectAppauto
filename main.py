from src.dataset.load_data import DataLoader
from src.dataset.preprocess import Dataset, Scaler
from src.train.trainer import Trainer
import numpy as np
import torch
import random
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

COMPLETION_METHOD = "average" # "average" or "knn"
DATA_NAME = "base_expe"
MODEL_NAME = "XGBRegressor"
CONFIG_PATH = f"configs/{DATA_NAME}.yaml"

# Load dataset without preprocessing
dataloader = DataLoader(config_path=CONFIG_PATH)
df, target_col, task, all_target_cols = dataloader.load_data(keepID = COMPLETION_METHOD == "average")
# df = dataloader.clean_correlated_features(df, all_target_cols)

# Divide between X_train, y_train, X_test, y_test (90-10)
data = Dataset(target_col, all_target_cols)
if COMPLETION_METHOD == "average":
    df = data.apply_common_prefix(df, "weld_id")
X_train, X_test, y_train, y_test = data.get_train_test(df, test_size=0.1)

# Apply NaN completion based on X_train to X_train, keep the Imputer in memory to apply it on X_test maybe later
scaler = Scaler(
    df, all_target_cols, x_num_scaler_name="standard", x_cat_encoder_name="onehot"
)
X_train, X_test, y_train, y_test = scaler.do_scaling(X_train, X_test, y_train, y_test)
# X_train, imputer = scaler.complete_nan(X_train)
# X_test = imputer.transform(X_test)
# X_train, fill_values, df_numerical = scaler.complete_train_moy(X_train)
# X_test = X_test[df_numerical.columns].fillna(fill_values)
if COMPLETION_METHOD == "average":
    X_train, fill_values, df_numerical = scaler.complete_train_moy(X_train, group_column="Common_Prefix")
    X_test = scaler.fill_test_values(X_test, fill_values, df_numerical, group_column="Common_Prefix")
    X_train = X_train.drop(columns = ["weld_id", "Common_Prefix"])
    X_test = X_test.drop(columns = ["weld_id", "Common_Prefix"])

    X_train, fill_values, df_numerical = scaler.complete_train_moy(X_train)
    X_test = scaler.fill_test_values(X_test, fill_values, df_numerical)
else: # knn case
    X_train, imputer = scaler.complete_nan(X_train)
    X_test = imputer.transform(X_test)
    

# scaler.find_nb_components_pca(X_train)
# Apply PCA with 20 components
X_train, pca = scaler.apply_pca(X_train, 25)
X_test = pca.transform(X_test)

# Divide X_train,y_train in 1 part for each target (so 5 groups) and divide each
# fold in X_train_i, y_train_i, X_test_i, y_test_i (80-20)
all_lists = data.get_train_test_lists_bytarget(X_train, y_train, test_size=0.2)
X_train_list_full = all_lists[0]
y_train_list_full = all_lists[1]
X_test_list_full = all_lists[2]
y_test_list_full = all_lists[3]
X_train_list_no_nan = all_lists[4]
y_train_list_no_nan = all_lists[5]
X_test_list_no_nan = all_lists[6]
y_test_list_no_nan = all_lists[7]

# Train one model for no-NaN y values of X_train_i, y_train_i and evaluate it on X_test_i, y_test_i
models = {}
for X_train_i, y_train_i, X_test_i, y_test_i in zip(
    X_train_list_no_nan, y_train_list_no_nan, X_test_list_no_nan, y_test_list_no_nan
):
    print("Training model for target: ", y_train_i.name)
    print("Number of samples: ", X_train_i.shape[0])
    trainer = Trainer(MODEL_NAME, task, {}, device="cuda")

    trainer.train(X_train_i, y_train_i)
    # params_grid_search = {
    #     "min_child_weight": [1, 5, 10],
    #     "gamma": [0.5, 1, 1.5, 2, 5],
    #     "subsample": [0.6, 0.8, 1.0],
    #     "colsample_bytree": [0.6, 0.8, 1.0],
    #     "max_depth": [3, 4, 5],
    # }
    # trainer.cross_val(
    #     X_train_i, y_train_i, params_grid_search, scoring="neg_root_mean_squared_error"
    # )
    model_i = trainer.model
    models[y_train_i.name] = model_i
    metrics = trainer.get_metrics(X_test_i, y_test_i)
    print(metrics)
    print("\n")

# Predict the other missing values of y_train_i/y_test_i using the model trained on no-NaN y values of X_train_i, y_train_i
y_train_res_list = []
y_test_res_list = []
for X_train_i, y_train_i, X_test_i, y_test_i in zip(
    X_train_list_full, y_train_list_full, X_test_list_full, y_test_list_full
):
    print("Predicting nan values for target: ", y_train_i.name)
    y_train_i_pred = models[y_train_i.name].predict(X_train_i)
    y_test_i_pred = models[y_train_i.name].predict(X_test_i)
    y_train_i_completed = np.where(np.isnan(y_train_i), y_train_i_pred, y_train_i)
    y_test_i_completed = np.where(np.isnan(y_test_i), y_test_i_pred, y_test_i)
    y_train_i_completed = pd.DataFrame(y_train_i_completed, columns=[y_train_i.name])
    y_test_i_completed = pd.DataFrame(y_test_i_completed, columns=[y_test_i.name])
    y_train_res_list.append(y_train_i_completed)
    y_test_res_list.append(y_test_i_completed)

# Merge all y_train_i_pred, y_test_i_pred into y_train
all_y_res = []
for y_train_i, y_test_i in zip(y_train_res_list, y_test_res_list):
    all_y_res.append(pd.concat([y_train_i, y_test_i], axis=0))
y_train_completed = pd.concat(all_y_res, axis=1)

# Apply clustering to the y_train values to improve the pipeline
k = 6
kmeans = KMeans(n_clusters=k, random_state=SEED)
kmeans.fit(y_train_completed)
res = kmeans.labels_
df_pivot = pd.pivot_table(
    y_train_completed, values=y_train_completed.columns, index=res, aggfunc="mean"
)
print(df_pivot)

# Test the final pipeline to the original X_test
