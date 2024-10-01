from src.dataset.load_data import load_data
from src.dataset.preprocess import Dataset
from src.train.trainer import Trainer

DATA_NAME = "base_expe"
MODEL_NAME = "XGBRegressor"
CONFIG_PATH = f"configs/{DATA_NAME}.yaml"

df, target_col, task, _ = load_data(from_cleaned=False, config_path=CONFIG_PATH)
data = Dataset(df, target_col)
X_train, X_test, y_train, y_test = data.get_train_test()

NN_PARAMS = {"input_dim": X_train.shape[1], "output_dim": 1}
trainer = Trainer(MODEL_NAME, task, NN_PARAMS)
trainer.train(X_train, y_train)
