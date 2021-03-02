import pathlib
import random
import matplotlib
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from deep_learning_for_time_series.MQRNN.mqrnn_model import MQRNNModel
from deep_learning_for_time_series.MQRNN.electricity_load_dataset import ElectricityLoadDataModule

matplotlib.use('TkAgg')
plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (6, 6)

DATA_DIR = pathlib.Path('../../yandex/data/')
eldata = pd.read_parquet(DATA_DIR.joinpath("LD2011_2014.parquet"))
eldata = eldata.resample("1H", on="timestamp").mean()

scaled_data = eldata / eldata[eldata != 0].mean() - 1
dm = ElectricityLoadDataModule(scaled_data, samples=100, batch_size=128)
dm.setup()
hist_len = 168
fct_len = 24
model = MQRNNModel(fct_len=fct_len)

trainer = pl.Trainer(max_epochs=25, progress_bar_refresh_rate=1, auto_select_gpus=True)
trainer.fit(model, dm)


# TEST
dm.setup(stage="test")
batch = next(iter(dm.test_dataloader()))
X, y = batch
future_covariates = y[:, :, 1:]
result = model(X, future_covariates).detach().numpy()
plt.figure(1)
for i in range(16):
    s = random.randint(0, X.shape[0] - 1)
    plt.subplot(4, 4, i+1)
    plt.plot(range(fct_len), result[s, 1, :, -1], 'b')
    plt.fill_between(range(fct_len), result[s, 0, :, -1], result[s, 2, :, -1], color='b', alpha=.1)
    plt.plot(y[s, :, 0], 'r')
plt.show()
