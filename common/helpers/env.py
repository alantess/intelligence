import cudf

class Env(object):
    def __init__(self,csv="/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv"):
        self.csv = csv
        self.df = None
        self._load()

    def _load(self):
        self.df = cudf.read_csv(self.csv)
