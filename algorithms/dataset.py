from pathlib import Path
import json
from montydb import MontyClient
import yaml


class DataBase:
    def __init__(self):
        self.drop = False
        self.client = None

    def drop_databases(self):
        client = MontyClient(":memory:")
        database_names = client.list_database_names()
        for db_name in database_names:
            client.drop_database(db_name)
        client.close()
        self.drop = True

    def initialize_database(self):
        self.client = MontyClient(":memory:")
        db = self.client.assigment_db
        return db


class DataLakeLoader:
    def __init__(self,
                 db,
                 data_path: str,
                 dim_path: str,
                 items_predict_path: str,
                 items_path: str,
                 items_bim_path: str):

        self.db = db
        self.data_path = data_path
        self.dim_path = dim_path
        self.items_path = items_path
        self.items_bim_path = items_bim_path
        self.items_predict_path = items_predict_path

    def get_aulas(self):
        with open(f'{self.dim_path}/aulas.json', "r") as f:
            return json.load(f)

    def get_reward_sedes(self):
        with open(f'{self.dim_path}/rewards_sedes.json', "r") as f:
            return json.load(f)

    def dim_loader(self):
        aulas = self.get_aulas()
        reward_sedes = self.get_reward_sedes()
        self.db.aulas.insert_many(aulas)
        self.db.reward_sedes.insert_many(reward_sedes)

    def items_loader(self):
        # self = DataLakeLoader("data", "dim", "items", "items_bim", "items_predict")
        #
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path/self.items_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        self.db.items.insert_many(collection)

    def items_bim_loader(self):
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_bim_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        self.db.items.insert_many(collection)

    def items_predict_loader(self):
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_predict_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        self.db.items.insert_many(collection)

    def load_all(self):
        self.dim_loader()
        self.items_loader()
        self.items_bim_loader()
        self.items_predict_loader()


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    periodo_predict = config['periodo_predict']
    ult_periodo = config['ult_periodo']
    n_periodos = config['n_periodos']
    full = config['full']
    db_path = config['db_path']
    excel_path = config['excel_path']