from pathlib import Path
import json
from montydb import MontyClient
import yaml


class DataBase:
    def __init__(self):
        print("Create a client in memory")
        self.client = MontyClient(":memory:")

    def drop_dbs(self):
        """
        Drop all databases in the client
        :return:
        """
        database_names = self.client.list_database_names()
        for db_name in database_names:
            self.client.drop_database(db_name)

    def initialize_database(self):
        db = self.client.assigment_db
        return db

    def close(self):
        self.client.close()


class DataLakeLoader:
    def __init__(self,
                 database,
                 data_path: str,
                 dim_path: str,
                 items_predict_path: str,
                 items_path: str,
                 items_bim_path: str):

        self.database = database
        self.data_path = data_path
        self.dim_path = dim_path
        self.items_path = items_path
        self.items_bim_path = items_bim_path
        self.items_predict_path = items_predict_path

    def get_aulas(self):
        with open(f'{self.data_path}/{self.dim_path}/aulas.json', "r") as f:
            return json.load(f)

    def get_reward_sedes(self):
        with open(f'{self.data_path}/{self.dim_path}/rewards_sedes.json', "r") as f:
            return json.load(f)

    def dim_loader(self):
        db = self.database.initialize_database()
        aulas = self.get_aulas()
        reward_sedes = self.get_reward_sedes()
        db.aulas.insert_many(aulas)
        db.reward_sedes.insert_many(reward_sedes)

    def items_loader(self):
        # self = DataLakeLoader("data", "dim", "items", "items_bim", "items_predict")
        #
        db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path/self.items_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        db.items.insert_many(collection)

    def items_bim_loader(self):
        db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_bim_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        db.items_bim.insert_many(collection)

    def items_predict_loader(self):
        db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_predict_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        db.items_predict.insert_many(collection)

    def room_log_loader(self):
        db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_predict_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        db.items_predict.insert_many(collection)

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