import os

import pandas


class ConstructDataset:
    def __init__(self, data_dir):
        # Directory containing core data files
        self.data_dir = data_dir

        # Filenames
        # ---------
        # ACIL Allen data
        self.acil_workbook = 'Fuel_and_Technology_Cost_Review_Data_ACIL_Allen.xlsx'

    def _load_acil_(self):
