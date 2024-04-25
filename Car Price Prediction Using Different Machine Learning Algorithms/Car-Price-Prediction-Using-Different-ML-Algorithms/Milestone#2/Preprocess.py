import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import collections
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Preprocessing:
    def __init__(self, data_frame, flag) -> None:
        self.df = data_frame
        self.flag = flag

    def run_preprocess(self):
        self.apply_to_all()
        self.drop()
        self.fill_null()
        self.standarizing()
        return self.df

    def drop(self):
        self.df.drop_duplicates(inplace=True)
        self.df.drop('car-info', inplace=True, axis=1)

    def fill_null(self):
        self.df['volume(cm3)'].fillna(value=self.df['volume(cm3)'].mean(), inplace=True)
        self.df['drive_unit'].fillna(value='front-wheel drive', inplace=True)
        self.df['segment'].fillna(value='D', inplace=True)
        self.df['model'].fillna(value='passat', inplace=True)

    def apply_to_all(self):
        self.df["model"] = self.df.apply(lambda x: self.apply_each(x["car-info"], 0), axis=1)
        self.df["manufacturer"] = self.df.apply(lambda x: self.apply_each(x["car-info"], 1), axis=1)
        self.df["year"] = self.df.apply(lambda x: self.apply_each(x["car-info"], 2), axis=1)
        self.df["fuel_type"] = self.df["fuel_type"].apply(lambda x: x.lower())

    def apply_each(self, info, indx):
        info = info.replace('[', '')
        info = info.replace(']', '')
        info = info.replace('(', '')
        info = info.replace(')', '')
        lst = info.split(',')
        lst[0] = self.converter(lst[0])
        if indx == 2:
            return int(lst[indx])
        else:
            return lst[indx]

    def converter(self, x):
        all = ''
        for i in range(0, len(x)):
            if x[i] == '0':
                all += 'zero'
            elif x[i] == '1':
                all += 'one'
            elif x[i] == '2':
                all += 'two'
            elif x[i] == '3':
                all += 'three'
            elif x[i] == '4':
                all += 'four'
            elif x[i] == '5':
                all += 'five'
            elif x[i] == '6':
                all += 'six'
            elif x[i] == '7':
                all += 'seven'
            elif x[i] == '8':
                all += 'eight'
            elif x[i] == '9':
                all += 'nine'
            elif x[i].isalpha():
                all += x[i]
        return all

    def standarizing(self):
        scaler = StandardScaler()
        self.df[["mileage(kilometers)", "volume(cm3)", "year"]] = scaler.fit_transform(
            self.df[["mileage(kilometers)", "volume(cm3)", "year"]])
        if self.flag:
            self.df = pd.DataFrame(self.df,
                                   columns=['car_id', 'condition', 'mileage(kilometers)', 'fuel_type', 'volume(cm3)',
                                            'color', 'transmission', 'drive_unit', 'segment', 'Price Category', 'model',
                                            'manufacturer', 'year'])
        else:
            self.df = pd.DataFrame(self.df,
                                   columns=['car_id', 'condition', 'mileage(kilometers)', 'fuel_type', 'volume(cm3)',
                                            'color', 'transmission', 'drive_unit', 'segment', 'model', 'manufacturer',
                                            'year'])


# ------------------------------------------------------------------------------------------
df_train = pd.read_csv("datasets/cars-train.csv")
df_test = pd.read_csv("datasets/cars-test.csv")
obj_train = Preprocessing(df_train, True)
obj_test = Preprocessing(df_test, False)
df_train = obj_train.run_preprocess()
df_test = obj_test.run_preprocess()

priceCat_to_num = {"Price Category": {"cheap": 0, "moderate": 1, "expensive": 2, "very expensive": 3}}
df_train = df_train.replace(priceCat_to_num)

train_new = pd.get_dummies(df_train,
                           columns=['condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment', 'model',
                                    'manufacturer'])
test_new = pd.get_dummies(df_test,
                          columns=['condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment', 'model',
                                   'manufacturer'])

train_columns = train_new.columns.tolist()
test_columns = test_new.columns.tolist()

for i in range(0, len(test_columns)):
    if not (test_columns[i] in train_columns):
        train_new[test_columns[i]] = 0

for i in range(0, len(train_columns)):
    if (not (train_columns[i] in test_columns)) and train_columns[i] != "Price Category":
        test_new[train_columns[i]] = 0

train_new = train_new.reindex(sorted(train_new.columns), axis=1)
test_new = test_new.reindex(sorted(test_new.columns), axis=1)

train_new.to_csv("datasets/train_preprocessed.csv", index=False)
test_new.to_csv("datasets/test_preprocessed.csv", index=False)
