# Imports
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from lightgbm import LGBMRegressor
import joblib

def loadDataset():
    test_data = pd.read_csv(r"dataset\test.csv")
    train_data = pd.read_csv(r"dataset\train.csv")
    df_ = pd.concat([test_data, train_data])
    df = df_.copy()
    return df

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def get_cols(dataframe):
    cat_cols, num_cols, cat_but_car_cols = grab_col_names(dataframe)
    num_cols = num_cols + ["BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "Fireplaces",
                           "GarageCars", "BsmtFullBath"]
    cat_cols = [col for col in cat_cols if col not in num_cols]
    return cat_cols, num_cols

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def find_outlier_cols(dataframe):
    outlier_columns = []
    for i in num_cols:
        if check_outlier(dataframe, i):
            outlier_columns.append(i)
    return outlier_columns

def fix_outliers(dataframe, outlier_cols):
    for outliers in outlier_cols:
        replace_with_thresholds(df, outliers)

def find_cols_with_missings(dataframe):
    missing_columns = []
    for i in df.columns:
        if df[i].isnull().any():
            missing_columns.append(i)
    return missing_columns

def small_changes_with_missings(missing_columns):
    missing_columns = [col for col in missing_columns if col not in "SalePrice"]
    missing_nums = [col for col in missing_columns if df[col].dtype in ["int64", "float64"]]
    missing_cats = [col for col in missing_columns if col not in missing_nums]
    return missing_cats, missing_nums

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    return missing_df
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def drop_with_missing_ratio(dataframe, ratio=40):
    missingfilterdf = missing_values_table(dataframe, na_name=True)
    variables_to_drop = list(missingfilterdf[missingfilterdf["ratio"] > ratio]["n_miss"].index) # Sale price dışındaki %40 üstünü düşür.
    variables_to_drop = [var for var in variables_to_drop if var not in "SalePrice"]
    dataframe.drop(variables_to_drop, inplace=True, axis=1)
    return dataframe, variables_to_drop

def fill_num_missings(dataframe, missing_nums):
    for missing in missing_nums:
        dataframe[missing].fillna(df[missing].mean(), inplace=True)

def fill_cat_missings(dataframe, missing_cats):
    dataframe[missing_cats] = dataframe[missing_cats].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

def ordinal_econding(dataframe):
    ordinenc = OrdinalEncoder(categories=[["IR3", "IR2", "IR1", "Reg"]])
    dataframe["LotShape"] = ordinenc.fit_transform(dataframe[["LotShape"]])

    ord2 = OrdinalEncoder(categories=[["Sev", "Mod", "Gtl"]])
    dataframe["LandSlope"] = ord2.fit_transform(dataframe[["LandSlope"]])

    ord3 = OrdinalEncoder(categories=[["Po", "Fa", "TA", "Gd", "Ex"]])
    dataframe["ExterQual"] = ord3.fit_transform(dataframe[["ExterQual"]])
    dataframe["ExterCond"] = ord3.fit_transform(dataframe[["ExterCond"]])
    dataframe["HeatingQC"] = ord3.fit_transform(dataframe[["HeatingQC"]])
    dataframe["KitchenQual"] = ord3.fit_transform(dataframe[["KitchenQual"]])

    ord4 = OrdinalEncoder(categories=[["NA","Po", "Fa", "TA", "Gd", "Ex"]])
    dataframe["BsmtQual"] = ord4.fit_transform(dataframe[["BsmtQual"]])
    dataframe["BsmtCond"] = ord4.fit_transform(dataframe[["BsmtCond"]])
    dataframe["GarageQual"] = ord4.fit_transform(dataframe[["GarageQual"]])
    dataframe["GarageCond"] = ord4.fit_transform(dataframe[["GarageCond"]])

    ord5 = OrdinalEncoder(categories=[["NA", "No", "Mn", "Av", "Gd"]])
    dataframe["BsmtExposure"] = ord5.fit_transform(dataframe[["BsmtExposure"]])

    ord6 = OrdinalEncoder(categories=[["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]])
    dataframe["BsmtFinType1"] = ord6.fit_transform(dataframe[["BsmtFinType1"]])
    dataframe["BsmtFinType2"] = ord6.fit_transform(dataframe[["BsmtFinType2"]])

    ord7 = OrdinalEncoder(categories=[["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]])
    dataframe["Functional"] = ord7.fit_transform(dataframe[["Functional"]])

    ord8 = OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]])
    dataframe["GarageFinish"] = ord8.fit_transform(dataframe[["GarageFinish"]])

    ord8 = OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]])

def binary_encoding(dataframe):
    binary_cols = [col for col in df.columns if (dataframe[col].dtype == object) & (dataframe[col].nunique() == 2)]
    le = LabelEncoder()
    for binary_col in binary_cols:
        dataframe[binary_col] = le.fit_transform(df[binary_col])

def onehot_encoding(dataframe, categ_cols, variables_to_drop):
    categ_cols = [col for col in categ_cols if col not in variables_to_drop]
    onehot_cols = [col for col in categ_cols if dataframe[col].dtype not in ["float64", "int32"]]
    onehot_cols = onehot_cols + ["Neighborhood"]
    dataframe = pd.get_dummies(dataframe, columns=onehot_cols, drop_first=True)
    return dataframe

def split_dataset(dataframe):
    test = dataframe[dataframe["SalePrice"].isnull()]
    test=test.drop("SalePrice", axis=1)
    train = dataframe[~dataframe["SalePrice"].isnull()]
    X_train = train.drop("SalePrice", axis=1)
    y_train = train[["SalePrice"]]
    return X_train, y_train, test

def create_lightgbm_model(xtrain, ytrain):
    # Train success
    lightgbm = LGBMRegressor(random_state=1601).fit(xtrain, ytrain)
    y_trainpred = lightgbm.predict(xtrain)
    print("Success on train set:", np.sqrt(mean_squared_error(ytrain, y_trainpred)))
    # Hyperparameter tuning
    lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1], "n_estimators": [250 ,400, 500, 1000],
                   "colsample_bytree": [0.9, 0.8, 1]}
    grid_lightgbm = GridSearchCV(lightgbm, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(xtrain, ytrain)
    # Final Model
    lightgbm_final = LGBMRegressor(**grid_lightgbm.best_params_).fit(xtrain, ytrain)
    lightgbm_final_for_test = LGBMRegressor(**grid_lightgbm.best_params_)
    # Tuned test success
    ytrain_pred = lightgbm_final.predict(xtrain)
    print("Success on train set when it's tuned:", np.sqrt(mean_squared_error(y_train, ytrain_pred)))
    # Train-test split
    x_train, x_test, Y_train, Y_test=train_test_split(xtrain, ytrain, test_size=0.15, random_state=1601)
    y_pred = lightgbm_final_for_test.fit(x_train, Y_train).predict(x_test)
    print("Success on test set with train tuning:", np.sqrt(mean_squared_error(Y_test, y_pred)))
    model_to_tune = LGBMRegressor()
    final_grid = GridSearchCV(model_to_tune, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(x_train, Y_train)
    finaltunedmodel = LGBMRegressor(**final_grid.best_params_).fit(x_train, Y_train)
    y_pred_final = finaltunedmodel.predict(x_test)
    print("Success on test set when it's tuned:", np.sqrt(mean_squared_error(Y_test, y_pred_final)))
    return finaltunedmodel


if __name__=="__main__":
    df = loadDataset()
    # Get Cols Types
    cat_cols, num_cols = get_cols(df)
    # Fix Outliers
    outlier_columns = find_outlier_cols(df)
    fix_outliers(df, outlier_columns)
    # Drop High Ratio Missing Values
    df, variables_to_drop = drop_with_missing_ratio(df)
    # Fix Low Ratio Missing Values
    miss_cols = find_cols_with_missings(df)
    miss_cats, miss_nums = small_changes_with_missings(miss_cols)
    # Fill Missings
    fill_num_missings(df, miss_nums)
    fill_cat_missings(df, miss_cats)
    # Feature Engineering
    df["UnfBsmtRatio"] = df["BsmtUnfSF"] / df["TotalBsmtSF"]
    # Encoding
    ordinal_econding(df)
    binary_encoding(df)
    df = onehot_encoding(df, cat_cols, variables_to_drop)
    df = df.drop("Id", axis=1)
    # Model
    X_train, y_train, test = split_dataset(df)
    final_model =create_lightgbm_model(X_train, y_train)
    joblib.dump(final_model, "hous_price_model.pkl")





