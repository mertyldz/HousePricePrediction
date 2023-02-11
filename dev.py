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



#pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 500)

##### TODO EDA #####

#! Verisetini Okuma ve Birleştirme
def loadDataset():
    test_data = pd.read_csv(r"dataset\test.csv")
    train_data = pd.read_csv(r"dataset\train.csv")
    df_ = pd.concat([test_data, train_data])
    df = df_.copy()
    return df
df = loadDataset()

#! Değişkenleri Yakalama ve Tip düzenlemesi

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

cat_cols, num_cols, cat_but_car_cols=grab_col_names(df)
num_cols = num_cols + ["BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","Fireplaces","GarageCars","BsmtFullBath"]
cat_cols = [col for col in cat_cols if col not in num_cols]
df[cat_cols] = df[cat_cols].astype(object)
df[cat_cols].info()
df[num_cols].info()

#! Numerik ve kategorik değişkenelerin veri içindeki dağılımlarını gözlemlemek
def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)       
        

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)       

for i in cat_cols:
    cat_summary(df, i)
    

for i in num_cols:
    num_summary(df, i)
    
#! Kategorik değişken ile hedef değişken incelemesi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    
for i in cat_cols:
    target_summary_with_cat(df, "SalePrice", i)
    
#! Aykırı gözlem var mı?
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
    
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    

for i in num_cols:
    print(f"{i} sütununda aykırı değer var mı?", check_outlier(df, i))
    
for i in df.columns:
    print(f"{i} sütununda eksik gözlem var mı?", df[i].isnull().any())
    
##### TODO Feature Engineering #####

#? Aykırı Gözlemler
outlier_columns = []
for i in num_cols:
    if check_outlier(df, i):
        outlier_columns.append(i)

for outliers in outlier_columns:
    replace_with_thresholds(df, outliers)
    
#? Eksik Gözlemler
missing_columns = []
for i in df.columns:
    if df[i].isnull().any():
        missing_columns.append(i)

missing_columns = [col for col in missing_columns if col not in "SalePrice"]
missing_nums = [col for col in missing_columns if df[col].dtype in ["int64","float64"]]
missing_cats = [col for col in missing_columns if col not in missing_nums]
df[missing_cats].info()
df[missing_nums].info()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    return missing_df
    print(missing_df, end="\n")

    if na_name:
        return na_columns
    
#* %80'den fazla missing value varsa dropla
import numpy as np
missingfilterdf=missing_values_table(df, na_name=True)
variables_to_drop = list(missingfilterdf[missingfilterdf["ratio"] > 40]["n_miss"].index) # Sale price dışındaki %40 üstünü düşür.
variables_to_drop = [var for var in variables_to_drop if var not in "SalePrice"]
df.drop(variables_to_drop, inplace=True, axis=1)
df.columns

missing_cats = [col for col in missing_cats if col not in variables_to_drop]

#? Sayısal değişkenleri ortalamayla, kategorikleri mode ile doldur.
for missing in missing_nums:
    df[missing].fillna(df[missing].mean(), inplace=True)
    
df[missing_cats] = df[missing_cats].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

    
#! Rare Encoder

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

cat_cols = [col for col in cat_cols if col not in variables_to_drop]

rare_analyser(df, "SalePrice", cat_cols=cat_cols)


#df2 = rare_encoder(df, 0.05)
#rare_analyser(df2, "SalePrice", cat_cols=cat_cols)

#? Rare Encoding işlemini gerçekleştirince, ordinal dönüşüm yapamıyorum? Yapmadım.


#! Yeni Değişkenler
df.columns
df.head()
df["UnfBsmtRatio"] = df["BsmtUnfSF"] / df["TotalBsmtSF"]

#! Encoding
#* Ordinaller
def ordinal_econding():
    ordinenc = OrdinalEncoder(categories=[["IR3", "IR2", "IR1", "Reg"]])
    df["LotShape"] = ordinenc.fit_transform(df[["LotShape"]])

    ord2 = OrdinalEncoder(categories=[["Sev", "Mod", "Gtl"]])
    df["LandSlope"] = ord2.fit_transform(df[["LandSlope"]])

    ord3 = OrdinalEncoder(categories=[["Po", "Fa", "TA", "Gd", "Ex"]])
    df["ExterQual"] = ord3.fit_transform(df[["ExterQual"]])
    df["ExterCond"] = ord3.fit_transform(df[["ExterCond"]])
    df["HeatingQC"] = ord3.fit_transform(df[["HeatingQC"]])
    df["KitchenQual"] = ord3.fit_transform(df[["KitchenQual"]])

    ord4 = OrdinalEncoder(categories=[["NA","Po", "Fa", "TA", "Gd", "Ex"]])
    df["BsmtQual"] = ord4.fit_transform(df[["BsmtQual"]])
    df["BsmtCond"] = ord4.fit_transform(df[["BsmtCond"]])
    df["GarageQual"] = ord4.fit_transform(df[["GarageQual"]])
    df["GarageCond"] = ord4.fit_transform(df[["GarageCond"]])

    ord5 = OrdinalEncoder(categories=[["NA", "No", "Mn", "Av", "Gd"]])
    df["BsmtExposure"] = ord5.fit_transform(df[["BsmtExposure"]])

    ord6 = OrdinalEncoder(categories=[["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]])
    df["BsmtFinType1"] = ord6.fit_transform(df[["BsmtFinType1"]])
    df["BsmtFinType2"] = ord6.fit_transform(df[["BsmtFinType2"]])

    ord7 = OrdinalEncoder(categories=[["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]])
    df["Functional"] = ord7.fit_transform(df[["Functional"]])

    ord8 = OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]])
    df["GarageFinish"] = ord8.fit_transform(df[["GarageFinish"]])

    ord8 = OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]])
    df["MiscVal"]
    

ordinal_econding()

binary_cols = [col for col in df.columns if (df[col].dtype == object) & (df[col].nunique()==2)]
df[binary_cols]

le = LabelEncoder()
for binary_cols in binary_cols:
    df[binary_cols] = le.fit_transform(df[[binary_cols]])
    

onehot_cols = [col for col in cat_cols if df[col].dtype not in ["float64", "int32"]]
df[onehot_cols]

df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

[col for col in X_train.columns if df[col].dtype == "object"]
df["Neighborhood"].value_counts()
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

####TODO Model Kurma####
#! ID DROPLAMAYI UNUTTUN!!!!!
test = df[df["SalePrice"].isnull()]
train = df[~df["SalePrice"].isnull()]

X_train = train.drop("SalePrice", axis=1)
y_train = train[["SalePrice"]]

#* Train seti üzerindeki başarısı
#! Train-Test Split Kullanabilirsin!
lightgbm = LGBMRegressor(random_state=1601).fit(X_train, y_train)
y_trainpred = lightgbm.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_trainpred))

#* Log dönüşümü

np.expm1

#* Hiperparametre optimizasyonu
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],"n_estimators": [200, 300, 350, 400],"colsample_bytree": [0.9, 0.8, 1]}
grid_lightgbm = GridSearchCV(lightgbm, lgbm_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
grid_lightgbm.best_params_ # colsample_bytree': 0.8, learning_rate': 0.02, n_estimators': 400

lightgbm2 = LGBMRegressor(colsample_bytree= 0.8, learning_rate= 0.02, random_state=1601)
lgbm_params2 = {"n_estimators": [400, 500, 1000, 2500, 5000, 7500, 10000, 15000]}
grid_lightgbm = GridSearchCV(lightgbm2, lgbm_params2, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
grid_lightgbm.best_params_ # n_estimators = 15000

lgbm_final = lightgbm2 = LGBMRegressor(colsample_bytree= 0.8, learning_rate= 0.02, n_estimators = 15000, random_state=1601).fit(X_train, y_train)

ytrain_pred = lgbm_final.predict(X_train)
np.sqrt(mean_squared_error(y_train, ytrain_pred)) # 4.47 ? Overfit mi??

grid_lightgbm = GridSearchCV(lightgbm2, lgbm_params2, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
grid_lightgbm.best_params_ #yukarıda overfit olmuş, 10 katlı için n_estimators 400
lgbm_final  = LGBMRegressor(colsample_bytree= 0.8, learning_rate= 0.02, n_estimators = 400, random_state=1601).fit(X_train, y_train)
ytrain_pred = lgbm_final.predict(X_train)
np.sqrt(mean_squared_error(y_train, ytrain_pred)) # 12068

mean_absolute_error(y_train, ytrain_pred) / y_train.mean() # 0.03 

#* Değişken önem düzeyi
lgbm_final.feature_importances_

def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X_train, num=20)

#* Kaggle Tahminleri

test.info()
test.drop("SalePrice", inplace=True, axis=1)
y_pred = lgbm_final.predict(test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ["PredictedPrice"]
PredictedDF = pd.concat([test, y_pred], axis=1)