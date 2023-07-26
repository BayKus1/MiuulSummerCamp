# PYTHON PROGRAMMING FOR DATA SCIENCE
import matplotlib.pyplot as plt

dir(str)
long_str = "bu bir veri "

"veri" in long_str  # içinde veri var mı kontrol eder.

"efg".strip("e")

name = "john"
type(name)
type(len)

len(name)
name.upper()

ben = [1, 2, 3, "asd"]

ben.pop(1)
ben.append(2)

x = {"name": "Peter", "Age": 21}
x

dictionary = {"elma": "meyve",
              "semiz otu": "sebze"}
# dictionary = {"REG" : ["YSA", 11],
#              "LOG" : ["MSE", 22]}
dictionary.items()
dictionary.keys()
dictionary.values()

t = ("Berkant", "Yiğit", 23, 8)
type(t)

t[0]
t[0:3]

t = list(t)
t[0] = "Berkan"
t = tuple(t)
t[0]


def summer(arg1, arg2):
    """
    Sum of two values

    Parameters
    ----------
    arg1
    arg2

    Returns
    -------

    """
    print(arg1 + arg2)


summer(5, 7)


def alternatif_with_enumerate(string):
    new_string = ""

    for i in range(len(string)):
        letter = string[i]
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


def alternatiff_with_enumerate2(string):
    new_string = ""

    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternatif_with_enumerate("merhaba genç adam")

name_list = ["john", "micheal", "susan"]

age_list = [22, 26, 23]

department_list = ["mathematics", "statistics", "physics"]

list(zip(name_list, age_list, department_list))

# lambda
new_sum = lambda a, b: a + b

new_sum(4, 5)

# map

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(salary):
    return salary * 120 / 100


for i in salaries:
    print(new_salary(i))

# veya

list(map(new_salary, salaries))

# lambda ve map

list(map(lambda x: x * 120 / 100, salaries))

# filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# reduce
from functools import reduce

list_store2 = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store2)

#############################################
# PYTHON ILE VERI ANALIZI
#############################################

# iki diziyi carpalim

import numpy as np

a = [1, 2, 3, 4]
b = [1, 2, 3, 4]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

ab

# daha kısa bir yöntem

a = np.array([1, 2, 3, 4])
b = np.array(b)  # yukaridakiyle ayni anlama geliyor
a * b

# array olusturma

c = [1, 2, 3, 4, 5]
my_array = np.array(c)
type(my_array)
np.zeros(10, dtype=int)
value = np.random.randint(0, 10, size=9)
np.random.normal(10, 4, (3, 4))

value.ndim  # boyut sayisi
value.shape  # boyut bilgisi
value.size  # toplam eleman sayisi
value.dtype  # array veri tipi
value.reshape(3, 3)  # array yeniden bicimlendirme
v = np.arange(0, 30, 3)  # (a,b,c) a dan b ye c ekleyerek git ve yazdir.

catch = [1, 8, 3]
v[catch]  # eleman numaralarini secer ve yazdirir

v < 10
v[v < 10]

# matematiksel islemler

# 5 * x0 + x1 = 12
# x0  + 3 * x1 = 10

denklem1 = np.array([[5, 1], [1, 3]])
sonuc1 = np.array([12, 10])

np.linalg.solve(denklem1, sonuc1)

# PANDAS SERIES

import pandas as pd

s = pd.Series([10, 77, 53, 90, 44, 3])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
s.head()  # varsayilan 5 veriyi getirir

# veri okuma (reading data)

df = pd.read_csv("C:/Users/Berkant_PC/Desktop/miuulYazKampı/caseStudy/dataset/LargestCompaniesInUSAbyReveneue.csv")
df.head()

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.describe().T  # istatiksel bilgi verir. Medyan
df.isnull().values.any()  # eksik degerler hic var mi?
df.isnull().sum()  # eksik deger toplamlarini listeler
df["sex"].head()
df["sex"].head(15).value_counts()

# PANDAS SECIM ISLEMLERI
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0).head()
delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)
df.drop(delete_indexes, axis=0, inplace=True)
df.head(10)

# DEGISKENI INDEXE CEVIRMEK

df["age"].head()
df.age.head()
df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
df["age"] = df.index
df.reset_index().head()

# DEGISKEN UZERINDE ISLEMLER

pd.set_option("display.max_columns", None)
df["age"].head()
type(df["age"].head())  # series
type(df[["age"]].head())  # dataframe
df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]
df["age2"] = df["age"] ** 2
df["age2"]
df["age3"] = df["age"] / df["age2"]
df.drop("age3", axis=1, inplace=True)
df.head()

# loc

col_names = ["age", "embarked", "alive"]
df.loc[:, df.columns.str.contains("age")].head()
df.loc[:, ~df.columns.str.contains("age")].head()
df.loc[0:3]
df.loc[0:3, "age"]
df.loc[0:3, col_names]

# iloc

df.iloc[0:3]
df.iloc[0, 0]
df.iloc[0:3, 0:4]

# kosullu secim

import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "sex", "class"]].head()
df.loc[(df["age"] > 50), ["age", "class"]].head()
df.loc[(df["age"] > 50) & (df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton") & (
        df["sex"] == "male"), ["age", "embark_town", "sex", "class"]].head()

# Toplulastirma ve gruplastirma (Aggregation & Grouping)

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": ["mean", "sum"]})
df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"], "survived": ["mean"]})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": ["mean"]})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": ["mean"], "sex": ["count"]})

# pivot table

df.pivot_table("survived", "sex", "embarked", aggfunc="std")
df.pivot_table("survived", "sex", ["embarked", "class"])

pd.set_option("display.width", 500)

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table("survived", "sex", ["new_age", "class"])
df.head()

# apply and lambda

import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"] / 10
df["age3"] = df["age"] / 10

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# birlestirme (join) islemleri

import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)

# merge

pd.merge(df1, df2)
pd.merge(df1, df2, on="var1")

# MATPLOTLIB

# kategorik degisken = sutun grafigi, countplot bar
# sayisal degisken = hist, boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()
df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.hist(df["age"])
plt.show()

sns.boxplot(df["fare"])
plt.boxplot(df["fare"])
plt.show()

plt.boxplot(x="fare", data=df)
plt.show()

x = np.array([1, 4, 8, 57, 10])
y = np.array([0, 75, 150, 90, 8])

plt.plot(x, y)

plt.plot(x, y, "o")

# marker

plt.plot(y, marker="o")

# line

z = np.array([1, 8, 33, 17, 23, 11])
plt.plot(z)
plt.plot(z, linestyle="dashed")  # kesikli
plt.plot(z, linestyle="dashdot", color="red")  # kesikli ve nokta

# multiple lines

x = np.array([1, 4, 8, 57, 10])
y = np.array([0, 75, 150, 90, 8])
plt.plot(x)
plt.plot(y)

# labels

plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.title("Grafik basligi")
plt.grid()  # izgara

# subplots

import matplotlib.pyplot as plt

# Örnek x ve y verileri
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 30, 25]

# 1 satır ve 2 sütunlu bir alt çizim oluşturun, iki grafik yan yana olacak
fig, axs = plt.subplots(1, 2, sharex=False)

# İlk alt çizimde x ve y verilerini çizin
axs[0].plot(x, y)
axs[0].set_title("Grafik 1")

# İkinci alt çizimde x ve y verilerini çizin
axs[1].plot(x, y)
axs[1].set_title("Grafik 2")

# Alt çizimler arasındaki yatay mesafeyi artırmak için
plt.tight_layout()

# Grafiği göster
plt.show()

# bir diger ornek

import matplotlib.pyplot as plt

# Verileri tanımlayın
x = [85, 90, 95, 100, 105, 110, 115, 120, 125]
y = [250, 260, 270, 280, 290, 300, 310, 320, 330]

# Yeni bir alt çizim (3. grafik) oluşturun
plt.subplot(1, 3, 3)

# Grafik için verileri çizin
plt.plot(x, y)

# Grafik başlığını ayarlayın
plt.title("3. Grafik")

# Grafiği göster
plt.show()

# SEABORN

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

sns.countplot(x=df["sex"], data=df)
plt.show()
# iki yontem de ayni
df["sex"].value_counts().plot(kind="bar", rot=0)
plt.show()

# sayisal degisken gorsellestirme
sns.boxplot(x=df["total_bill"])
plt.show()

df['total_bill'].hist()
plt.show()

# GELISMIS FONKSIYONEL KESIFCI VERI ANALIZI (ADVANCED FUNCTIONAL EDA)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T  # istatistiksel degerleri ortaya cikarir
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("################# Shape #################")
    print(dataframe.shape)
    print("#########################################")
    print("################# Types #################")
    print(dataframe.dtypes)
    print("#########################################")
    print("################# Head #################")
    print(dataframe.head(head))
    print("#########################################")
    print("################# Tail #################")
    print(dataframe.tail(head))
    print("#########################################")
    print("################# NA #################")
    print(dataframe.isnull().sum())
    print("#########################################")
    print("################# Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Kategorik degisken analizi (Analysis of Categorical Variables)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

df.dtypes

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
100 * df['survived'].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("asasfdsdsgsg")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)

df[["age", 'fare']].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# degiskenlerin yakalanmasi ve islemlerin genellestirilmesi
# capturing  variables and generalizing operations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")

df.head()
df.info()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Parameters
    ----------
    dataframe: dataframe
        degisken isimleri alinmak istenen dataframedir.
    cat_th: int64, float64
        numerik fakat kategorik olan degiskenler icin sinif esik degeridir.
    car_th: int64, float64
        kategorik fakat kardinal degiskenler icin sinif esik degeridir.

    Returns
    -------
    cat_cols: list
        Kategorik degisken listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik gorunumlu kardinal degisken listesi

    Notes
    -----
        cat_cols + num_cols + cat_but_car = toplam degisken sayisi
        num_but_cat cat_cols'un icinde
    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


help(grab_col_names)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# hedef degisken analizi (analysis of target variable)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Parameters
    ----------
    dataframe: dataframe
        degisken isimleri alinmak istenen dataframedir.
    cat_th: int64, float64
        numerik fakat kategorik olan degiskenler icin sinif esik degeridir.
    car_th: int64, float64
        kategorik fakat kardinal degiskenler icin sinif esik degeridir.

    Returns
    -------
    cat_cols: list
        Kategorik degisken listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik gorunumlu kardinal degisken listesi

    Notes
    -----
        cat_cols + num_cols + cat_but_car = toplam degisken sayisi
        num_but_cat cat_cols'un icinde
    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_summary(df, "survived")

# hedef degiskenin kategorik degiskenle analizi

df.groupby("sex")["survived"].mean()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_mean": dataframe.groupby(categorical_col)[target].mean()}), "\n")


target_summary_with_cat(df, "survived", "sex")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

# hedef degiskenin sayisal degiskenler ile analizi

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": "mean"})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

# korelasyon analizi (analysis of  correlation)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

df = df.iloc[: 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
corr = df[num_cols].corr()
corr

sns.set(rc = {"figure.figsize" : (12, 12)})
sns.heatmap(corr, cmap = "Accent")
plt.show()

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1) .astype(np.bool))
drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > 0.90)]
cor_matrix[drop_list]

df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot = False, corr_th = 0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k = 1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc = {"figure.figsize" : (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis = 1)
high_correlated_cols(df.drop(drop_list, axis = 1), plot=True)








