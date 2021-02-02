import gc
import re
import string
import operator
from wordcloud import wordcloud as wc
from collections import defaultdict
import snowballstemmer
from snowballstemmer import stemmer
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor

en_stemmer = stemmer(lang='english')
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import nltk
# nltk.download()
from nltk.corpus import stopwords

np.random.seed(42)
stop_words = stopwords.words("english")
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import re
from sklearn.svm import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.neural_network import MLPRegressor
import time
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.linear_model import *
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from lightgbm import LGBMRegressor
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import *
import category_encoders as ce


def load_data():
    data = pd.read_table('data/dev.tsv')
    eval = pd.read_table('data/eval.tsv')
    df_train = data.copy()
    df_test = eval.copy()
    # print(df_train.columns,df_test.columns)
    return data, eval, df_train, df_test


def dataPrepration(df,df_eval):
    df = df.sort_values(by=['country', 'winery', 'quality',
                            'region_1',
                            'designation',
                            'variety'
                            ])
    df = df[df['country'] != 'US-France'].copy()

    df = df.drop_duplicates(keep='last')
    df = df.drop_duplicates(subset=['description'], keep='last')
    df[df['winery'] == 'Tsililis'].country = 'Greece'
    df[df['winery'] == 'Tsililis'].province = 'Greece'

    df.loc[(df['region_1'] == 'Walla Walla Valley (WA)') &
           (df['variety'] == 'Pinot Noir'), ['country']] = 'Uruguay'
    df.loc[(df['region_1'] == 'Walla Walla Valley (WA)') &
           (df['variety'] == 'Pinot Noir'), ['province']] = 'Uruguay'

    df.loc[(df['country'] == 'Argentina') &
           (df['winery'] == 'Achaval-Ferrer')
           & (df['designation'].isnull())
    , ['designation']] = 'Finca Bella Vista'

    df = df[df['country'] != 'US-France'].copy()

    n_eval = df_eval.copy()
    for i in range(0, df_eval.shape[0]):
        rec = df_eval.loc[i, :]
        if str(rec[2]) == 'nan':
            d = df[(df['country'] == rec[0]) &
                   (df['description'] == rec[1]) |
                   (df['province'] == rec[3]) &
                   (df['region_1'] == rec[4]) &
                   (df['winery'] == rec[7]) &
                   (df['variety'] == rec[6])
                   ]
            if d.shape[0] != 0:
                # print(i,d['designation'].values[0])
                n_eval.loc[i, :].designation = d['designation'].values[0]

    df = df.fillna(value='OTHER').copy()
    n_eval = n_eval.fillna(value='OTHER').copy()

    df = remove_outlier(df, 'quality')

    return df,n_eval

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def tfidf(train, test=None):
    tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), analyzer='word', lowercase=True)
    # tfidf_vectorizer = TfidfVectorizer()
    train_feature = tfidf_vectorizer.fit_transform(train)
    if test is not None:
        test_feature = tfidf_vectorizer.transform(test)
    return train_feature, test_feature, tfidf_vectorizer


def count_vec(train, test=None):
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)
    train_feature = vectorizer.fit_transform(train)
    if test is not None:
        test_feature = vectorizer.transform(test)
    return train_feature, test_feature, vectorizer


def scaler(scaler, data, test=None):
    scaler.fit(data)  # Apply transform to both the training set and the test set.
    train_scale = scaler.transform(data)
    if test is not None:
        test_scale = scaler.fit_transform(test)

    return train_scale, test_scale, scaler


def MLPmodel(self):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(int(256 * self.alpha), activation='relu'),
        keras.layers.Dense(int(256 * self.alpha), activation='relu'),
        keras.layers.Dense(int(256 * self.alpha), activation='relu'),
        keras.layers.Dense(self.n_output)
    ])

    return model


def dim_reduction(train, test=None):
    tsvd = TruncatedSVD(n_components=90, random_state=0)
    tsvd.fit(train)
    print('strat transformation SVD')
    X_train_svd = tsvd.transform(train)
    if test is not None:
        X_test_svd = tsvd.transform(test)
    percentage = np.sum(tsvd.explained_variance_ratio_) * 100
    print(f'{percentage:.2f}%')

    return X_train_svd, X_test_svd


def train_model(classifier, X_tr, y_tr, X_te, y_te):
    print('start training...')
    classifier.fit(X_tr, y_tr)
    print('evaluation...')
    y_p = classifier.predict(X_te)
    score = r2_score(y_te, y_p)
    print(f'score is {score}')
    return classifier, score


def WriteOnFile(name, y_eval):
    f = open(name, "w")
    f.write("Id,Predicted\n")
    for index, i in enumerate(y_eval):
        f.write(f"{index},{i}\n")
    f.close


if __name__ == '__main__':
    data, eval, df_train, df_test = load_data()

    df, df_eval = dataPrepration(df_train,df_test)



    col = ['country', 'description', 'designation', 'province', 'region_1',
           'region_2', 'variety', 'winery']

    X_train, X_test, y_train, y_test = train_test_split(df[col], df['quality'],
                                                        test_size=0.1,
                                                        random_state=0)
    cols = ['country', 'designation', 'province', 'region_1',
            'region_2', 'variety', 'winery']
    encoder = ce.TargetEncoder(cols=cols, smoothing=10)
    encoder.fit(X_train, y_train)
    X_train_encode = encoder.transform(X_train, y_train)
    X_test_encode = encoder.transform(X_test, y_test)

    xbg_params = {'colsample_bytree': 0.7,
                  'learning_rate': 0.1,
                  'max_depth': 10,
                  'min_child_weight': 1,
                  'n_estimators': 200,
                  'random_state': 42,
                  'objective': 'reg:squarederror',
                  'subsample': 0.7}
    xbg = XGBRegressor(**xbg_params)

    classifier, r2score = train_model(xbg, X_train_encode, y_train, X_test_encode,
                                      y_test)
    print(f"model: {classifier} r2 Score: {r2score}")

