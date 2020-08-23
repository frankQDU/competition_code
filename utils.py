import time
import numpy as np
import pandas as pd
import xgboost as xgb

def check_consistence(X_train,X_test,feature_col):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    
    xgb_para = {"nthread":-1,
                "learning_rate":0.01,
                'objective': 'binary:logistic',
                'eval_metric':'auc',
                "max_depth":2,
                "subsample":0.6,
                "colsample_bytree":0.6,
                "lambda":10,
                "alpha":0.05}
    
    X_train['label'] = 0
    X_test['label'] = 1
    df = X_train.append(X_test).reset_index(drop = True)
    df = shuffle(df, random_state=2020)
    
    X_train, X_test, y_train, y_test = train_test_split(
             df[feature_col], df['label'], test_size=0.33, random_state=42)
    
    train_set = xgb.DMatrix(X_train,y_train)
    test_set = xgb.DMatrix(X_test,y_test)
    xgb_model = xgb.train(xgb_para,
                          train_set,
                          evals=[(train_set,'train'),
                                 (test_set, 'test')],
#                           early_stopping_rounds=0, 
                          num_boost_round=100,
                          verbose_eval=10)
    return roc_auc_score(y_test, xgb_model.predict( xgb.DMatrix(X_test[feature_col])))



    
def reduce_mem_usage(df, verbose=True,feature_name = []):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    if len(feature_name) == 0:
        feature_name = df.columns()
    else:
        pass
    for col in feature_name:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def timmer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        result=func(*args,**kwargs)
        end_time = time.time()
        m, s = divmod(end_time - start_time, 60)
        h, m = divmod(m, 60)
        print(f'{int(h)}:{int(m)}:{s}')
        return result
    return wrapper


def gen_w2v_features(df, value):
    from tqdm import tqdm
    from gensim.models import Word2Vec
    w2v_dim = 50
    df[value] = df[value].astype(str)
    text_ = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    texts = text_[0].values.tolist()
    w2v = Word2Vec(texts, size=w2v_dim, window=10, iter=45,
               workers=12, seed=1017, min_count=5)

    vacab = w2v.wv.vocab.keys()
    w2v_feature = np.zeros((len(texts), w2v_dim))
    w2v_feature_avg = np.zeros((len(texts), w2v_dim))

    for i, line in tqdm(enumerate(texts)):
        num = 0
        if line == '':
            w2v_feature_avg[i, :] = np.zeros(w2v_dim)
        else:
            for word in line:
                num += 1
                vec = w2v[word] if word in vacab else np.zeros(w2v_dim)
                w2v_feature[i, :] += vec
            w2v_feature_avg[i, :] = w2v_feature[i, :] / num
    w2v_avg = pd.DataFrame(w2v_feature_avg)
    
    w2v_avg.columns = [f'{value}_w2v_avg_{i}' for i in w2v_avg.columns]
    w2v_avg['user'] = text_['user'].tolist()
    return w2v_avg
