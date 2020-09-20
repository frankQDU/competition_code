import category_encoders as ce

def cal_woe(df_tr, col):
    enc =  ce.WOEEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                    df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'woe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'woe_{col}'].mean()

def cal_cat(df_tr, col):
    enc =  ce.cat_boost.CatBoostEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'cat_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'cat_{col}'].mean()

def cal_js(df_tr, col):
    enc =  ce.james_stein.JamesSteinEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'js_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'js_{col}'].mean()

def cal_loe(df_tr, col):
    enc =  ce.LeaveOneOutEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'loe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'loe_{col}'].mean()

def cal_moe(df_tr, col):
    enc =  ce.MEstimateEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'moe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'moe_{col}'].mean()


def kfold_stats_feature(df_tr, df_te, feats, n_splits):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    df_tr['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr, df_tr['isDefault'])):
        df_tr.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['isDefault']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            df_tr[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr, df_tr['isDefault'])):
                tmp_trn = df_tr.iloc[trn_idx]

                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = df_tr.loc[df_tr.fold == fold_, [feat]]
                df_tr.loc[df_tr.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = df_tr[f].mean()
                df_tr.loc[df_tr.fold == fold_, colname] = df_tr.loc[df_tr.fold == fold_, colname].fillna(global_mean)
            df_tr[colname] = df_tr[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            df_te[colname] = None
            order_label = df_tr.groupby([feat])[f].mean()
            df_te[colname] = df_te[feat].map(order_label)
            # fillna
            global_mean = df_tr[f].mean()
            df_te[colname] = df_te[colname].fillna(global_mean)
            df_te[colname] = df_te[colname].astype(float)
    del df_tr['fold']
    return df_tr, df_te