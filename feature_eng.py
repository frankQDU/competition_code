import category_encoders as ce

def cal_woe(df_tr, col):
    enc =  ce.WOEEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                    df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'woe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'woe_{col}'].mean(), f'woe_{col}'

def cal_cat(df_tr, col):
    enc =  ce.cat_boost.CatBoostEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'cat_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'cat_{col}'].mean(), f'cat_{col}'

def cal_js(df_tr, col):
    enc =  ce.james_stein.JamesSteinEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'js_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'js_{col}'].mean(), f'js_{col}'

def cal_loe(df_tr, col):
    enc =  ce.LeaveOneOutEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'loe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'loe_{col}'].mean(), f'loe_{col}'

def cal_moe(df_tr, col):
    enc =  ce.MEstimateEncoder(cols=[col]).fit(df_tr.loc[::,feature_col],
                                                        df_tr.loc[::,'isDefault'])
    tmp = pd.DataFrame({f'{col}':df_tr.loc[::,col],
                        f'moe_{col}':enc.transform(df_tr.loc[::,feature_col],
                                                   df_tr.loc[::,'isDefault'])[col]})
    return tmp.groupby([col])[f'moe_{col}'].mean(), f'moe_{col}'



def kfold_encoding_feature(df_tr, df_te, feats, n_splits):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    df_tr['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr, df_tr['isDefault'])):
        df_tr.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr, df_tr['isDefault'])):
            tmp_trn = df_tr.iloc[trn_idx]

            dict_woe, colname_woe = cal_woe(tmp_trn, feat)
            dict_cat, colname_cat = cal_cat(tmp_trn, feat)
            dict_js , colname_js  = cal_js(tmp_trn, feat)
            dict_loe, colname_loe = cal_loe(tmp_trn, feat)
            dict_moe, colname_moe = cal_moe(tmp_trn, feat)

            if fold_==0:
                df_tr[colname_woe] = None
                df_tr[colname_cat] = None
                df_tr[colname_js ] = None
                df_tr[colname_loe] = None
                df_tr[colname_moe] = None

                kfold_features.append(colname_woe)
                kfold_features.append(colname_cat)
                kfold_features.append(colname_js)
                kfold_features.append(colname_loe)
                kfold_features.append(colname_moe)


            tmp = df_tr.loc[df_tr.fold == fold_, [feat]]
            df_tr.loc[df_tr.fold == fold_, colname_woe] = tmp[feat].map(dict_woe)
            df_tr.loc[df_tr.fold == fold_, colname_cat] = tmp[feat].map(dict_cat)
            df_tr.loc[df_tr.fold == fold_, colname_js ] = tmp[feat].map(dict_js )
            df_tr.loc[df_tr.fold == fold_, colname_loe] = tmp[feat].map(dict_loe)
            df_tr.loc[df_tr.fold == fold_, colname_moe] = tmp[feat].map(dict_moe)

        df_tr[colname_woe] = df_tr[colname_woe].astype(float)
        df_tr[colname_cat] = df_tr[colname_cat].astype(float)
        df_tr[colname_js ] = df_tr[colname_js ].astype(float)
        df_tr[colname_loe] = df_tr[colname_loe].astype(float)
        df_tr[colname_moe] = df_tr[colname_moe].astype(float)


        df_te[colname_woe] = None
        df_te[colname_cat] = None
        df_te[colname_js ] = None
        df_te[colname_loe] = None
        df_te[colname_moe] = None

        order_label = df_tr.groupby([feat])[colname_woe].mean()
        df_te[colname_woe] = df_te[feat].map(order_label)

        order_label = df_tr.groupby([feat])[colname_cat].mean()
        df_te[colname_cat] = df_te[feat].map(order_label)

        order_label = df_tr.groupby([feat])[colname_js ].mean()
        df_te[colname_js ] = df_te[feat].map(order_label)

        order_label = df_tr.groupby([feat])[colname_loe].mean()
        df_te[colname_loe] = df_te[feat].map(order_label)

        order_label = df_tr.groupby([feat])[colname_moe].mean()
        df_te[colname_moe] = df_te[feat].map(order_label)

    del df_tr['fold']
    return df_tr, df_te


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