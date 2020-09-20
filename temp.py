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