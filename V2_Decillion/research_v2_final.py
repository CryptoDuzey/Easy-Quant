# Easy-Quant V2: 十倍计划 (纯多头轮动版)
# 说明: 放弃宏观择时，专注截面 Alpha 挖掘。对头部个股采取平方加权。

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def run_v2_final():
    # 数据加载逻辑同探索版 (实际应用中封装为独立模块)
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    stocks = get_index_stocks('000300.SH')
    all_dfs = []

    for yr in years:
        try:
            s, e = f"{yr}-01-01", f"{yr}-12-31" if yr < 2026 else "2026-04-28"
            data = get_price(stocks, start_date=s, end_date=e, fields=['open','high','low','close','volume'])
            if data:
                for ticker, df in data.items():
                    tmp = df.copy()
                    tmp['ticker'] = ticker
                    all_dfs.append(tmp)
        except Exception:
            continue

    df_real = pd.concat(all_dfs).reset_index()
    df_real.columns = [c.lower() for c in df_real.columns]
    df_real = df_real.rename(columns={'index':'date'}).set_index(['date', 'ticker']).sort_index()
    df_real = df_real[~df_real.index.duplicated(keep='first')]

    # 多因子构建 (引入波动率与成交量异动因子)
    def get_attack_alphas(df):
        f = pd.DataFrame(index=df.index)
        f['a101'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)
        f['mom'] = df.groupby(level='ticker')['close'].pct_change(20)
        f['vol'] = df.groupby(level='ticker')['close'].transform(lambda x: x.pct_change().rolling(20).std())
        f['v_pump'] = df.groupby(level='ticker')['volume'].transform(lambda x: x / x.rolling(20).mean())
        return f.groupby(level='date').transform(lambda x: (x - x.mean()) / x.std()).fillna(0)

    X = get_attack_alphas(df_real)
    y = (df_real.groupby(level='ticker')['close'].shift(-5) / df_real['close'] - 1).groupby(level='date').rank(pct=True)

    v = X.notna().all(axis=1) & y.notna()
    model = lgb.train({'objective':'regression', 'learning_rate':0.1, 'max_depth':6, 'verbose':-1}, lgb.Dataset(X[v], label=y[v]), 150)

    signal_df = pd.DataFrame({'predict_score': model.predict(X)}, index=X.index)
    signal_df['target_weight'] = 0.0

    # 截面非线性权重分配
    for date, group in signal_df.groupby(level='date'):
        top10 = group['predict_score'].nlargest(10).index
        scores = group.loc[top10, 'predict_score']
        # 平方放大头部得分差异
        adj_scores = (scores - scores.min() + 0.5) ** 2
        signal_df.loc[top10, 'target_weight'] = adj_scores / adj_scores.sum()

    signal_df.to_csv('alpha_signals_v2_final.csv')
