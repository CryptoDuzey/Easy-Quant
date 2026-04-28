# Easy-Quant V2: 探索版 (LightGBM + HMM 宏观择时)
# 说明: 此版本引入了 HMM 隐马尔可夫模型进行大盘状态识别。
# 历史回测结论: HMM 能有效规避 2022 年大跌，但在 2023-2025 震荡市中容易产生状态漂移导致踏空。

import pandas as pd
import numpy as np
import lightgbm as lgb
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

def run_hmm_exploration():
    # 1. 获取基础数据 (分年拉取，避免长周期数据缺失导致 NoneType)
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

    # 2. 特征工程 (Alpha101极值, 动量)
    def get_features(df):
        f = pd.DataFrame(index=df.index)
        f['a101'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)
        f['mom_20'] = df.groupby(level='ticker')['close'].pct_change(20)
        return f.groupby(level='date').transform(lambda x: (x - x.mean()) / x.std()).fillna(0)

    X = get_features(df_real)
    y = (df_real.groupby(level='ticker')['close'].shift(-5) / df_real['close'] - 1).groupby(level='date').rank(pct=True)

    # 3. 拟合大盘 HMM 状态 (使用截面平均收益合成虚拟指数，规避 API 单只指数拉取报错)
    market_ret = df_real['close'].groupby(level='date').pct_change().groupby(level='date').mean()
    market_vol = market_ret.rolling(10).std()
    market_env = pd.concat([market_ret, market_vol], axis=1).dropna()
    market_env.columns = ['ret', 'vol']

    if HAS_HMM:
        hmm = GaussianHMM(n_components=2, n_iter=1000, random_state=42).fit(market_env.values)
        states = hmm.predict(market_env.values)
        bear_state = 0 if market_env['ret'][states==0].mean() < market_env['ret'][states==1].mean() else 1
        regime = pd.Series(np.where(states==bear_state, 0, 1), index=market_env.index)
    else:
        regime = pd.Series(1, index=market_env.index)

    # 4. 训练与信号生成 (仅在 regime == 1 时开仓)
    v = X.notna().all(axis=1) & y.notna()
    model = lgb.train({'objective':'regression', 'learning_rate':0.05, 'verbose':-1}, lgb.Dataset(X[v], label=y[v]), 100)

    signal_df = pd.DataFrame({'predict_score': model.predict(X)}, index=X.index)
    signal_df['target_weight'] = 0.0

    for date, group in signal_df.groupby(level='date'):
        dt = pd.Timestamp(date)
        if dt in regime.index and regime.loc[dt] == 1:
            top10 = group['predict_score'].nlargest(10).index
            scores = group.loc[top10, 'predict_score']
            w = (scores - scores.min() + 0.1)
            signal_df.loc[top10, 'target_weight'] = w / w.sum()

    signal_df.to_csv('alpha_signals_v2_hmm.csv')
