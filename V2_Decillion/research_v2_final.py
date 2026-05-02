# Easy-Quant V2: 十倍计划 (纯多头轮动版) — Walk-Forward 修复版
#
# 核心修复: 滚动训练消除未来函数
#   - 原版: 全量数据(2021-2026)一次性训练+预测 → 严重未来数据泄露
#   - 修复: 每个预测月份只用该月之前的数据训练 LightGBM
#   - 因子截面标准化本身是日内的, 无未来函数问题
#
# 修复日期: 2026.05.01 — 通哥的寄居蟹 🦞

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def run_v2_final():
    # ============================================================
    # 1. 数据加载
    # ============================================================
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    stocks = get_index_stocks('000300.SH')
    all_dfs = []

    for yr in years:
        try:
            s, e = f"{yr}-01-01", f"{yr}-12-31" if yr < 2026 else "2026-04-28"
            data = get_price(stocks, start_date=s, end_date=e,
                             fields=['open', 'high', 'low', 'close', 'volume'])
            if data:
                for ticker, df in data.items():
                    tmp = df.copy()
                    tmp['ticker'] = ticker
                    all_dfs.append(tmp)
        except Exception:
            continue

    df_real = pd.concat(all_dfs).reset_index()
    df_real.columns = [c.lower() for c in df_real.columns]
    df_real = df_real.rename(columns={'index': 'date'}).set_index(['date', 'ticker']).sort_index()
    df_real = df_real[~df_real.index.duplicated(keep='first')]

    # ============================================================
    # 2. 多因子构建 (因子计算逻辑不变, 截面标准化是日内的故无泄露)
    # ============================================================
    def get_attack_alphas(df):
        f = pd.DataFrame(index=df.index)
        f['a101'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)
        f['mom'] = df.groupby(level='ticker')['close'].pct_change(20)
        f['vol'] = df.groupby(level='ticker')['close'].transform(
            lambda x: x.pct_change().rolling(20).std())
        f['v_pump'] = df.groupby(level='ticker')['volume'].transform(
            lambda x: x / x.rolling(20).mean())
        # 每日截面标准化 — 仅用当日数据, 无未来泄露
        return f.groupby(level='date').transform(lambda x: (x - x.mean()) / x.std()).fillna(0)

    X = get_attack_alphas(df_real)
    y = (df_real.groupby(level='ticker')['close'].shift(-5) / df_real['close'] - 1).groupby(level='date').rank(pct=True)

    # ============================================================
    # 3. Walk-Forward 滚动训练 (核心修复)
    #    - 按月分组, 每月只用过去数据训练
    #    - 重训间隔可调 (当前每3个月重训一次以平衡速度与新鲜度)
    #    - 标签边界安全: 训练数据中 d+5 必须也在训练集内
    # ============================================================
    all_dates = sorted(df_real.index.get_level_values('date').unique())

    def month_key(d):
        return d.strftime('%Y-%m')

    month_labels = sorted(set(month_key(d) for d in all_dates))

    INITIAL_MONTHS = 12   # 至少积累12个月才开始预测
    RETRAIN_EVERY = 3     # 每3个月重训一次
    LABEL_GAP_DAYS = 7    # 标签安全边距 (天), 确保 shift(-5) 的标签不跨入测试期

    signal_parts = []
    current_model = None
    current_train_end_month_idx = None

    for i, test_month in enumerate(month_labels):
        # ---- 3a. 跳过初始积累期 ----
        if i < INITIAL_MONTHS:
            continue

        test_dates = [d for d in all_dates if month_key(d) == test_month]
        if not test_dates:
            continue

        # ---- 3b. 确定训练数据范围 ----
        # 训练数据截止到测试月开始前一天, 所以用 month_labels[i - 1] 的最后一天
        train_end_month_idx = i - 1
        if train_end_month_idx < 0:
            continue

        train_end_date_raw = max(
            d for d in all_dates if month_key(d) == month_labels[train_end_month_idx]
        )

        # ★ 标签安全边界: 训练集末尾要留足天数,
        #   确保 shift(-5) 的标签不依赖测试期的价格
        training_dates_all = [d for d in all_dates if d <= train_end_date_raw]
        if len(training_dates_all) <= LABEL_GAP_DAYS:
            continue
        # 砍掉最后 LABEL_GAP_DAYS 个交易日, 让标签的 d+5 完全落在训练集内
        train_cutoff_date = training_dates_all[-(LABEL_GAP_DAYS + 1)]

        # ---- 3c. 模型训练 (按需重训) ----
        need_retrain = (
            current_model is None
            or (train_end_month_idx != current_train_end_month_idx
                and (train_end_month_idx - INITIAL_MONTHS + 1) % RETRAIN_EVERY == 0)
        )

        if need_retrain:
            train_mask = df_real.index.get_level_values('date') <= train_cutoff_date
            train_idx = X.index.intersection(X.index[train_mask])
            # 只保留 X + y 都有效的样本
            valid = X.loc[train_idx].notna().all(axis=1) & y.loc[train_idx].notna()
            valid_idx = train_idx[valid]

            if len(valid_idx) < 1000:
                print(f"[Walk-Forward] 警告: {test_month} 训练样本不足 ({len(valid_idx)}), 跳过")
                continue

            lgb_train = lgb.Dataset(
                X.loc[valid_idx],
                label=y.loc[valid_idx]
            )
            current_model = lgb.train(
                {'objective': 'regression', 'learning_rate': 0.1, 'max_depth': 6, 'verbose': -1},
                lgb_train,
                num_boost_round=150
            )
            current_train_end_month_idx = train_end_month_idx

        # ---- 3d. 预测当前月 ----
        test_idx = X.index[X.index.get_level_values('date').isin(test_dates)]
        pred_valid = X.loc[test_idx].notna().all(axis=1)
        pred_idx = test_idx[pred_valid]

        if len(pred_idx) == 0:
            continue

        pred = current_model.predict(X.loc[pred_idx])

        month_df = pd.DataFrame({'predict_score': pred}, index=pred_idx)
        month_df['target_weight'] = 0.0

        # 每日截面: 选 top 10, 平方加权放大头部
        for date, group in month_df.groupby(level='date'):
            n_top = min(10, len(group))
            top_idx = group['predict_score'].nlargest(n_top).index
            scores = group.loc[top_idx, 'predict_score']
            # 平移+平方, 让高分股获得不成比例的高权重
            adj = (scores - scores.min() + 0.5) ** 2
            month_df.loc[top_idx, 'target_weight'] = adj / adj.sum()

        signal_parts.append(month_df)

    # ============================================================
    # 4. 合并输出
    # ============================================================
    if signal_parts:
        signal_df = pd.concat(signal_parts)
        signal_df.to_csv('alpha_signals_v2_final.csv')
        print(
            f"[Walk-Forward ✓] 信号生成完毕: {len(signal_df)} 条, "
            f"{signal_df.index.get_level_values('date').nunique()} 个交易日, "
            f"首次预测月: {month_labels[INITIAL_MONTHS]}"
        )
    else:
        print("[Walk-Forward ✗] 未生成任何信号, 请检查数据范围/训练参数")
        signal_df = pd.DataFrame()

    return signal_df
