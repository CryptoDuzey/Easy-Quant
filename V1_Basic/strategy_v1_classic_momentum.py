import pandas as pd
import numpy as np

def init(context):
    set_benchmark('000300.SH')
    set_commission(PerShare(type='stock', cost=0.0002))
    set_slippage(PriceSlippage(0.005))
    context.index_code = '000300.SH'

    # 策略参数配置
    context.p_ma_macro = 40         # 大盘防守均线周期
    context.p_ma_micro = 60         # 个股趋势均线周期
    context.p_mom_days = 20         # 动量计算周期
    context.p_pool_size = 15        # 动量备选池容量
    context.p_hold_num = 5          # 目标持仓数量

    context.p_rsi_period = 14       # RSI计算周期
    context.p_rsi_threshold = 75    # RSI超买过滤阈值
    context.p_stop_loss = 0.90      # 移动止损阈值 (0.90 = 10%回撤)
    context.p_vol_days = 22         # 波动率计算周期

    # 状态变量
    context.pro_pool = []
    context.last_high = {}
    context.rebalance_days = 5
    context.days_count = 0

def handle_bar(context, bar_dict):
    # 1. 移动止损
    hold_stocks = list(context.portfolio.stock_account.positions)
    for s in hold_stocks:
        if s not in bar_dict:
            continue
        curr_price = bar_dict[s].close
        context.last_high[s] = max(context.last_high.get(s, curr_price), curr_price)

        if curr_price < context.last_high[s] * context.p_stop_loss:
            order_target(s, 0)
            context.last_high.pop(s)

    # 调仓周期控制
    context.days_count += 1
    if context.days_count % context.rebalance_days != 1:
        return

    # 获取历史数据
    max_hist = max(context.p_ma_macro, context.p_ma_micro, context.p_mom_days, context.p_vol_days) + 5
    all_stocks = get_index_stocks(context.index_code)
    all_stocks = [s for s in all_stocks if not bar_dict[s].is_paused and not bar_dict[s].is_st]
    df_close = history(all_stocks + [context.index_code], ['close'], max_hist, '1d', False, 'pre', is_panel=1)['close']

    # 2. 宏观风控：大盘跌破均线清仓
    idx_close = df_close[context.index_code]
    idx_ma = idx_close.iloc[-(context.p_ma_macro+1):-1].mean()
    if idx_close.iloc[-2] < idx_ma:
        if hold_stocks:
            for s in hold_stocks:
                order_target(s, 0)
            context.last_high.clear()
        return

    # 3. 股票池初筛：均线多头 + 动量排序
    df_stocks = df_close[all_stocks]
    stock_ma = df_stocks.iloc[-(context.p_ma_micro+1):-1].mean()
    y_close = df_stocks.iloc[-2]

    uptrend_stocks = y_close[y_close > stock_ma].index.tolist()
    if not uptrend_stocks:
        return

    df_up = df_stocks[uptrend_stocks]
    mom_rank = (df_up.iloc[-2] / df_up.iloc[-(context.p_mom_days+2)] - 1).sort_values(ascending=False)
    context.pro_pool = mom_rank.head(context.p_pool_size).index.tolist()

    # 清理不在备选池的持仓
    actual_holdings = []
    for s in hold_stocks:
        if s not in context.pro_pool:
            order_target(s, 0)
            if s in context.last_high:
                context.last_high.pop(s)
        else:
            actual_holdings.append(s)

    # 4. 进场信号：RSI超买过滤
    final_targets = []
    for s in context.pro_pool:
        if s in actual_holdings:
            continue

        s_hist = df_stocks[s].dropna()
        if len(s_hist) < context.p_rsi_period + 5:
            continue

        diff = s_hist.diff()
        up = diff.clip(lower=0).rolling(context.p_rsi_period).mean()
        down = -diff.clip(upper=0).rolling(context.p_rsi_period).mean()
        rsi = 100 * up / (up + down)
        y_rsi = rsi.iloc[-2]

        if pd.notna(y_rsi) and y_rsi < context.p_rsi_threshold:
            final_targets.append(s)

    buy_slots = context.p_hold_num - len(actual_holdings)
    if buy_slots <= 0 or not final_targets:
        return
    final_targets = final_targets[:buy_slots]

    # 5. 资金分配：倒数波动率加权
    df_vol = df_stocks[final_targets].iloc[-(context.p_vol_days+1):-1]
    returns = df_vol.pct_change().dropna()
    vols = returns.std()

    inv_vols = 1.0 / (vols + 1e-6)
    norm_weights = inv_vols / inv_vols.sum()
    total_new_weight = buy_slots * (1.0 / context.p_hold_num)

    for s in final_targets:
        w = total_new_weight * norm_weights[s]
        order_target_percent(s, w)
        context.last_high[s] = bar_dict[s].close
