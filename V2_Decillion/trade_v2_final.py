# Easy-Quant V2: 回测执行引擎
import pandas as pd
import numpy as np

def init(context):
    set_benchmark('000300.SH')
    set_commission(PerShare(type='stock', cost=0.0002))
    set_slippage(PriceSlippage(0.005))

    context.rebalance_days = 5
    context.days_count = 0
    context.trailing_stop_loss = -0.12 # 宽幅移动止损，防止异常大跌
    context.highest_prices = {}

    try:
        context.signals = pd.read_csv('alpha_signals_v2_final.csv', index_col=['date', 'ticker'], parse_dates=['date'])
    except Exception as e:
        log.error(f"信号文件读取失败: {e}")
        context.signals = pd.DataFrame()

def handle_bar(context, bar_dict):
    if context.signals.empty: return
    curr_ts = pd.Timestamp(get_datetime().strftime('%Y-%m-%d'))
    # Supermind/MindGo 适用持仓字典对象
    pos_dict = context.portfolio.stock_account.positions

    # 1. 移动风控监控
    for s in list(pos_dict.keys()):
        if s not in bar_dict: continue
        px = bar_dict[s].close
        if np.isnan(px): continue

        context.highest_prices[s] = max(context.highest_prices.get(s, 0), px)
        highest = context.highest_prices[s]

        if (px - highest) / highest <= context.trailing_stop_loss:
            order_target(s, 0)
            del context.highest_prices[s]
            continue

    # 2. 定期轮动
    context.days_count += 1
    if context.days_count % context.rebalance_days != 0: return
    if curr_ts not in context.signals.index.levels[0]: return

    today_sig = context.signals.loc[curr_ts]
    buy_map = today_sig[today_sig['target_weight'] > 0]['target_weight'].to_dict()

    for s in list(pos_dict.keys()):
        if s not in buy_map:
            order_target(s, 0)
            if s in context.highest_prices: del context.highest_prices[s]

    for s, w in buy_map.items():
        order_target_percent(s, w)
