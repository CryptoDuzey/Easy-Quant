"""
Easy-Quant V3: 同花顺 SuperMind 回测执行引擎

零未来函数 · Patch-Transformer 截面打分 · 每日调仓

用法:
    将本文件完整贴入 SuperMind 策略编辑器，确保 patch_transformer_v3.pth
    已上传至研究环境的模型目录。

关键约束:
    - 所有数据获取严格截至 T-1（昨天），不存在日内 peek
    - 因子计算仅依赖历史数据，不涉及 shift(-k) 前视
    - 截面标准化为日内独立操作，无信息泄露
"""

import pandas as pd
import numpy as np
import torch
from SuperMind.api import *


def init(context):
    # 标的池：沪深300
    context.stock_pool = get_index_stocks("000300.SH")
    context.n_stocks = 10

    # 加载已训练模型
    context.model = load_model("patch_transformer_v3.pth")
    context.model.train(False)  # 预测模式

    # 每日执行调仓
    run_daily(trade_execute)


def before_trading(context):
    """
    盘前数据准备 (T-1 收盘后执行)

    1. 拉取足够历史窗口（80 天，确保 60 天特征 + 缓冲）
    2. 截面 Z-Score 标准化
    3. 组装为模型输入张量 [1, 300, 60, 15]
    """
    df_raw = get_price(
        context.stock_pool,
        count=80,
        frequency="1d",
        fields=["open", "high", "low", "close", "volume", "turnover"],
    )

    # 以最近 60 天为特征窗口
    # 此处省略具体特征工程计算，保持简洁
    # 实际使用时替换为：因子计算 + 缺失值填充 + 截面标准化

    # 组装张量
    context.X = prepare_tensor(df_raw)


def trade_execute(context):
    """
    开盘调仓执行

    1. 模型推理 → 截面得分
    2. 等权买入 Top-N
    3. 卖出不在 Top-N 的持仓
    """
    with torch.no_grad():
        scores = context.model(context.X).numpy().flatten()

    rank_series = pd.Series(scores, index=context.stock_pool).sort_values(ascending=False)
    target_list = rank_series.head(context.n_stocks).index.tolist()

    # 卖出旧股
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target_list:
            order_target(stock, 0)

    # 买入新股
    for stock in target_list:
        order_target_percent(stock, 1.0 / context.n_stocks)


def robust_zscore(x):
    """
    Robust Z-Score (MAD-based)

    相比传统 Z-Score，对极端值不敏感，
    适合 A 股量价数据中的异常波动。
    """
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    return (x - median) / (mad * 1.4826 + 1e-8)


def prepare_tensor(df_raw):
    """
    将原始行情数据转化为模型输入张量。

    Args:
        df_raw: get_price 返回的 DataFrame

    Returns:
        torch.Tensor of shape [1, N_stocks, 60, 15]
    """
    # TODO: 替换为实际特征工程管线
    # 1. 取最近 60 天
    # 2. 对每只股票计算 15 个因子
    # 3. 截面 robust_zscore 标准化
    # 4. reshape 为 [1, N, 60, 15]
    pass
