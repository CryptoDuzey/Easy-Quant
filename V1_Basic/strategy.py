def init(context):
    set_benchmark('000300.SH')
    set_commission(PerShare(type='stock', cost=0.0002))
    
    # 策略参数
    context.index_code = '000300.SH'
    context.hold_num = 5       # 满仓5只
    context.pool_size = 15     # 动量观察池
    context.momentum_days = 20 # 动量计算周期
    
    # 止损参数
    context.stop_loss_rate = 0.90  # 追踪止损线：从最高点回撤10%则卖出
    
    # --- 修复：必须在init中声明这些全局变量 ---
    context.pro_pool = []      # 动量股票池
    context.last_high = {}     # 用于追踪每只持仓股的最高价 {股票代码: 最高价}
    
    log.info('🚀 终极融合策略初始化完成')

def handle_bar(context, bar_dict):
    # 1. 大盘双均线风控 (第一道防线：环境识别)
    idx_hist = history(context.index_code, ['close'], 30, '1d', False, 'pre', is_panel=1)['close']
    idx_ma5 = idx_hist.iloc[-5:].mean()
    idx_ma20 = idx_hist.iloc[-20:].mean()
    
    # 如果大盘 5日线 < 20日线，清仓离场
    if idx_ma5 < idx_ma20:
        if len(context.portfolio.stock_account.positions) > 0:
            log.warn('⚠️ 大盘趋势走弱（5/20日死叉），清仓避险')
            for s in list(context.portfolio.stock_account.positions):
                order_target(s, 0)
            context.last_high.clear() 
        return

    # 2. 动态追踪止损逻辑 (第二道防线：利润保护)
    hold_stocks = list(context.portfolio.stock_account.positions)
    for s in hold_stocks:
        curr_price = bar_dict[s].close
        # 更新该股买入后的最高价
        if s not in context.last_high:
            context.last_high[s] = curr_price
        else:
            context.last_high[s] = max(context.last_high[s], curr_price)
            
        # 检查是否触碰回撤止损线 (当前价 < 最高价 * 0.9)
        if curr_price < context.last_high[s] * context.stop_loss_rate:
            log.warn('🛑 追踪止损触发: %s (最高点 %.2f, 当前价 %.2f)' % (s, context.last_high[s], curr_price))
            order_target(s, 0)
            context.last_high.pop(s) 

    # 3. 更新动量池 (仅在周一或初始为空时执行)
    if get_datetime().weekday() == 0 or len(context.pro_pool) == 0:
        all_stocks = get_index_stocks('000300.SH')
        all_stocks = [s for s in all_stocks if not bar_dict[s].is_paused and not bar_dict[s].is_st]
        
        df = history(all_stocks, ['close'], context.momentum_days + 1, '1d', False, 'pre', is_panel=1)['close']
        mom_rank = (df.iloc[-1] / df.iloc[0] - 1).sort_values(ascending=False)
        context.pro_pool = mom_rank.head(context.pool_size).index.tolist()
        
        # 自动汰弱留强：不在新池子里的股清仓
        for s in list(context.portfolio.stock_account.positions):
            if s not in context.pro_pool:
                log.info('⬇️ 移出动量池: %s' % s)
                order_target(s, 0)
                if s in context.last_high: context.last_high.pop(s)

    # 4. 复合信号进场 (RSI + 突破)
    current_holdings = list(context.portfolio.stock_account.positions)
    for s in context.pro_pool:
        # 如果已持仓或满仓，跳过
        if s in current_holdings or len(current_holdings) >= context.hold_num:
            continue
            
        s_hist = history(s, ['close', 'high'], 30, '1d', False, 'pre', is_panel=1)
        # 计算 RSI
        diff = s_hist['close'].diff()
        up = diff.clip(lower=0).rolling(14).mean()
        down = -diff.clip(upper=0).rolling(14).mean()
        rsi = 100 * up / (up + down)
        curr_rsi = rsi.iloc[-1]
        # 计算突破位
        h20 = s_hist['high'].iloc[-21:-1].max()
        curr_price = bar_dict[s].close
        
        # 信号确认：RSI < 45 (强势股回调) 或 价格突破新高
        if curr_rsi < 45 or curr_price > h20:
            log.info('⬆️ 进场买入: %s (RSI:%.1f, 突破价:%.2f)' % (s, curr_rsi, h20))
            order_target_percent(s, 0.19)
            context.last_high[s] = curr_price # 买入时记录初始价格
            current_holdings.append(s) # 手动更新局部变量防止单根bar超买
