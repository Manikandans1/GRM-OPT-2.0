def calculate_future_value(monthly_investment, r, years):
    N = years * 12
    FV = monthly_investment * (((1 + r) ** N - 1) / r)
    return FV

def calculate_grm(fv, volatility):
    # return fv / volatility
    return fv / (volatility * 1000)