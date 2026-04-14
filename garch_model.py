from arch import arch_model

def calculate_volatility(returns):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    forecast = model_fit.forecast(horizon=1)
    volatility = forecast.variance.iloc[-1].values[0] ** 0.5

    return volatility