import pandas as pd
from mcos.utils import convert_price_history


def test_convert_price_history():
    df = pd.read_csv('test/stock_prices.csv', parse_dates=True, index_col='date')
    mu, cov = convert_price_history(df)

    assert mu["GOOG"], 0.26770283812412754
    assert mu['AAPL'], 0.36378640487631986
    assert cov['GOOG']['AAPL'], 0.04620227917174632
    assert cov['GE']['FB'], 0.014789747995647461
