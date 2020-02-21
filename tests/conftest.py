import pandas as pd
import pytest


@pytest.fixture
def prices_df() -> pd.DataFrame:
    return pd.read_csv('tests/stock_prices.csv', parse_dates=True, index_col='date')
