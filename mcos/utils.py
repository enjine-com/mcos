import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns


def convert_price_history(df: pd.DataFrame):
    """
    converts a price history dataframe into expected returns and covariance
     :param df: Dataframe of price histories indexed by data
     @return tuple of DataFrames representing expected returns and covaraince matrix
    """
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    cov = risk_models.sample_cov(df)

    # convert to match our current format for mu, cov
    cov = cov.reset_index(drop=True)
    for count, x in enumerate(cov.columns):
        cov = cov.rename(columns={x:str(count)})
    return mu.to_numpy(), cov.to_numpy()
