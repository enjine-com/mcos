#Monte Carlo Optimization Simulation

This library is the fulfillment of all your hopes and dreams, assuming your hopes and dreams consist entirely of an implementation of the Monte Carlo Optimization selection from Dr. Marcos Lopez de Prado's 2019 paper ['A Robust Estimator of the Efficient Frontier'](https://poseidon01.ssrn.com/delivery.php?ID=560120086072067000023119067087001005063062020029025039112121114081030090076000064068060031127103104029043093066122064093066123051020006028053100121067093122114073005020005127087082126007008124024106088066094097086072096024113073076127127015020000085&EXT=pdf). Those are some weirdly specific hopes and dreams. But you're our kind of people. MCOS allows you to compare the allocation error of different portfolio optimization methods, given a particular set of inputs. 


##WHY MCOS?

Optimizing a portfolio is like petting a cat: the same approach doesn't work for every situation. Cats and markets are both complex creatures and you need to be cautious before you dive in. It is naive to think that one method is best for everything until the end of time. MCOS allows you to quickly compare different optimization methods to find which is most robust in your particular case.

##HOW?

After you have calculated the covariance matrix and the expected returns for your portfolio you feed them into the simulator. Using the optimization methods you've selected, the library then calculates the optimal weights. Then a Monte Carlo simulation is performed, where optimal allocations are computed on a large number of simulated covariances and returns. It then compares these allocations to the ideal and calculates the error. 


##GETTING STARTED

Install and update using pip:
> pip install mcos

##RUNNING SIMULATIONS

Before using the MCOS library, it is assumed that you have calculated both the covariance and the expected return vectors of your portfolio for all of the optimizers that you wish to compare. 

The main entry point for the library is the mcos.py file. The entry point is mcos.simulate_observations(). This function completes the MCOS procedure for all of the specified optimizers, and returns to you a DataFrame with the results. It takes an observation simulator, the number of simulations you'd like to run, a list of all the optimizers you would like to compare, the type of metric you'd like to test, and an optional covariance transformer. All of these are described below. 

The call to the library should look something like this:

> results = mcos.simulate_optimizations(obs_sim, num_sims, optimizers, error_estimator, covariance_transformers)


##DATA INPUTS

As mentioned above, the key input for the system is the expected return vector and covariance of the portfolio that you are trying to analyze. You choose the type of Observation Simulator (currently supported are standard and Ledoit-Wolf, which is detailed in this unfortunately titled paper) and initialize it with the covariance and expected returns. Please feed the Simulator only numpy arrays, lest it get cranky and uncooperative. The expected return vector is a 1 dimensional array of expected returns for the portfolio constituents, while the covariance is a n x n matrix. You must also include the number of simulations that you wish to run. 

> obs_sim = MuCovObservationSimulator(expected_returns, covariance, num_sims)

##CONFIGURATION INPUTS

Along with selecting your choice of  Observation Simulator, you can also specify the optimzers that you would like to compare. These are passed in as a list of Optimizer class objects. The Optimizers currently supported are: 

1. [Markowitz Optimization](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf) – Modern Portfolio Theory: the original gangster of portfolio optimizations created in 1952.

   > MarkowitzOptimizer()

2. [Nested Cluster Optimization](https://poseidon01.ssrn.com/delivery.php?ID=560120086072067000023119067087001005063062020029025039112121114081030090076000064068060031127103104029043093066122064093066123051020006028053100121067093122114073005020005127087082126007008124024106088066094097086072096024113073076127127015020000085&EXT=pdf) – Optimization developed by Marcos Lopez de Prado and laid out in his 2019 paper “A Robust Estimator of the Efficient Frontier”. There are two optional variables that you can pass to this specific optimizer. They are the maximum number of clusters to use during clustering, and the number of times to perform the clustering.  

   > NCOOptimizer(max_num_clusters, num_clustering_trials)

3. [Risk Parity](https://www.investopedia.com/terms/r/risk-parity.asp) – Risk Parity builds on the work of Markowitz to create portfolios that focus on diversifying risk instead of diversifying capital. If you do not want equal risk distribution you are able to initialize the Risk Parirty Optimizer with an array of weights of your choosing.

   >RiskParityOptimizer(weights_array)

4. [Hierarchical Risk Parity](http://620116007095114078106074071067113067035074090016037034077026115100120002078005085068098110016004116055039007017120016108004066098025029084039103017090030002008062017046068083006008123089028103069080108004112123027095076096004125124115092064072087/) – Another triumph for Dr. Lopez de Prado, as he details an optimization method that does not require inverting a covariance matrix. 

   > HRPOptimizer()

Almost as important as your choice in optimizer is your choice in error estimator. In order to compare something you need the criteria by which to judge. In the quant world we can't just say “this thing is better than that thing”, we need to say “this this thing is better than that thing based on this measure”. The MCOS library is no different. When you call the simulate_observations() function you will have to pass it an instance of the AbstractErrorEstimator class. The current available selections for the Error Estimators are:

1. Expected Outcome: Calculates the mean difference with respect to expected outcomes of the portfolios
    > ExpectedOutcomeErrorEstimator()
2. Variance: Calculate the mean difference in variance of the portfolios
    > VarianceErrorEstimator()
3. Sharpe Ratio: DONT USE THIS IT CAUSES THE WHOLE THING TO EXPLODE! Just kidding. As you can guess, this calculates the mean difference with respect to the Sharpe ratio of the portfolios
    > SharpeRatioErrorEstimator()

You may also pass in an instance of a CovarianceTransformer. Currently the only transformer available is  the DeNoiserCovarianceTransformer, which is described in deatail in this paper. Essentially, the denoising of the covariance matrix allows us tackle the instability that noise can bring into the calculations, greatly improving our simulation. 

##RETURN VALUES

The library will return to you a pandas DataFrame with the name of the optimizer, the mean of whichever error estimator you chose, and the standard deviation of the estimator. 

##AUTHORS

The library was constructed by the team at [Enjine](http://www.enjine.com).
