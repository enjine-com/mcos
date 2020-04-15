# Monte Carlo Optimization Simulation

This library is the fulfillment of all your hopes and dreams, assuming your hopes and dreams consist entirely of an implementation of the Monte Carlo Optimization selection from Dr. Marcos Lopez de Prado's 2019 paper ['A Robust Estimator of the Efficient Frontier'](https://poseidon01.ssrn.com/delivery.php?ID=560120086072067000023119067087001005063062020029025039112121114081030090076000064068060031127103104029043093066122064093066123051020006028053100121067093122114073005020005127087082126007008124024106088066094097086072096024113073076127127015020000085&EXT=pdf). Those are some weirdly specific hopes and dreams. But you're our kind of people. MCOS allows you to compare the allocation error of different portfolio optimization methods, given a particular set of inputs. 


## WHY MCOS?

Optimizing a portfolio is like petting a cat: the same approach doesn't work for every situation. Cats and markets are both complex creatures and you need to be cautious before you dive in. It is naive to think that one method is best for everything until the end of time. MCOS allows you to quickly compare different optimization methods to find which is most robust in your particular case.

## HOW?

After you have calculated the covariance matrix and the expected returns for your portfolio you feed them into the simulator. Using the optimization methods you've selected, the library then calculates the optimal weights. Then a Monte Carlo simulation is performed, where optimal allocations are computed on a large number of simulated covariances and returns. It then compares these allocations to the ideal and calculates the error. 


## GETTING STARTED

Install and update using pip:
> pip install mcos

## RUNNING SIMULATIONS

Before using the MCOS library, it is assumed that you have calculated both the covariance and the expected return vectors of your portfolio for all of the optimizers that you wish to compare. 

The main entry point for the library is the mcos.py file. The entry points are mcos.simulate_observations() and mcos.simulate_optimization_from_price_history(). mcos.simulate_observations requires that you have already calculated both expected returns and covariance, whereas mcos.simulate_optimization_from_price_history() allows you to pass a price history DataFrame that the system will calculate expected returns and covariance from. These functions complete the MCOS procedure for all of the specified optimizers, and returns to you a DataFrame with the results. It takes an observation simulator, the number of simulations you'd like to run, a list of all the optimizers you would like to compare, the type of metric you'd like to test, and an optional covariance transformer. All of these are described below. 

## SAMPLE SIMULATION

> #simulate_optimizations
>
> obs_sim = MuCovObservationSimulator(expected_returns, covariance, num_sims)  
> optimizers = [HRPOptimizer(), MarkowitzOptimizer(),NCOOptimizer(max_num_clusters, num_clustering_trials)]   
> error_estimator = VarianceErrorEstimator()  
> covariance_transformers = [DeNoiserCovarianceTransformer()]  
> num_sims = 50  
>    
> results = mcos.simulate_optimizations(obs_sim, num_sims, optimizers, error_estimator, covariance_transformers)  


> #simulate_optimizations_from_price_history
>
> optimizers = [HRPOptimizer(), MarkowitzOptimizer(),NCOOptimizer(max_num_clusters, num_clustering_trials)]   
> error_estimator = VarianceErrorEstimator()  
> covariance_transformers = [DeNoiserCovarianceTransformer()]         
> simulator_name = "MuCov"  
> num_sims = 50  
> num_observations = 50  
>   
> results = mcos.simulate_optimizations_from_price_history(price_history, simulator_name, num_observations, num_sims, optimizers, error_estimator, covariance_transformers)  


## DATA INPUTS

As mentioned above, when calling mcos.simulate_optimizations() the key input for the system is the expected return vector and covariance of the portfolio that you are trying to analyze. You choose the type of Observation Simulator and initialize it with the covariance and expected returns. Please feed the Simulator only numpy arrays, lest it get cranky and uncooperative. The expected return vector is a 1 dimensional array of expected returns for the portfolio constituents, while the covariance is a n x n matrix. You must also include the number of observations that you wish to run.


If you are calling mcos.simulate_optimizations_from_price_history(), expected return vector and covariance are calculate for you. So instead of passing an Observation Simulator object, you would instead pass the price history, the name of the simulator you'd like to run, and the number of observations you'd like make. 

The observation simulators that are currently supported are:

1. Standard - The chicken fingers of simulators. Plain, unexciting, but darn it, it gets the job done. Regular estimation of the covariance matrix. For simulate_optimizations_from_price_history calls, pass "MuCov" as the simulator name. 

2. Ledoit-Wolf - If you prefer your covariance matrix shrunken, this is the one for you. Read all about it in this [unfortunately titled paper](http://www.ledoit.net/honey.pdf). For simulate_optimizations_from_price_history calls, pass "MuCovLedoitWolf" as the simulator name
  


## CONFIGURATION INPUTS

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

You may also pass in an instance of a CovarianceTransformer if you so choose. This can be useful in helping remove some error from the simulation due to things such as noise. Currently we only have this single transformer available:  

1. Denoiser Transformer - as detailed in [this paper](https://poseidon01.ssrn.com/delivery.php?ID=489024064102117109091077096101101064027075072041043035077073019004118011104120069072123098043034107058119101127077107089081076059012026078015006095118070112111086032085044067091079116085069123114124013083086031102022097077123007004068111066094003118&EXT=pdf) by Dr. Lopez de Prado, this transformer helps shrinks the noise to aid in the simulation. 

## RETURN VALUES

The library will return to you a pandas DataFrame with the name of the optimizer, the mean of whichever error estimator you chose, and the standard deviation of the estimator. 

## AUTHORS

The library was constructed by the team at [Enjine](http://www.enjine.com).
