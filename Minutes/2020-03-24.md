# Minutes 24th March

## Methodology

- Implement 'look-back stopping': let the model run for a longer time and check afterwards what has been the best stopping time under two conditions: A) Chi2_training < treshold (eg 1.5) and B) Chi2_validation has an absolute minimum. Use this to define the best stopping criteria. 
- Construct a Monte Carlo representation to determine the uncertainties in the model (also used in Tutorial 2 of Juan's online course). Goal is to train the model on the Monte Carlo replicas and find the uncertainties; this represents the variances from the original data. 


## Checks

- A good model should lead to chi2/ndat \sim 1 in the training data. Running the training longer should really let the Chi2 approach 1.
- Check what happens when we change the amount of validation data (now it is 0.2, decrease and increase to eg 0.5).
- Repeat training with random seeds such that N_model >> 1 and find uncertaincies in predictions.

## Lookout

### Few notes for in a later stage: 
- Correlations between data points;
- Fit both a NN and a Gaussian in the script to compare the results of both methods;
- Extrapolation: in this region, if the Chi2 goes to values >>1 then the model is not predicting the data well, which means that something else must be going on. In this way we can use the uncertainties to really make a difference from the manual fitting method.
