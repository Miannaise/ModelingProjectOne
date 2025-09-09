%
% Sample script for using the fitted model to predict values.
%
% Your pred.m script should assume that all variables have been
% cleared before it is run- you'll need to load in a saved copy of
% any fitted model coefficient as in the example below.  Do not
% assume that the original training data will be available.
%
% Don't edit this part.
%
load testingdata.mat
%
% Put your code between the --- lines
% ------------------------------------------
%
% We'll load in the beta coefficients and use them to predict y.
%
load beta.mat
%
% Extract the predictors from testing data.
%
X=testingdata(:,1:79);
%
% Insert a column of ones.
%
X=[ones(size(X,1),1) X];
%
% Make the predictions.
%
predviolent=X*betaviolent;
prednonviolent=X*betanonviolent;
%
% End of prediction code.
%
% -------------------------------------------
%
% At this point, the variables predviolent and prednonviolent
% should be set.  They'll be used by the evaluate script to score
% your model.  
%