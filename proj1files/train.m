%
% Sample training script.  This script uses ordinary linear
% regression to build a model of the data.  
%
%
% Read in the data.
%
load trainingdata.mat
%
% Your code goes below this line
% ------------------------------------------------------
%
% Just use a simple least squares linear regression.  This will not
% be optimal for the MAPE measure.
%
% Separate out X and y.
%
X=trainingdata(:,1:79);
yviolent=trainingdata(:,80);
ynonviolent=trainingdata(:,81);
%
% Insert a column of ones for the beta0 coefficients.
%
X=[ones(size(X,1),1) X];
%
% Find the least squares solutions.
%
betaviolent=X\yviolent;
betanonviolent=X\ynonviolent;
%
% Save the coefficients for later use. The "-V4" option saves a
% file that can be read by Octave as well as MATLAB. 
%
save -V4 beta.mat betaviolent betanonviolent
%
% Output the performance on the training data.
%
MAPEviolent=100*sum(abs(X*betaviolent-yviolent)./yviolent)/length(yviolent);
fprintf(['MAPE for the violent crime rate training set is %.1f\n'],MAPEviolent);
MAPEnonviolent=100*sum(abs(X*betanonviolent-ynonviolent)./ynonviolent)/ length(ynonviolent);
fprintf(['MAPE for the nonviolent crime rate training set is %.1f\n'],MAPEnonviolent);


