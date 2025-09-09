%
% This script evaluates the quality of the predictions.  Do not
% edit it.
%
load testingdata.mat
%
% Extract the actual y values.
%
yviolent=testingdata(:,80);
ynonviolent=testingdata(:,81);
%
% Compute the MAPE
%
MAPEviolent=100*sum(abs(yviolent-predviolent)./yviolent)/length(yviolent);
MAPEnonviolent=100*sum(abs(ynonviolent-prednonviolent)./ynonviolent)/length(ynonviolent);
%
% Output the scores.
%
fprintf('The MAPE for violent crime is %.1f%%\n',MAPEviolent);
fprintf('The MAPE for nonviolent crime is %.1f%%\n',MAPEnonviolent);



