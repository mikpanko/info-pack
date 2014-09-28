function predictedTgts = LDAdecoder(trainX, trainY, testX, discrimType)

% by Scott Brincat

% Linear Discriminant Analysis decoder

% remove any predictors from training and testing X that are all 0
zeroPreds = all(trainX == 0,1);
if all(zeroPreds)
    predictedTgts = nan(size(testX,1),1);
    return;
end
trainX(:,zeroPreds)= [];
testX(:, zeroPreds) = [];

% fit LDA model to training data
dcdObj = ClassificationDiscriminant.fit(trainX, trainY, 'discrimType', discrimType);

% calculate LDA predictions on testing data
predictedTgts = predict(dcdObj, testX);

% % calculate LDA predictions using old MATLAB (<R2011b) routines
% predictedTgts = classify(testX, trainX, trainY, discrimType);

end