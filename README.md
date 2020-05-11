# Stock-Price-Prediction-with-multiple-inputs
Stock 'Open/Close' price prediction using LSTM RNN's with multiple inputs (both 'Open' and 'Close' prices).

The project tries to predict the stock close prices of Google(Aplhabet) for the month of April, 2020 using the LSTM (Long Short Term Memory cells) RNN's.

The data is taken from the Yahoo Finance between 2015/04/01 and 2020/03/31. The sudden rise and dips in the data (if any) are assumed to be free of any external factors (Though this is not true, and especially the given time frame includes the effect of Covid-19). The assumption is that the data is free of any and all external factors.

Note that this model is built to check how the predicted 'Close' price vary when compared to the predicted price given by the single-input-single-output model. In this model we will see multiple-input-single-output and multiple-input-multiple-output variations.

An in-depth full model is not presented here. For building a model from scratch to improving and optimizing, please check my other project [Stock-Price-Prediction](https://github.com/revanthtalluri001/Stock-Price-Prediction). The accuracy of the model is measured by its R-squared value.

## Results
- For multiple-input-single-output model, the initial accuracy is at 91.5%
- For multiple-input-multiple-output model, 
  - The initial accuracy for 'Open' price is at 85%
  - The initial accuracy for 'Close' price is at 82.5%
  
## Note
These models can be optimized and can be used to predict the future prices. For a complete process, have a look at [this](https://github.com/revanthtalluri001/Stock-Price-Prediction).
