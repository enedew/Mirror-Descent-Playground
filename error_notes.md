### Errors found throughout development and testing

# MLP config and running errors
* Incrementing the learning rate sometimes causes NaN values to be predicted on epoch 0, causing a crash once the calculate_metrics function is called 