### Errors found throughout development and testing

# MLP config and running errors
* Incrementing the learning rate sometimes causes NaN values to be predicted on epoch 0, causing a crash once the calculate_metrics function is called 
    * This issue is fixed for the euclidean mirror map, by clipping the gradient norms - I think this helps restrict them from getting way too large and causing NaN parameters values.
    * Still an issue for using the KL mirror map - this is just when using the multi layer perceptron, as the weight parameters are not in the range (0, 1)

# GUI errors
* Current callback set up for run experiment button causes an error for minimise experiments, doesn't effect the experiment at all



## TO DO 
* Set up KL properly and ensure it's working - do the same for multple divergences
* Add an option for a linear regression model instead of MLP
* Allow variables to be vectors in functions
* Allow for multi variable functions 
* Switch between batch mirror descent and mini-batch mirror descent, also classic gradient descent to compare the difference with the (theoretically) equivalent euclidean-based mirror descent 
* Function presets - classic optimisation benchmark functions
* Allow for inputting a distance generating function instead of a bregman divergence. Feel like this would be tricky as would have to find a way to determine the mirror map and its inverse for any function. 










