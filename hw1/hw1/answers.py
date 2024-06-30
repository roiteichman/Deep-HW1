r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. **False** - The test error helps to evaluate the generalization of the model and not the evaluation of the training error.
2. **False** - Lets say we have many numbers in the range [0,10]. If we will train on numbers from [0,2] it wont constitute a useful train-test split because it doesn't represent the data well.
3. **True** - In cross validation you use validation set, and not the test set.
4. **True** - The performance metrics obtained from the folds in cross-validation are averaged to provide an estimate of the model's generalization error.
"""

part1_q2 = r"""
**Your answer:**
Our friend's approach is not justified. 
He is using the test set as a validation set and chooses a lambda that gives the best results for the test set.
Therefore, the model might overfit the test set and not generalize well.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
The selection of $\Delta > 0$ is arbitrary because the regularization term ensures that the weight vectors remain bounded, and the relative trade-off between the margin size and model complexity can be adjusted through the regularization parameter 
$\lambda$. Therefore, while $\Delta$ sets the margin width, its specific value can be scaled and adjusted without altering the overall learning process, due to the presence of the regularization term.


"""

part2_q2 = r"""
**Your answer:**

From the plots above, we can see that the weights are derived from the image pixels: the brighter the pixel, the higher the corresponding weight value. For new predictions, the model classifies the sample as the class with the most similar average "lighting spread" to it.
Classification errors can occur when trying to classify a sample that has two different classes that are relatively close, 
for example, 6 and 5 or 4 and 9.


"""

part2_q3 = r"""
1. The graphs show that the loss function decreases with each iteration while accuracy increases, indicating convergence to a low error and high accuracy. This suggests that our chosen learning rate is appropriate. A larger learning rate might have caused the model to miss the convergence point and diverge, whereas a smaller learning rate would have required many more iterations to reach convergence and optimal results.

2. Based on the graphs, we can see that the performance on the validation set is pretty similar to the performance on the training set. This indicates that the model generalizes well to unseen data and is not overfitted to the training data.
However, if we must chose one of the options in the question so we will chose: Slightly overfitted. We can see that the accuracy on the train set is better than the accuracy on the validation set. This indicates a lack of sufficient generalization and suggests that the model is slightly overfitted to the training data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

We aim to generate a plot where the points are scattered randomly around the x-axis with a zero mean. By examining the plots, we can ascertain that cross-validation has resulted in improvement, as evidenced by smaller residuals post-validation. In the initial plot, the residuals were much larger compared to those in the final plot.
"""

part3_q2 = r"""
**Your answer:**

1. The model remains a linear regression model even after incorporating non-linear features.
Although the representation of the data changes with the addition of non-linear transformations,
the underlying model structure remains linear because it seeks a linear combination of
features that best fits the data.

2. Adding non-linear features allows us to represent the data in a transformed space that increases the flexability of the model.
Therefore, we can fit any model with complex enough features. However, it might not generalize well to unseen data.

3. The decision boundary in the original feature space will become non-linear due to the addition of non-linear features. However, in the new feature dimension, the decision boundary will be a hyperplane.

"""

part3_q3 = r"""
**Your answer:**

1. Using np.logspace instead of np.linspace allows us to generate values for $\lambda$ that are logarithmically spaced. This means smaller differences between values for lower $\lambda$ values and larger differences as $\lambda$ increases. As a result, np.logspace enables us to efficiently test a wide range of $\lambda$ values with the same number of samples, offering a broader exploration of regularization strengths compared to np.linspace, which provides values spaced evenly.

2. We considered 20 different values for $\lambda$.
   We evaluated 3 different degrees.
   The cross-validation itself was performed using 3 folds.
   Therefore, each unique combination of $\lambda$ and degree was tested across all 3 folds.
   This totals to 20 (values for $\lambda$) * 3 (degrees) * 3 (folds) = 180 model fittings in total.
"""

# ==============

# ==============
