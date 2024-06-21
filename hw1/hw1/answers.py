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

2. Slightly overfitted – based on the graphs, we can see that the performance on the validation set is slightly less better than on the training set. This indicates a lack of sufficient generalization and suggests that the model is slightly overfitted to the training data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
