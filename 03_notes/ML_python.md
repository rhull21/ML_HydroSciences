## Machine Learning in Python:
### Overview
  1. A general workflow for ML and DL modeling:
    1. Define Problem.
    2. Prepare Data.
    3. Evaluate Algorithms.
    4. Improve Results.
    5. Present Results

  2. ML/DL tools and techniques in Python
    * SciPy - Science package
    * Sklearn - traditional ML
        Logistic Regression (LR)
        Linear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN).
        Classification and Regression Trees (CART).
        Gaussian Naive Bayes (NB).
        Support Vector Machines (SVM).
        K-Fold Cross-Validation
    * pytorch - up and coming DL techniques
    * Keras and TensorFlow - traditional DL techniques

### Notes
  1. https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    * SciPy
    * Workflow
        Define Problem.
        Prepare Data.
        Evaluate Algorithms.
        Improve Results.
        Present Results
    * Models used:
        Logistic Regression (LR)
        Linear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN).
        Classification and Regression Trees (CART).
        Gaussian Naive Bayes (NB).
        Support Vector Machines (SVM).
      * Are any of these truly DL?
    * Further Reading:
      * K-Fold Cross-Validation
        * https://machinelearningmastery.com/k-fold-cross-validation/
          * That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
          * There are common tactics that you can use to select the value of k for your dataset.
          * There are commonly used variations on cross-validation such as stratified and repeated that are available in scikit-learn.
          * The general procedure:
              1 Shuffle the dataset randomly.
              2 Split the dataset into k groups
              3 For each unique group:
                Take the group as a hold out or test data set
                Take the remaining groups as a training data set
                Fit a model on the training set and evaluate it on the test set
                Retain the evaluation score and discard the model
              4 Summarize the skill of the model using the sample of model evaluation scores
          * Care should be taken in choosing 'k'
            * a common approach is to choose k = 5 or 10
            * more on k-fold https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
            There are a number of variations on the k-fold cross validation procedure.
          * variations on K-fold cross-validation
            * Three commonly used variations are as follows:

              * Train/Test Split: Taken to one extreme, k may be set to 2 (not 1) such that a single train/test split is created to evaluate the model.
              * LOOCV: Taken to another extreme, k may be set to the total number of observations in the dataset such that each observation is given a chance to be the held out of the dataset. This is called leave-one-out cross-validation, or LOOCV for short.
              * Stratified: The splitting of data into folds may be governed by criteria such as ensuring that each fold has the same proportion of observations with a given categorical value, such as the class outcome value. This is called stratified cross-validation.
              * Repeated: This is where the k-fold cross-validation procedure is repeated n times, where importantly, the data sample is shuffled prior to each repetition, which results in a different split of the sample.
              * Nested: This is where k-fold cross-validation is performed within each fold of cross-validation, often to perform hyperparameter tuning during model evaluation. This is called nested cross-validation or double cross-validation.

  2. https://pytorch.org/
    * Deep Learning
      * Tutorials: https://pytorch.org/tutorials/

  3. https://medium.com/towards-artificial-intelligence/machine-learning-algorithms-for-beginners-with-python-code-examples-ml-19c6afd60daa
    * Detailed
      * Preferred ML Algorithms
        * Numpy
        * SciPy
        * Matpotlib
        * Scikit-learn
    * This is basically about how to use regression

  4. Moshe 2020:
    * Eventually Figure it out

  5. https://earthml.holoviz.org/
    * EarthML
      * Tools
        * Keras
        * TensorFlow
        * PyTorch

  6. MachineLearning in Python Course
    * https://www2.atmos.umd.edu/~xliang/aosc447/
