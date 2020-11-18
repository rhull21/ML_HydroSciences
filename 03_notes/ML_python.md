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
      * LSTM: We implemented a fast and flexible LSTM code that can utilize the highly-optimized NVIDIA CUDA® Deep Neural Network (cuDNN) library from the PyTorch deep learning platform.
    * Keras and TensorFlow - traditional DL techniques
    * Numba - a python compiler from Anaconda that ca compile Python code on CUDA-capable GPUs, and provides Python developers with an easy entry to GPU-accelerated computing. Using CUDA and Numba (from Nvidia)

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
      * Usage: Shen
        * We implemented a fast and flexible LSTM code that can utilize the highly-optimized NVIDIA CUDA® Deep Neural Network (cuDNN) library from the PyTorch deep learning platform.
      * Tutorials:
        * https://pytorch.org/tutorials/
          1. https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
            * See script notes pytorchblitz.py:
              * Loading Data:
                  * Convert from Numpy Array to torch.*Tensor
                  * Loading:  
                    * For images, packages such as Pillow, OpenCV are useful
                    * For audio, packages such as scipy and librosa
                    * For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful
                    * For vision, torchvision
          2. https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
            * See script pytorch_examples.py
              * Outline:
                1. implement network using numpy a kind of hack approach, but will work for very simple ML questions
                2. Introduce the Tensor PyTorch is necessary to utilize GPUs for complex DL Tensor is fundamental to PyTorch Similar to Numpy Arrays, with added bells and whistles including
                  1. track graph and gradients
                  2. additional scientific computing tools
                  3. GPUs can be used to accelerate numeric computations
                3. Autograd   autograd allows for automatic computation of gradients 'backward passes' in neural networks utilizes automatic differentiation because gradients.  are partial derivatives, after all
                4. Defining new autograd functions the autograd operator is actually two functions
                  * Forward function computes outputs from inputs
                  * backward function receives the gradient of the output tensors and computes the gradeint of the input tensors, counting backwards through all layers
                5. nn module computational graphs and autograd are powerful but too low level for large neural networks in this case, we arrange computation into layers which have learnable parameters that can be optimized during learning
                6. optim  the optim package in PyTorch abstracts the idea of an  an optimization algorithm and provides implementaions  of commonly used optimization algorithms
                7. Custom nn modules
                8. Control Flow and Weight Sharing  a fully-connected ReLU network that on each forward  pass chooses a random number between 1 and 4 and uses  that many hidden layers, reusing the same weights  multiple times to compute the innermost hidden layers.

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


  7. PythonDataScienceHandbook, ML:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html
    * Intro:
      * Supervised Learning:
        * classification
        * Regression
      * Unsupervised Learning:
        * Clustering
        * Dimensionality Reduction
    * Scikit-Learn


  8. PyTorch v TensorFlow
    * https://realpython.com/pytorch-vs-tensorflow/
    * TensorFlow, from google
      * note that it is integrated in Keras, which takes away 'a lot of the sweat'
      * Good for production-grade deep LEARNING
      * TensorFlow 2.0 is real good.
      * Downside - Manual:
        * required you to make an abstract syntax tree
        * compile via session.run()
      * A session object is a class for running TensorFlow operations: https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session
        * i.e. this is the environment in which operations are executed, and tensor objects are evaluated
        * It made sense to use session as a context manager -> https://realpython.com/courses/python-context-managers-and-with-statement/
      * Now things are made using eager execution, which is the way python normally works.
      * Continue later
    * pytorch
      * from Facebook
        * offered to make models easier, and it is integrated into python better
        * more widely used in research than in production
      * PyTorch is based on Torch, and built from C up to be optimized for Python so...
          * Better memory and optimization
          * More sensible error messages
          * Finer-grained control of model structure
          * More transparent model behavior
          * Better compatibility with NumPy
      * easy to move back and forth between numpy.array and torch.Tensor objects
          >>> import torch
          >>> import numpy as np

          >>> x = np.array([[2., 4., 6.]])
          >>> y = np.array([[1.], [3.], [5.]])

          >>> m = torch.mul(torch.from_numpy(x), torch.from_numpy(y))

          >>> m.numpy()

      * torch.Tensor.numpy() lets you print out the result of matrix multiplication as a numpy array
      * THE torch.Tensor class has different methods and attributes, like backward(), which computes the gradient and CUDA (GPU) compatability
      * Special Features:
        * module for auto-differentiation (https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
          * This automatically calculates the gradient of the functions in torch.nn during backpropogation
        * API - is built using fast.ai, which is eas
          * https://www.fast.ai/
        * TouchServe - TorchServe is a flexible and easy to use tool for serving PyTorch models.
          * https://github.com/pytorch/serve
        * TorchElastic - for training deep neural networks at scale using Kubernetes
          * https://pytorch.org/elastic/0.2.0rc0/kubernetes.html
          * https://github.com/pytorch/elastic/tree/master/kubernetes
            * TorchElastic Controller for Kubernetes manages a Kubernetes custom resource ElasticJob and makes it easy to run Torch Elastic workloads on Kubernetes.
          * PyTorch HUB - a community of researcher using pytorch
            * https://pytorch.org/hub/research-models

  9. Learning Rate:
    * Defintion: The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck, whereas a value too large may result in learning a sub-optimal set of weights too fast or an unstable training process. **This is one of ht emost important hyperparameters when configuring a neural network**
    * Tutorial:
      * https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
        * How large learning rates result in unstable training and tiny rates result in a failure to train.
        * Momentum can accelerate training and learning rate schedules can help to converge the optimization process.
        * Adaptive learning rates can accelerate training and alleviate some of the pressure of choosing a learning rate and learning rate schedule.


  10. https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network
    * Fantastic Learning Resource

  11. How to use GPU Accelerated Computing in Python
    * NUMBA https://developer.nvidia.com/how-to-cuda-python
      * Traditionally Python had been considered too slow for high-performance computing
      * Numba is a python compiler from Conda that can compile Python code for execution on CUDA-capable
      * More Tutorials:
        * https://developer.nvidia.com/blog/numba-python-cuda-acceleration/
        * https://github.com/ContinuumIO/gtc2017-numba
    * While NUMBA is a good option for compiling generic python code, pytorch actually has built-in gpu-enabling via CuDa.
      * Refs:
        * https://colab.research.google.com/drive/1OnA-zV3xHDlmL6S6OHp42JoB3PCYm1z_?authuser=1
        * https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
        * https://groups.google.com/g/torch7/c/CkB57025yRY?pli=1
      * Approaches:
        * You can utilize gpu by dimming tensor variables with `.to(cuda)`
        * use `.cuda()` on any input batches/tensors
        * use `.cuda()` on your network module, which will hold your network.
        * `torch.set_default_tensor_type('torch.cuda.FloatTensor')`
        * `cudnn.benchmark = True`
    * Accelerated Deep Learning on MacBook with PyTorch: the eGPU
      * Refs:
        * https://medium.com/@janne.spijkervet/accelerated-deep-learning-on-a-macbook-with-pytorch-the-egpu-nvidia-titan-xp-3eb380548d91
      * Approaches:
        * Install macOS-eGPU.sh
        * CUDA Installation
        * cuDNN Installation
        * Conda Installation
        * Compile and install PyTorch
