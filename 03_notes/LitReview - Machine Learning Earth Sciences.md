## Machine Learning References and Notes:
### Overview

1. Opportunities:
  * use physical background knowledge to build statistical models (ie incorporate physical process method)
  * use NN to reduce complexity of numerical simulation
    * ie fluid dynamics, which can be computationally demanding
  * use sparse regression method for discovering governing partial differential equations
  * Optical Flow
  * Video Prediction
  * Integration with physical modelling:
    * (1) Improving parameterizations
    * (2) Replacing physical 'sub-model' with machine learning model
    * (3) Analysis of model-observation mismatch
    * (4) Constraining submodels
    * (5) Surrogate modelling or emulation
      * faster sensitivity analysis, model parameter calibration, derivation of confidence intervals
      * 'emulators'
  * 1 Interdisciplinarity and human dynamics (more complicated and interconnected systems)
  * 2 Data Deluge and Data Discoverability (finding and interpreting data)
  * 3 Unrecognized / Unrepresented Linkages (exploring complex correlations, such as feedabcks between flows, ecosystem dmo graphics, rooting depth, and root hydraulics)
  * 4 Model Scaling and Equifinality Challenge
    * model scaling: (governing equations like Darcy's Law and Richard's Equation don't scale well from laboratory conditions up to field conditions, and resolving dynamics at field scale is too expensive)
    * equifinality: (multiple models generate the smae outcomes)
  * 5 regionalized parameters and models: (big models need to be regionalized (redone) when applied to smaller regions in order to produce god results. However, typical approaches suffer from lack of knowledge and lack of data for calibration and parameterization)

4.
### Readings
1. Emmanuel de Bézenac et al J. Stat. Mech. (2019) 124009, Deep learning for physical processes:
incorporating prior scientific knowledge
  * Advances in deep learning but...
  * machine learning is not ready for the level of complexity of difficult questions

  * 1 Introduction
    * Two approaches to modeling: (i) physical process method, (ii) statistical Machine Learning (ML) method
      * physical process - model physic using 'analytic descriptions' of underlying processes (differential eqs)
      * statistical machine learning - prior-agnostic approach
    * ML is new and doesn't work great, but doesn't require 'scientists prior knowledge' to work.
    * Goals:
      (1) are modern ML techniques ready to be used to model complex physical phenomena, and
      (2) how general knowledge gained from the physical modeling paradigm could help designing efficient ML models.
    * NN = Neural Network models
    * PDE = Partial Differential Equations

  * 5 ML for physical modeling:
    * use physical background knowledge to build statistical models (ie incorporate physical process method)
    * use NN to reduce complexity of numerical simulation
      * ie fluid dynamics, which can be computationally demanding
    * use sparse regression method for discovering governing partial differential equations
    * Optical Flow
    * Video Prediction

  * Conclusion and Future Work:
    * Cross Fertilization of traditional and innovative ML methods is essential


  * Further reading:
    * Cressie an Dikle 2015 - use physical background knowledge to build statistical models
    * Rudy et al (2017) make use of a sparse regression method for discovering
        the governing partial dierential equation(s) of a given system by time series measurements
        in the spatial domain.

2. Reichstein et al, Deep learning and process understanding for data-driven Earth system science, https://doi.org/10.1038/s41586-019-0912-1
  * Introduction
    * advocating a hybrid modelling approach, coupling physical process models with the versatility of data-driven machine learning.
    * short-term, simple prediction -> progress (weather); long-term, complex prediction -> no progress (climate)
    * earth system data are examplary of 'big data': volume, velocity, variety, and veracity.
      * So we need big data approaches to extracted meaningful information
    * major tasks in the future:
      (1) extract knowledge from data deluge
      (2) derive models that learn better than traditional data assimilation approaches, while respecting nature's laws

    * Further reading:
      * Developers big data: https://developers.google.com/machine-learning/glossary/
      * Developers big data: http://www.wildml.com/deep-learning-glossary/

  * State-of-the-art geoscientific machine learning
    * better exploit spatial and temporal structures in data
    * regression
    * map temporally varying features onto temporally varying target variables in land, ocean, and atmosphere
      * example carbon (CO2) fluxes
    * random forest

  * pitfalls:
    * naive extrapolation
    * sampling / other biases
    * ignorance of confounding factors
    * interpretation of statistical association as causal relation
    * fundamental flaws in hypothesis testing ("P-fishing")

  * Deep Learning Opportunities in Earth System science
    * Key Opportunity-classes:
      * classification, anomaly detection, regression, space- or time-dependent state prediction
      * spatial + sequence learning (video + motion prediction)

  * Deep-learning challenges in Earth system science
    * Violations of key assumptions
    * more complicated data sets -> computational challenges
    * large labelled geoscientific datasets do not always exist in geoscience, in part owing to conceptual difficulty in labelling datasets
    * geoscientific problems are often under-constrained
        (the extrapolation problem)
    Challenges (1) Interpretability, (2) Physical consistency, (3) complex and uncertain data,
      (4) limited labels, (5) computational demand

  * Integration with physical modelling:
    * (1) Improving parameterizations
    * (2) Replacing physical 'sub-model' with machine learning model
    * (3) Analysis of model-observation mismatch
    * (4) Constraining submodels
    * (5) Surrogate modelling or emulation
      * faster sensitivity analysis, model parameter calibration, derivation of confidence intervals
      * 'emulators'

  * Conclusion (recommendations)
    * Recognition of the particularities of the data
    * Plausibility and Interpretability of inferences
    * Uncertain estimation
    * testing against complex physical models

  * Further Reading:
    * random forest method
    * Machine Learning v Deep Learning (?)
    * emulators

3. Chaopeng Shen, 2018, 'A Transdisciplinary Review of Deep Learning Research and Its Relevance for Water Resources Scientists'
  * Abstract:
    * DL can help address new / old challenges such as:
      * interdisciplinarity
      * data discoverability
      * hydrologic scaling
      * equifinality
      * parameter regionalization
    * DL is especially suited for information extraction from (1) images and (2) sequential data
    * DL as an exploratory tool

  * Motivations:
    * 1 Interdisciplinarity and human dynamics (more complicated and interconnected systems)
    * 2 Data Deluge and Data Discoverability (finding and interpreting data)
    * 3 Unrecognized / Unrepresented Linkages (exploring complex correlations, such as feedabcks between flows, ecosystem dmo graphics, rooting depth, and root hydraulics)
    * 4 Model Scaling and Equifinality Challenge
      * model scaling: (governing equations like Darcy's Law and Richard's Equation don't scale well from laboratory conditions up to field conditions, and resolving dynamics at field scale is too expensive)
      * equifinality: (multiple models generate the smae outcomes)
    * 5 regionalized parameters and models: (big models need to be regionalized (redone) when applied to smaller regions in order to produce god results. However, typical approaches suffer from lack of knowledge and lack of data for calibration and parameterization)

  * 2 DL Basics:
    * 2.1 Supervised v Unsupervised learning
      * supervised learning is trained to predict some observed trarget variable(s) (categorical or continuous), given some input attributes
        * target variable (dependent variables or labeled data)
          * tasks
            * classification tasks (for categorical data)
            * regression tasks (continuous data) <- more common
      * unsupervised learning is not given a target, and so seeks to learn how to represent input data instances efficiently and meaningfully by finding hidden structures.
        * hidden structures include principal components, clusters, and parameters of data distributions.
        * ex: Principal Component Analysis (PCA)
        * typically employed to remove less important details and leave only key bits of information that are most important
      * ANN
        * ANN is supervised learning to approximate functions by connecting units call neurons or cells
        * a cell receives inputs from (multiple) cells in the above layer, and uses it to generate an output sent to (multiple) cells in the below layer.
        * the tuning process (called training) defines the transformation between inputs and outputs
          * initially, the neural network is an 'empty vessel'
          * over time, the training 'teaches' the neural network its 'purpose'
          * In training, the neural network learns by minimizing the 'loss' function (mismatch between data and predictions)
          * An efficient and common method is 'backpropagation'
        * Multidimensional, where # neurons in a layer is 'width' and number of layers is called 'depth' of network
          * a deep network implies a large depth
        * testing error refers to errors incurred when the algorithm is applied to data not in the training set.
          * also known as the generalization errors
          * when applied to forward run, the network is making an 'inference'
        * nb, these days the term ANN is usually reserved for nondeep networks.
    * 2.2 deep v nondeep Learning
      * no clearcut definition
      * DL is generally large, multilayer neural networks workign directly on big, raw data.
        * DL: allow for automatic extraction of abstract features (or representations, as they are sometimes called), which are the discriminative information about the data.
        * nondeep: require input features to be constructed by expert humans
        * DL: enambles transfer learning (use of the model on another task)
        * DL: increased depth allows exponential growth of the ability to represent complex functions.
      * drawbacks to going so so so deep:
        * vanishing gradient - the training signal becomes exponentially small as it propogates into the networks
          * this makes backpropogation ineffective for deep networks
    * 2.3 Popular DL Network encoders:
      * CNN - for image tasks and LSTM (Long Short-Term Memory)
      * Multilayer perceptron (MLP) popular in hydrology, but not the architecture of choice.
      * Autoencoders and SDAEs
        * Autoencoders create a bottleneck that reduces the size of the input, and then tries to produce an output identical to the input using the limited data created by the bottleneck.
        * Denoising Autoencoders train the network to reproduce the inputs inexactly, with 'noise corruption', so as to prevent over fitting
        * SDAE - stacks of multiple denoising Autoencoders
      * CNN a cascade of layers that shrink in wideth from input to output
        * reminiscent of a geometric multigrid matrix solver
      * LSTM
      * DBN
    * 2.4 Regularization (Model Complexity Penalization) Techniques
      * empty
    * 2.5 Generative Adversarial Networks
      * empty
    * 2.6 Hardware Innovations and Software Support
      * empty
    * 2.7 Understanding the Generalization Power of DL
      * empty

  * 3 Transdisciplinary Applications of DL and Its Interpretation
    * 3.1. DL Applications in Sciences
      * hydrology (limited)
        * (i) extracting hydrometeorological and hydrologic information from images
          * predicting precipitation.
          * using timeseries
        * (ii) dynamic modeling of hydrologic variables or data collected by sensor networks
          * time series of inflow at three gorges, learning from trend, period, and random elements of inflow
          * FSKY17
        * (iii) learning and generating complex data distributions.
          * geological media simulation, helping to model pollutant transport in Groundwater
    * 3.2 The budding area of AI interpretation and its application in sciences
      * AI could be a useful knowledge discovery tool
      * AI neuroscience
        * learn from AI - not black box, but gray box

  * 4 Tackling Water Resources Challenges With the Help of DL
    * 1) DL can be water scientists’ operational approach to modeling interdisciplinary processes for which mathematical   formulations are not well defined but sufficient data exist, especially those related to human dynamics. (ie flooding risks, irrigation and consumptive water demands, water saving strategies)
      * mathematically difficult, not all processes known
      * its reasonable to expect that DL can extract or model collective human behaviors related to Water
    * 2) harness the power of big data amidst the emergence of new data sources.
      * ex: use images to extract ecosystem states like flood inundation, water levels, irrigation amounts,
      precipitation, vegetation stress, and other observable evetns.
      * off the shelf DL could work
      * challenge (find needed data sets)
    * 3) DL-based data-driven models can measure whether a new source of information could help model a phenomenon of interest.
        * could uncover relationships between landcover, climate, soil, and geomorphology.
        * off the shelf DL algorithms could work
        * challenge (find needed data sets)
    * 4) scaling and equifinality:
      * eg. climate model downscaling
      * DL can help redict effective model parameters, for example runoff coefficient or variable infiltration curve parameters, from raw data.
    * Other uses:
      * GAN can be used for stochastic generation of variables from weather to hydraulic conductivity
        * using Gan that respect observed constraints
      * Doing hydrology backward, inferring driving forces from outcomes
      * quantitative changes in predictive accuracy sometimes bring ofrth qualitative change in our Understanding
4. Zhao, PHysics Constrained Machine Learning of Evapotranspiration
  * use of a physics constrained machine learning model (hybrid model to yield ecosystem ET estimate
    * better than just the ML model, but not better than just physics model?

5. Tartakovsky, Physics-Informed Deep Neural Networks for Learning Parameters and... in Subsurface Flow Problems
  Physics constraints improve the accuracy of machine learning methods, especially
    when learning from sparse data.

  Physics constraints allow learning constitutive relationships without direct obser-
    vations of the quantities of interest.

  For considered examples, the proposed physics-informed neural networks provide a more accurate  parameter estimation than the maximum a posteriori probability method.

  Thoughts: This is pretty impressive because it beats the state of the art posteriori probability method

  Works for Specifically, we consider saturated flow in heterogeneous porous media with 80 unknown conductivity and unsaturated flow in homogeneous porous media with an un-81 known relationship between capillary pressure and unsaturated conductivity.

  * Further Reading:
    * Parameter estimation - J. Carrera https://link.springer.com/chapter/10.1007/978-94-009-2301-0_15
      * Groundwater modelling involves several steps
        1. conceptualization
        2. parameter estimation
          * note that model parameters in gwater are things like transmissivity, storage coefficient, porosity, recharge, etc...
          * the model parameters must be close to their physical counterparts if the model is expected to closely produce the behavior of the real system
        3. error analysis on the parameterizations
        4. Prediction
        5. error analysis on the predictions
      * Automatic parameter estimation can help overcome trial and error 'attempts' at using a satisfactory model result
        * The most interesting part is that parameter estimation can allow the modellerto try alternative conceptual models and choose the one that best fits not only available data but also non-quantifiable information and the modeller's subjective conception of the system.
    * ANN, neurons, cells, ReLU
    * What's the difference between ANN and other spatial statistical methods?
      * earlier methods were mostly not designed with relevant architectural elements to enable automatic extraction of features. Thus their input features must be computed from raw data with formulas provided by domain experts.
      * Many methods do not scale well for large data sometimes
      * Cannot automatically extract features
    * Equifinality

6. 2020. Youchan Hu, Stream-Flow Forecasting of Small Rivers Based on LSTM
  * LSTM (Long Short Term Memory) is a kind of circular memory neural network developed from RNN (Recurrent Neural Network)  
  * LSTM has memory, and every output is based on previous outputs, thus has the ability to take advantage of information
  between time-series data
  * Uses large amount of stream flow data and rainfall data collected nearby to estimate future stream flow
  * Compares with traditional ML (like SVR) (Support Vector Regression) and MLP (Multilayer Perceptions).
    * Results in:
      * Better Model Stability
      * Better Model Reliability
      * More Intelligent Capturing Features of Data
  * Models:
    * RNN: connected neurons that are self-looped. Good for handling temporal behaviors
      * hidden state (ht) is function of previous hidden state and current state f(xt, ht-1)
      * predicted state (yt) = f(xt, ht)
      * note that hidden state is weighted via transformation of non-linear function, like tanh or sigmoid
      * one drawback: the error of backward propogation depends on the weights in an exponential manner.
        * (so the error signals vanish or blow up in a long term process, becoming vanishing large or small)
    * LSTM: solves vanishingly small and large problem
      * introduces memory cells and gates to control the long-term information in the network
      * See Picture (a bit more complicated)
  * Experiments
    * Collection and Division
    * Pre-Processing
      * uses streamflow data and precipitation data from 11 rainfall stations in the past 12 hours to
      forecast stream-flow 6 hrs into the future
      * series_to_supervised for supervised Learning
      * consult again for more details
      * 13000 pieces of info used in training
      * 6000 used in test
  * Model training
    * LSTM USING KERAS LIBRARY, A PYTHON DEP LEARNING LIBRARY
    * Amt of hidden nodes (via experimentation) = 64
    * optimizer, batch size, and epochs
      * optimizer - loss function minimization (Adam chosen)
      * Batch size - smaller batch size slows down, bigger batch size over fitting
        * set to 72
      * epochs - number of times model runs whole data
        * maximized when loss function is minimized (note will decay anti-log, and select when gains are minimized)
  * Comparative Model Selection
    * SVR: ”The idea of SVR is based on the computation of a linear regression function in a high dimensional feature space where the input data are mapped via a nonlinear function
    * MLP: "Multilayer perceptrons are a class of ANN, which the nonlinear computing elements are arranged in a feedforward layered structure"
  * Evaluation Criteria:
    * root mean square error (RMSE)
    * median absolute error (MAE)
    * coefficient of determination (R^2)
  * Results:
    LSTM > others
    LSTM = no prediction of random valleys
    LSTM still struggles with higher level flows (log appraoch?)
  * Extended Experiments
    Rainfall and flow data are valuable inputs
    Forecasts further into the future suck More
    encoder time step in range of 12-14 have best accuracy (number of hours ed into LSTM model)

  * Further Reading:
    * Yaseen et al - general Review
    * Shiri and Kisi - daily, monthly, and yearly stream-flow model
    * Lotfi Z Zadeh - neuro-fuzzy mode
    * Keras library

7. 2020. Zhang. Developing a Long Short-Term Memory (LSTM) based model for predicting
water table depth in agricultural areas

8. 2020. Fu. Deep Learning Data-Intelligence Model Based on
Adjusted Forecasting Window Scale: Application
in Daily Streamflow Simulation

* Further Reading:
  * General Approach; Jiang et al, 'Improving AI system awareness of Geoscience Knowledge: Symbiotic Integration of PHysical Approaches and Deep Learning'
  * HydroNets; Moshe et al, HydroNets: Leveraging River Structure for Hydrologic Modeling.
  * Guiding Questions; Ebeerrtt--Uphoffff, Thoughtfully Using Artificial Intelligence in Earth Science


### Summary - Further Reading:
* Further reading:
  * Cressie an Dikle 2015 - use physical background knowledge to build statistical models
  * Rudy et al (2017) make use of a sparse regression method for discovering
      the governing partial dierential equation(s) of a given system by time series measurements
      in the spatial domain.
  * Developers big data: https://developers.google.com/machine-learning/glossary/
  * Developers big data: http://www.wildml.com/deep-learning-glossary/
  * random forest method
  * Machine Learning v Deep Learning (?)
  * emulators
  * Parameter estimation - J. Carrera https://link.springer.com/chapter/10.1007/978-94-009-2301-0_15
    * Groundwater modelling involves several steps
      1. conceptualization
      2. parameter estimation
        * note that model parameters in gwater are things like transmissivity, storage coefficient, porosity, recharge, etc...
        * the model parameters must be close to their physical counterparts if the model is expected to closely produce the behavior of the real system
      3. error analysis on the parameterizations
      4. Prediction
      5. error analysis on the predictions
    * Automatic parameter estimation can help overcome trial and error 'attempts' at using a satisfactory model result
      * The most interesting part is that parameter estimation can allow the modellerto try alternative conceptual models and choose the one that best fits not only available data but also non-quantifiable information and the modeller's subjective conception of the system.
  * ANN, neurons, cells, ReLU
  * What's the difference between ANN and other spatial statistical methods?
    * earlier methods were mostly not designed with relevant architectural elements to enable automatic extraction of features. Thus their input features must be computed from raw data with formulas provided by domain experts.
    * Many methods do not scale well for large data sometimes
    * Cannot automatically extract features
  * Equifinality
  * Yaseen et al - general Review of DL in hydrosciences
  * Shiri and Kisi - daily, monthly, and yearly stream-flow model
  * Lotfi Z Zadeh - neuro-fuzzy mode
  * Keras library
  * General Approach; Jiang et al, 'Improving AI system awareness of Geoscience Knowledge: Symbiotic Integration of PHysical Approaches and Deep Learning'
  * HydroNets; Moshe et al, HydroNets: Leveraging River Structure for Hydrologic Modeling.
  * Guiding Questions; Ebeerrtt--Uphoffff, Thoughtfully Using Artificial Intelligence in Earth Science
  - PINN (Physically Informed Neural Networks)
  - DI v LSTM (Data Integration)
  - Enhancing Streamflow Forecast and Extracting Insights using Long-Short Term aMemory Networks with Data Integration at Continental Scales (Dapeng Feng, Kuai Fang, Chaopeng Shen)
  - How to use LSTM as a forward extrapolator
  - Uncertainty Estimation

  * BackPropogation: http://neuralnetworksanddeeplearning.com/chap2.html
