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

9. General Approach; Jiang et al, 'Improving AI system awareness of Geoscience Knowledge: Symbiotic Integration of PHysical Approaches and Deep Learning'
  * A nice overview of how to apply AI in simulating and predicting run-off
  * provides a framework to apply a specially structured design for AI to memorize physical rules behind system dynamics
  * Shows the model can be applied in other contexts

10. HydroNets; Moshe et al, HydroNets: Leveraging River Structure for Hydrologic Modeling.
  * Using basin structure to simulate and predict streamflow

11. Guiding Questions; Ebeerrtt--Uphoffff, Thoughtfully Using Artificial Intelligence in Earth Science
  * An overview of AI in earth science

12. 2020. Feng. Enhancing streamflow forecast and extracting insights using long-short term
memory networks with data integration at continental scales
  * Recommended by Chaopeng Shen
  * Abstract:
    * Integrated recent discharge observations to improve streamflow forecasts.
    * Procedure called 'data integration', which uses a CNN (convolutional network) to add various types of observations together.
    * Integrating moving average discharge, discharge form the last few days, or even average discharge from previous month
      improves predictions of daily forecasts
    * DI improves LSTM in all domains acept for super-arid flashy flow, especially in groundwater-dominated or surface water storage areas

  * Intro:
    * DA (data assimilation), a previous similar approach that uses recent observations to update process model internal states, to better forecast future variables and/or update model structures or parameters.
    * Although DL has been used in some time-series applications, there is significant potential.
    * Specifically, ability to flexibly absorb recent observations is unexplored
    * In general, the daily forecast prediction approach might best be served by traditional regression models
    * The question is how to incorporate other types of data, like snapshots and moving average, that may represent important facets of system state but are not integrated into the larger process.
    * This work uses an LSTM method to assimilate various types of discharge observations via DI (data inegration)
      * DI - integrates the data injection and prediction steps into one step, altering the internal state of the LSTM and improving the prediction of future prediction variables (but does not predict unobserved variables)
      * uses snapshot, moving average, and regularly spaced stream flow observations in LSTM

  * LSTM model with Data Integration:
    * We implemented a fast and flexible LSTM code that can utilize the highly-optimized NVIDIA CUDA® Deep Neural Network (cuDNN) library from the PyTorch deep learning platform.
    * To reduce overfitting, 𝓓 applies constant dropout masks
(Boolean matrices to set some weights to 0) to the recurrent connections following Gal & Ghahramani
(2015). We did not apply dropout to 𝒈 in equation 6 as we did in previous work (Fang et al., 2018), as
customization at this level in the code is not well supported by cuDNN.

  * Further Reading:
    * Streamflow Prediction:  (Hu et al., 2018; Kratzert et al., 2018; Kratzert, Klotz, et al., 2019; Le et al., 2019; Sudriani et al., 2019)

13. Convergence Accelerator Proposal:
  * Intro:
    * integrated team to make groundwater data accessible and improve hydrologic forecasting with ML
    * HydroFrame-ML will be a platform to share model emulators, reduced order models, data products, and ML accelerated PDE solutions (or ML-models)
    * upshot - help gw models fill gaps in groundater data
    * collaborators: federal water infrastructure managers, climate and earth system modelers
    * issues: these collaborators don't have a 'computationally feasible' way to incorporate into workflows

  * why gwater? groundwater is critical to predicting responses of global extremes

  * gwater is usually excluded or simplified in global simulations due to data and modeling limitations (see fig)
    * even with advances in PDE-based approaches, this is too complicated to be utilized in conjunction with other efforts

  * ML assistance with PDE-simulations and observations can help, but there are issues with subsurface usage of ML
    1. the state (groundwater storage) and physical processes (gwater flow) are not easily observed
    2. current PDE simulations and subsurface observations are not set up for ML access and Analysis
      * Need to reformat both for training
    3. complicated (spatially distributed) storage fluxes, whereas stream forecasting is predicting at a point
      * check This
    4. hydrologists and planners are not trained in ML, so transparency

  * Solution: HydroFrame ML - a kind of 'clearning house' of data, model, and ML
    * data-centric, model-centric, and user-centric features
    * improve hydrologic forecasting by addressing blind spot in current approaches

  * Deliverables:
    1. National data platform (FAIR), Data Mining, API (back-end)
      * An API will drive data mining of simulation data that can be 'ingested' into ML routines
    2. User Portal for building, evaluating, sharing ML-groundwater models: (front-end)
      * training and model interpretation to get around 'black-box' iness of solutions for users
        * MLFlow
    3. Improved Hydrologic Forecasts (applied solutions)
      * Make beter predictions
    4. Educational Activities (broadening participation)
      * STEM outreach


  * More on Back-end Development:
    * Making Groundwater DAta FAIR and ML ready
    * Areas:
      1. Facilitating efficient access to big data from multiple access points
        * custom C++ application needed (not python) to access data
        * something about a new server with ML applications
      2. developing FAIR metadata standards and protocols for gwater data
        * CF-compliant metadata standards and APIs to interface with existing libraries (Numpy, SciKit, Pangeo, Xarray)
        * What is CF?
      3. providing standardized access to existing groundwater and hydrogeology observations that are distributed and heterogeneous (data discoverability)
        * workflows for sharing and make datasets discoverable
        * Dockers, reanalyzed, data store, etc..

  * Tasks:
    * What if we think bigger, and use ML to fill data-sparse domains?
    * Change the data for implementation in ML data training
    * I'm not sure I get the streamflow v spatially distributed dichotomy. ML can definitely handle multi-dimensional learning, and in fact is great at doing things with photos and with videos, for e.g.
    * Could be a part of STEM outreach!
      * I feel like finding new collaborators should be a part of your outreach objective. Build a field, will they come?
    * From Table (1):
      * review datasets and developing FAIR metadata standards
      * Developing flexible, interoperable, itelligent APIs, linking data to existing ML frameworks
      * Building and testing infrastructure for efficient access to large datasets
      * tiered staged data access and archiving

  * Further Questions:
    * What do you mean by model emulator? reduced order model? and ML- Accelerated PDE?
    * CyVerse?
    * API design, cool. But the way in which the API mines the groundwater simulation data and puts it into ML routines
      is enormously confusing
    * What is ML Flow?
    * What means federating data?
    * Which deliverable uses the 'emulators'
    * I feel like the opportunities for me (as a hydrologiest) are probably within the 'backend' development
    * Read links [20-22]
    * What exactly are the data being trained on Like are these models of multiple dimensions and directions?
    * What does CF mean?


  * Further Reading:
    * Sahoo, S., et al., Machine learning algorithms for modeling groundwater level changes in agricultural regions of the U.S. Water Resources Research, 2017. 53(5): p. 3878-3895.
    * Sun, A.Y., et al., Combining Physically Based Modeling and Deep Learning for Fusing GRACE Satellite Data: Can We Learn From Mismatch? Water Resources Research, 2019. 55(2): p. 1179-1195.
    * Amaranto, A., et al., A Spatially Enhanced Data-Driven Multimodel to Improve Semiseasonal Groundwater Forecasts in the High Plains Aquifer, USA. Water Resources Research, 2019. 55(7): p. 5941-5961.


14. Convolutional Networks for Visual Recognition:
  * https://cs231n.github.io/optimization-2/
  * Summary: Developing an intuitive understanding of backpropogation, which is a way of computing gradients of expressive through recursive application of the chain rule
    * a rarely developed view of backpropagation as backward flow in real-valued circuits
    * Notice that backpropagation is a beautifully local process. Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2. the local gradient of its output with respect to its inputs. Notice that the gates can do this completely independently without being aware of any of the details of the full circuit that they are embedded in. However, once the forward pass is over, during backpropagation the gate will eventually learn about the gradient of its output value on the final output of the entire circuit. Chain rule says that the gate should take that gradient and multiply it into every gradient it normally computes for all of its inputs.
    * Backpropagation can thus be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher.

15. Balaji Rajagopalan: A Nonlinear Dynamical Systems‐Based Modeling Approach for Stochastic Simulation of Streamflow and Understanding Predictability
  * Key Points:
    * The dynamics of multidecadal streamflow signal from long paleo and observed record uncovered by reocnstructing the phase space
    * Local Lyapunov Exponents are used to understand temporal variability of predictability potentially enabling predictabiility-based management
    * Streamflow simulated by block resampling of trajectories from neighbors in phase space, with skills consistent with Predictability
      * NOTE: This last thought shows: Forecasting changes of state (streamflow) as a consequence of historical state evolution (i.e. the trajectories of streamflow evolution) can be better than simple regression techniques, and can show us when a state evolves in a predictable way, and when it doesn't.

  * Abstract:
    * Novel modeling approach applied to historical (measured) and paleo (Reconstructed) streamflows in the Colorado River
    * Insight 1: Flows had 'epochal' variations in predictability
    * Insight 2: The Local Lyaponov Exponent quantifies the variance of flow signal and climate indices to predict flow during these different epochs
    * Insight 3: 'Blind' flow projections during 'high-predictable' epochs were good, and flow projections during 'low-predictable' epochs were poor
    * Conclusion: Assessing the skill of this modeling technique could shift water management paradigms
    * other jibber jabber:
      * novel 'time-series modeling approach' is borrowed from those of 'nonlinear dynamical systems'.
      * the goal is to use these 'non-linear' assumptions to understand when river flow is predictability (and also their 'dynamics'), so as to produce good projections.
      * Understanding when flow is predictable (time-varying predictability) comes from understanding the divergence of trajectories in phase space with time. This is done using 'local lyaunov exponents'.

  * Introduction
    * Dichotomies in Models:
      1. Parametric / linear models = traditional for time series (e.g autoregressive)
        * simulations reproduce distributional statistics like mean, standard deviation, and correlations
        * DO NOT reproduce non-gaussian and non-stationary features
      2. Non parametric models = nontraditional timeseries models (e.g. K-nearest neighbor (K-NN) bootstrap, and kernal density)
        * simulations reproduce non-gaussian Features
        * DO NOT model low-frequency variability
      3. An alternative to both is to use 'wavelet spectra' as a way to model
        a) dominant periodicities and
        b) capture nonstationarity.
     * A nonlinear dynamical system-based time series modeling approach:
      * reconstruct underlying dynamics (referred to as 'phase space') for prediction and simulation
      * within the phase space, the dynamics evolve
      * The phase space is constructed from observed time series (long time series needed)
      * The state of the system at any point is mapped on to the phase space, and using local maps short term forecasts can be made.
      * The skill of the forecast depends on how predictable system is according to the phase state.
        * The 'predictability' is ascertained through Local Lyapunov Exponents (LLEs)
      * Forecasts can outperform traditional approaches
    * Previous applications of nonlinear appraoches
      * salt lake WLs due to El Nino. Easy due to noise-less time serires
      * for noisy time series, use noise reduction, ex wavelet analysis
        * In this paper, a blend of K-NN block simulation through embedding of system recovered from wavelet reconstructed signal of time series
          * requires time delay (sigma) and embedding dimension (D)
          * see methods
      * identify time series epochs through LEE

  * Questions:
    * What is phase space?
      * https://en.wikipedia.org/wiki/Phase_space
        * a space in which all posible states of a system are represented, with each possible state corresponding to one unique point in the phase space.
        * every degree of freedom (parameter) of a system is represented on an axis of a dimensional space.
          * 1-D system = phase line (2 axis graph), 2-D system - phase plane (3-axis graph), etc...
          * for every state of the system (allowed combo of values of paarameters), there is a point on the phase space
          * the systems evolution over time traces a path (phase space trajectory)
            * trajectory = set of states compatible with a particular initial condition
          * complex systems can have 'any number of parameters'
    * What are Local Lyapunov Exponents, and how do they tell us if flow is predictable or not?
      * https://en.wikipedia.org/wiki/Lyapunov_exponent#:~:text=also%20been%20explored.-,Local%20Lyapunov%20exponent,x0%20in%20phase%20space.&text=These%20eigenvalues%20are%20also%20called%20local%20Lyapunov%20exponents.
        * The Lyapunov exponent (LE) of a dynamic (changing) system characterizes the 'rate of separation of infinitesimally close trajectories' [as defined in phasespace]
        * The rate of separation can be different for different orientations of the separation vector.
          * So there is a 'spectrum' of Lyaunov exponents (= dimensions (parameters) of phase space)
          * Conventions: We often refer to the Maximum (MLE) because it is most deterministic.
            * A positive MLE = chaotic, unpredictable system
            * < 0, or small suggests high predictability
        * In a way, we can think of this component as representing the predictability of a system.
        * The Local Lyapunov Exponent (LLE)
          * The global exponent gives a measure of the total predictability of the system.
            * But the LLE estimates the predictability around a point in phase space.
              * Done with eigenvalues in the Jacobian matrix
    * What is block resampling?
    * What are 'trajectories' from 'neighbors' in 'phase space'?
      * https://en.wikipedia.org/wiki/Phase_space
        * from above ^ we can assume trajectories to be the path of evolution of the phase space
        * neighbors may be the collection of 'states' (points) in phase space most similar to the observed state from which a prediction needs to be made
    * What is a wavelet spectral analysis?
      * Summary: wavelet analysis is a method for reducing noise. This is to smooth the Co River to obtain the signal present in the flow series.f
      * https://www.sciencedirect.com/science/article/abs/pii/S0165993614001757#:~:text=Wavelet%20Transform%20(WT)%20is%20one,to%20its%20scale%20%5B1%5D.
        * Wavelet transforms CAN BE for:
          * noise removal and resolution enhancement.
          * data compression and chemometrics modeling.
          * outperform traditional signal-processing methods.
        * wavelets are a topic of pure math, but have shown great promise in signal-Processing
          * unprecendented success in asymptotic optimality, spatial adaptivity, and computational efficiency.
      * Nowak et al, 2011 - https://www.sciencedirect.com/science/article/abs/pii/S002216941100607X?via%3Dihub 'Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary spectra'
        * abstract: using wavelet spectra to capture nonstationarity
          * Geophysical data (like streamflow) show 'quasi-periodic' and 'non-stationary' variability drive by climate features
          * traditional stochastic methods fail to capture trends due to this non-stationarity so...
          * we need methods that can account for time evolution of system
          * WARM (Wavelet-based Auto Regression Modeling)
            * i) reconstruction of wavelet transform of a timeseries into periods
            * ii) time-varying 'power' and component scaling via SAWP (scale averaged wavelet power)
              * ^NOTE: So is this like a data transform similar to 'normalizing' or 'scaling' data?
              * An improvement upon Kwon et al
            * iii) AR model fit to scaled components
            * iv) simulations (training) using AR model, rescale, and recombine to simulate original time series.
        * introduction:
          * interesting, the impetus for this research is not prediction, but producing synthetic streamflow traces in history.
          * traditional (stochastic) methods have failed to capture quasi-periodic climate forcings (like El Nino) and the presence of wet/dry epochs
          * Parametric and non-parametric (auto regressive moving average, K-nearest neighbor) can show historical dependences, but are poor at capturing 'data spectrum' and statistics from wet v dry periods.
          * The inability to reproduce spectral properties can yield great harm to predictions
          * Traditional WARM reproduces distributional data (mean, variance, skew, pdf) as well as the 'global spectrum of observed data'
            * But traditional WARM cannot capture 'non-stationary' spectral characteristics
            * New WARM (this Paper) can do non-stationary and multiple sites
      * Kwon et al 2007
        * using wavelet spectra to model dominant periodicities
          * See Nowak et al for more comprehensive look at this idea

    * What is reconstructions across bands to obtain signal time series?
    * what are 'spectral characteristics'?
      * I feel like this means 'the wide spectrum of characteristics that can be seen in a dynamic, non-stationary system'
    * What is D-dimensional space?
    * What are sigma lags?
    * What are ensembles of projections?
    * What is 'the current vector in phase space'?
    * What are stochastic processes?
      * https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322
      * A real stochastic process is a family of real random variables 𝑿={xᵢ(ω); i∈T}, all defined on the same probability space (Ω, F, P). (conventional statistics)
    * Stationary, Non-Stationary, periodic, quasi-periodic, gaussian, non-guassian processes
      * Refs:
        * https://www.investopedia.com/articles/trading/07/stationary.asp#:~:text=Data%20points%20are%20often%20non,cannot%20be%20modeled%20or%20forecasted.
        * https://www.quora.com/What-is-non-stationary-data
        * https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322
        * https://towardsdatascience.com/detecting-stationarity-in-time-series-data-d29e0a21e638
        * https://otexts.com/fpp2/stationarity.html
        * https://towardsdatascience.com/why-data-scientists-love-gaussian-6e7a7b726859
        * https://www.quora.com/What-is-an-example-of-a-dataset-with-a-non-Gaussian-distribtion
        * Quasiperiodicity
      * Stationary v Non-Stationary Data
        * Stationary data:
          * **In the most intuitive sense, stationarity means that the statistical properties of a process generating a time series do not change over time.**
            * This does not mean that the system doesn't change over time, but that the way it changes doesn't change. (2nd derivative)
          * Data dominated by processes that do not change through time. means, variances, and covariances are not a function of time
        * Non-Stationary Data:
          * data that have means, variances, and covariances that change over time.
            Ex: trends, cycles, or random walks
          * heuristics for evaluation non-stationarity:
            * Seasonality
            * Trends and changing levels
            * increasing variance
        * NOTE 1: typically we have sought to turn non-stationary time series into stationary data in order to make meaningful forecasts.
          * 'The convolution of a time variate [non-stationarity] will make your information [model results] less decipherable. However, if you can remove and isolate it [the effect of the time-variation], you can predict with great accuracy by making it stationary and de-stationary..ing it after the predictions.'
        * NOTE 2: Some cases of stationarity and non-stationarity can be confusing
          * A time series with cyclic behavior (but with no trend or seasonality) is stationary (like a pure 'sine' curve). This is because cycles are not of fixed length, so before we observe the series we cannot be sure where the peaks and troughs of the cycles will be.
      * Periodic v quasi periodic Data:
        * Periodic Data:
          * recurring at regular intervals (i.e. every 24 hrs)
        * Quasiperiodic data:
          * Quasiperiodicity is the property of a system that displays irregular periodicity.
          * a pattern of recurrence with a component of unpredictability that does not lend itself to precise measurement
          * This is a commen term in climatology, that can be used to describe El Nino, for example
      * Gaussian v Non-Gaussian Processes:
        * Gaussian Distribution is just a distribution characterized by a bell curve. See also: normal distribution
        * Anything that doesn't have a normal distribution is non-gaussian
    * What are dominant quasiperiodic bands?
      * Nowak et al:
        * quasiperiodic 'forcings' include climate and wet/dry epochs. Cycles that aren't apparent in the flow data itself, or without greater context.
        * See also periodicities, and non-stationary features
    * What do we mean by maps?

  * Actions:
    * Plot streamflow using a phase type approach (Can we plot scatter plot as a line plot?)

  * Further Reading:
    * Rajagopalan, B., & Lall, U. (1999). A nearest neighbor bootstrap resampling scheme for resampling daily precipitation and other weather variables. Water Resources Research, 35(10), 3089–3101. https://doi.org/10.1029/1999WR900028
      * A review of parametric and nonparametric methods for hydrologic time series models
    * Nowak, K., Rajagopalan, B., & Zagona, E. (2011). Wavelet Auto‐Regressive Method (WARM) for multi‐site streamflow simulation of data with non‐stationary spectra. - COMPLETE
      * using wavelet spectra to capture nonstationarity
      * Kwon et al 2007 - COMPLETE
        * using wavelet spectra to model dominant periodicities
    * Takens 1981
      * A reconstructed phase space with appropriate dimensinos and time delay is a proxy for the true space within which the unknown dynamics of the system unforlds.

16. A great reference for statistics: https://otexts.com/fpp2/stationarity.html

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
  * An excellent Discussion on ML: https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network
