# Laura Meetings

## Laura Meeting 10222020:
### Opportunties:
  1.
    * use physical background knowledge to build statistical models (ie incorporate physical process method)
    * use NN to reduce complexity of numerical simulation
      * ie fluid dynamics, which can be computationally demanding
    * use sparse regression method for discovering governing partial differential equations
    * Optical Flow
    * Video Prediction

  2.
    * Integration with physical modelling:
      * (1) Improving parameterizations
      * (2) Replacing physical 'sub-model' with machine learning model
      * (3) Analysis of model-observation mismatch
      * (4) Constraining submodels
      * (5) Surrogate modelling or emulation
        * faster sensitivity analysis, model parameter calibration, derivation of confidence intervals
        * 'emulators'

  3.
    * 1 Interdisciplinarity and human dynamics (more complicated and interconnected systems)
    * 2 Data Deluge and Data Discoverability (finding and interpreting data)
    * 3 Unrecognized / Unrepresented Linkages (exploring complex correlations, such as feedabcks between flows, ecosystem dmo graphics, rooting depth, and root hydraulics)
    * 4 Model Scaling and Equifinality Challenge
      * model scaling: (governing equations like Darcy's Law and Richard's Equation don't scale well from laboratory conditions up to field conditions, and resolving dynamics at field scale is too expensive)
      * equifinality: (multiple models generate the smae outcomes)
    * 5 regionalized parameters and models: (big models need to be regionalized (redone) when applied to smaller regions in order to produce god results. However, typical approaches suffer from lack of knowledge and lack of data for calibration and parameterization)

## Laura Meeting 10/29/2020:
  1. Generic LSTM

  2. Simple Investigation in Python
    Code:
    # Questions
    # # 1) How do we use large datasets without computer processing being
    # # impossible
    # # 2) Better predictive variable than precip accum
    # # 3) Scaling data
    # # 4) Multiple Variable
    # # 5) Messing around with different inputs:
        # Time Sample (Hr, Da, Week, etc..)
        # number of nodes
        # epochs, batch_size, optimizer, etc...
    # # 6) Turn into multiple functions
    # # 7) Compare with other 'traditional' ML methods
        # Linear Regression, SVR and MLP

  3. Tools
      * Numpy
      * SciPy
      * Matpotlib
      * Scikit-learn
      * Keras
      * TensorFlow
      * PyTorch

  4. Heat Tracing 'pitch':
    Physical Environment
    Full Dataset of heads, seepage, chemistry (nitrogen, oxygen, stable isotope) temperature
    Difficulties
      - Temperature, reactive transport, unsat flow, density t

## Laura Meeting 11/05/20
Goals:
  1. Set up Git Repo - done
  2. Continue Lit Review and share readings - ongoing
  3. Doc with links and summary - ongoing
  4. Share Scripts - done
  - Continue Experimentation:
    1. How do we use large datasets without computer processing being impossible
    2. Better predictive variable than precip accum
    3. Scaling data
    4. Multiple Variable
    5. Messing around with different inputs:
       - Time Sample (Hr, Da, Week, etc..)
        - number of nodes
        - epochs, batch_size, optimizer, etc...
    6. Turn into multiple functions
    7. Compare with other 'traditional' ML methods
        - Linear Regression, SVR and MLP
  5. Think about Applications - Read Convergence Accelerator - done
  6. Followup Machine Learning Hackathon - followed up, but haven't gone into much here yet.


  * Questions / Itinerary:
    * Git Directory:
    * Class:
      * Homebrew
      * Saving graph objects for later use
    * Coding:
      * How to manage large datasets without computer processing being impossible
      * Saving graphs for later
      * Managing large data sources was difficult, and so I used
        simpler tools at the PyTorch simpler
        * Share PyTorch Script
    * Project WET:
      * Meeting next Week
      * Demoing this weekend
      * Demoing in our session next week?
    * Convergence Accelerator:
      * From integrated team to make gwater modeling / data more accessible to forecasters using ML
      * Product HydroFrame-ML - a platform for model emulators, reduced order models, etc...
      * Issues with subsurface ML all addressed within HydroFrame ML
        1. the state (groundwater storage) and physical processes (gwater flow) are not easily observed
        2. current PDE simulations and subsurface observations are not set up for ML access and Analysis
          * Need to reformat both for training
        3. complicated (spatially distributed) storage fluxes, whereas stream forecasting is predicting at a point
          * check This
        4. hydrologists and planners are not trained in ML, so transparency
      * Deliverables: include 'back-end' - Make gwater results ML-ready
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
          (See Figure)
      * See Brainstorming Research Directions
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


  * Future Work:
    * Hackathon (fup on Abe's remarks)
    * comparing results to similar 'traditional' ML methods
  * Google Drive: - ongoing
    * Doc
    * pdfs
    * jpg
  * Big Files run on GPU
    * AUHPC
      * Go to Google Doc for UA_HPC_Notes
      * Launch and interactive Python session
      * Update as you go the google doc
      * Abe could help, so could Louise
        * How do we get charged running on GPUs
        * Submit a job for 20 minutes (no more than an hr or 2)
    * Google Collab
      * What is the computing power?
  * PyTorch v TensorFlow
    * https://realpython.com/pytorch-vs-tensorflow/
  * WET meeting put on the calendar
    * https://sandtank.hydroframe.org/
  * Patricia:
    * Nonlinear embedding to figure out when streamflow is predictable and when it isn't predictable
    * Chaos Butterfly
      * Time-linear embedding putting it into phase space
      * Can ML capture transitions between hydrologic regimes?
    * ML approaches on upper colorado river
      * Contact Patricia
        * Access Repo
        * Paper from Patricia
        * Read about Upper Colorado River

## Laura Meeting 11/12/20
  * minutes / notes
    1. Google Drive: - ONGOING, finish setting up
      * Doc
      * pdfs
      * jpg
      * collab scripts
    2. Big Files run on GPU
      * AUHPC
        * Go to Google Doc for UA_HPC_Notes - COMPLETE
        * Launch and interactive Python session - COMPLETE
        * Run script as written - COMPLETE
        * Continue updating script (from last week) - ONGOING
           * different normalizing approaches - ONGOING
           * compare to Linear Regression, SVR, and MLP methods - ONGOING
        * Update as you go the google doc - COMPLETE
        * Abe could help, so could Louise
          * How do we get charged running on GPUs
          * Submit a job for 20 minutes (no more than an hr or 2)

    3. Google Collab
        * What is the computing power? - COMPLETE

    4. PyTorch v TensorFlow
      * https://realpython.com/pytorch-vs-tensorflow/
      * Hack-a-thon

    5. Patricia:
      * Nonlinear embedding to figure out when streamflow is predictable and when it isn't predictable
      * Chaos Butterfly
        * Time-linear embedding putting it into phase space
        * Can ML capture transitions between hydrologic regimes?
      * ML approaches on upper colorado river
        * Contact Patricia
          * Access Repo
          * Paper from Patricia
          * Read about Upper Colorado River

  * Action items
    1. Finish setting up Google Drive.
      * Doc
      * pdfs
      * jpg
      * collab scripts
      * Fix git issues with git pull

    2. Big files on GPUs
      * work on ML scripts in colab
      * Run simple pytorch script on HPC
      * Run complicated pytorch scripts on HPC

    3. Coding Background:
      * Hack-a-Thon (see Abe notes)
        * Google Collab tutorial
        * June 22/23 slides
      * Look into NUMBA (taking advantages of GPU for non-ML applications)
      * Take a look into Patricia's scripts
        * install R
        * What signal-processing (via wavelets) was utilized?
        * What time-lags did she use?
        * How many time-lags (dimensions) did she use?

    4. Research Background:
      * Read more about Upper Colorado
      * Finish the Non Linear Embedding R. article
      * To Research Question:
        * Different modes exist in dynamic watersheds. In an age of climate change, the dynamics are changing
        * How can we use ML to help us 'extract' not only if trends are predictable/unpredictable 'extract' when we are in one 'epoch' of predictability vs. another. And transitioning from one to another?

    5. Products:
      * Applications of ML to NLE research question?
        * ML works best on ample and noisy data .. so maybe eschewing noise removal
        * running unsupervised learning of trends in phase space... hmmm...
      * Get a working PyTorch / NUMBA gpu script working on UA_HPC
      * Plot Verde River in phase space
        * Might require arbitrary decisions about dimensions and lags
        * utilize patricia's code
      * Predictability / Unpredictability. (?)
