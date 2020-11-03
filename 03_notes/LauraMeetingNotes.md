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
