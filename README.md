# LSTM Nudging scheme for data assimilation of geophysical flows

**Abstarct:**

Reduced rank nonlinear filters are increasingly utilized in data assimilation of geophysical flows, but often require a set of ensemble forward simulations to estimate forecast covariance. On the other hand, predictor-corrector type nudging approaches are still attractive due to their simplicity of implementation when more complex methods need to be avoided. However, optimal estimate of nudging gain matrix might be cumbersome. In this paper, we put forth a fully nonintrusive recurrent neural network approach based on a long short-term memory (LSTM) embedding architecture to estimate the nudging term, which plays a role not only to force the state trajectories to the observations but also acts as a stabilizer. Furthermore, our approach relies on the power of archival data and the trained model can be retrained effectively due to power of transfer learning in any neural network applications. In order to verify the feasibility of the proposed approach, we perform twin experiments using Lorenz 96 system. Our results demonstrate that the proposed LSTM nudging approach yields more accurate estimates than both extended Kalman filter (EKF) and ensemble Kalman filter (EnKF) when only sparse observations are available. With the availability of emerging AI-friendly and modular hardware technologies and heterogeneous computing platforms, we articulate that our simplistic nudging framework turns out to be computationally more efficient than either the EKF or EnKF approaches. 


**LSTM Nudging framework:**              

<img src="https://github.com/surajp92/LSTM_Nudging/blob/master/da_lstm_framework.png" width="700" height="450" >
