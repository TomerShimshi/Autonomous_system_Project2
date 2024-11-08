# Autonomous_system_Project2
This project aims to implement and analyze three algorithms, namely, the Kalman Filter, the Extended Kalman Filter, and the EKF-SLAM algorithm.
The first section of the project involves implementing the classic Kalman Filter.
The KITTI OXTS GPS trajectory from the recorded data 2011_09_26_drive_0061 will be used to extract latitude, longitude, and timestamps. The LLA coordinates will be transformed into the ENU coordinate system and Gaussian noise will be added to x and y of the ENU coordinates.
The Kalman Filter will then be implemented on the constant velocity model with noisy trajectories to approximate the ground truth trajectory.
The appropriate matrices and initial conditions will be calibrated and initialized to minimize the RMSE error and achieve a maximum error less than 7.
The covariance matrix of the state vector and dead reckoning the Kalman gain after 5 seconds will be analyzed to observe their impact on the estimated trajectory.
The x-y values and the corresponding sigma value along the trajectory will also be analyzed separately.

In the second section, the Extended Kalman Filter will be implemented using the same data and noised trajectories as in the first section. The nonlinear motion model will be dealt with while still applying the Kalman Filter on it.
The appropriate matrices will be initialized and computed, and the results will be plotted and analyzed similarly to section 1 to reduce the RMSE and maxE and observe how it deals differently with dead reckoning.

In the third section, the EKF-SLAM algorithm will be implemented. The odometry motion model with Gaussian noised inputs will be run, and assumed measurements from the state with some Gaussian noise will be used.
The predicted state of the motion will be implemented, and the correction of the state will be computed by considering the effect of each observed landmark on the Kalman gain, corrected mean, and uncertainty matrix for each time step.
The estimated localization and mapping will be fully computed using observations at the relevant time steps and motion commands.
The results will be analyzed to reach minimum RMSE and maxE values. The estimation error of X, Y, Theta, and two landmarks will also be analyzed. 
