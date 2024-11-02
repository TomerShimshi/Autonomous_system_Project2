import numpy as np
import matplotlib.pyplot as plt
import math
from utils.plot_state import plot_state
from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:
    """
    class for the implementation of Kalman filter
    """

    def __init__(self, enu_noise, times, sigma_xy, sigma_n, is_dead_reckoning):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
        """
        self.enu_noise = enu_noise
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_n = sigma_n
        self.is_dead_reckoning = is_dead_reckoning

    
    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """
        #TODO
        maxE = 0
        e_x = X_Y_GT[100:,0] - X_Y_est[100:,0]
        e_y = X_Y_GT[100:,1] - X_Y_est[100:,1]
        
        maxE = max(abs(e_x)+abs(e_y))
        RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(X_Y_GT.shape[0]-100))
        return RMSE, maxE
    
    def run(self):
        """
        Runs the Kalman filter

        outputs: enu_kf, covs
        """
        delta_t= self.times[1]-self.times[0]
        A_t = np.array([[1,delta_t,0,0],[0,1,0,0],[0,0,1,delta_t],[0,0,0,1]])
        C = np.array([[1,0,0,0],[0,0,1,0]])#np.eye(4)
        R = np.array([[0,0,0,0],[0,delta_t,0,0],[0,0,0,0],[0,0,0,delta_t]])*self.sigma_n**2#np.pow(self.sigma_n,2)
        Q = np.array([[self.sigma_xy**2,0],[0,self.sigma_xy**2]])
        #Q = np.array([[self.sigma_xy[0]**2,0],[0,self.sigma_xy[1]**2]])
        P = np.array([[ self.sigma_xy**2,0,0,0], [0,100,0,0], [0,0,self.sigma_xy**2,0],[0,0,0,100] ]) #np.eye(4)  # Initial covariance matrix
        x= [self.enu_noise[0][0],1,self.enu_noise[0][1],10]
        # Initialize variables
        enu_kf = np.zeros_like(self.enu_noise[:,:2])
        covs = np.zeros((len(self.enu_noise), 4* 4))

        # Apply Kalman filter to the data
        for i in range(1,len(self.enu_noise)):
            # Prediction
            delta_t= self.times[i]-self.times[i-1]
            A_t = np.array([[1,delta_t,0,0],[0,1,0,0],[0,0,1,delta_t],[0,0,0,1]])
            R = np.array([[0,0,0,0],[0,delta_t,0,0],[0,0,0,0],[0,0,0,delta_t]])*math.pow(self.sigma_n,2)#self.sigma_n
            x = A_t @ x
            R = np.array([[0,0,0,0],[0,delta_t,0,0],[0,0,0,0],[0,0,0,delta_t]])*np.random.normal(self.sigma_n)
            P = A_t @ P @ A_t.T + R

            
            
            #Kalman gain
            S = C @ P @ C.T + Q
            K = P @ C.T @ np.linalg.inv(S)
            if self.is_dead_reckoning and self.times[i] >= 5.0:
                K = np.zeros((4,2))
            else:
                K = P @ C.T @ np.linalg.inv(S)
            #correction
            Z = [self.enu_noise[i][0],self.enu_noise[i][1]]+ np.random.normal(0,self.sigma_n,(2))#@C + R#Q*np.random.normal(self.sigma_xy)
            x = x + K@(Z-C@x)
            P= (np.eye(4)-K@C)@P
            
            
            # Save filtered values and covariance matrices
            enu_kf[i, :] = [x[0],x[2]]#x[:2]
            covs[i, :] = np.squeeze(P).flatten() #P

            
            

        return enu_kf, covs










class ExtendedKalmanFilter:
    """
    class for the implementation of the extended Kalman filter
    """
    def __init__(self, enu_noise, yaw_vf_wz, times, sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, is_dead_reckoning =False, dead_reckoning_start_sec=5.0,sigma_n = 1.0):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            yaw_vf_wz: the yaw, forward velocity and angular change rate to be used (either non noisy or noisy, depending on the question)
            sigma_theta: sigma of the heading
            sigma_vf: sigma of the forward velocity
            sigma_wz: sigma of the angular change rate
            k: hyper parameter to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
            dead_reckoning_start_sec: from what second do we start applying dead reckoning, used for experimentation only
        """
        self.enu_noise = enu_noise
        self.yaw_vf_wz = yaw_vf_wz
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_theta = sigma_theta
        self.sigma_vf = sigma_vf
        self.sigma_wz = sigma_wz
        self.k = k
        self.is_dead_reckoning = is_dead_reckoning
        self.dead_reckoning_start_sec = dead_reckoning_start_sec
        self.sigma_n= sigma_n
        sigma_theta = sigma_theta


    #TODO
    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        #TODO
        maxE = 0
        e_x = X_Y_GT[100:,0] - X_Y_est[100:,0]
        e_y = X_Y_GT[100:,1] - X_Y_est[100:,1]
        
        maxE = max(abs(e_x)+abs(e_y))
        RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(X_Y_GT.shape[0]-100))
        print(f"RMSE: {RMSE:.2f} meters, Max Error: {maxE:.2f} meters")
        return RMSE, maxE
        

    def run(self):
        """
        Runs the extended Kalman filter
        outputs: enu_ekf, covs
        """
        
        
        delta_t = self.times[1] - self.times[0]
        x_prev = np.array([self.enu_noise[0, 0], self.enu_noise[0, 1], self.yaw_vf_wz[0, 0]])  # initial state
        P_prev = np.array([[self.sigma_xy ** 2, 0, 0],
                                   [0, (self.sigma_xy ** 2), 0], [0, 0, self.sigma_theta ** 2]]) * self.k
        Q = np.array([[self.sigma_xy ** 2, 0], [0, self.sigma_xy ** 2]])
        R = np.array([[self.sigma_vf ** 2, 0], [0, self.sigma_wz ** 2]])
        H = np.array([[1, 0, 0], [0, 1, 0]])
        enu_kf = np.zeros_like(self.enu_noise)
        enu_kf[0, :] = x_prev
        covs = np.zeros((len(self.enu_noise), 3* 3))
        covs[0] = np.squeeze(P_prev).flatten()

        for i in range(1, len(self.times)):

            delta_t = self.times[i] - self.times[i - 1]
            current_v = self.yaw_vf_wz[i, 1]
            current_w = self.yaw_vf_wz[i, 2]
            prev_tetha = np.asscalar(x_prev[2])

            z = np.array([[self.enu_noise[i, 0]], [self.enu_noise[i, 1]]])

            # predict
            ratio = current_v / current_w
            
            V = np.array([[(-1 / current_w) * np.sin(prev_tetha) + (1 / current_w) * np.sin(
                prev_tetha + current_w * delta_t), (current_v / (current_w ** 2)) * np.sin(prev_tetha) - (
                                   current_v / (current_w ** 2)) * np.sin(prev_tetha + current_w * delta_t) + (
                                   current_v / current_w) * np.cos(prev_tetha + current_w * delta_t) * delta_t],
                          [(1 / current_w) * np.cos(prev_tetha) - (1 / current_w) * np.cos(prev_tetha + current_w * delta_t),
                           (-current_v / (current_w ** 2)) * np.cos(prev_tetha) + (
                                   current_v / (current_w ** 2)) * np.cos(prev_tetha + current_w * delta_t) + (
                                   current_v / current_w) * np.sin(prev_tetha + current_w * delta_t) * delta_t],
                          [0, delta_t]])
            G = np.array([[1, 0, -ratio * np.cos(prev_tetha) + ratio * np.cos(prev_tetha + current_w * delta_t)],
                          [0, 1, -ratio * np.sin(prev_tetha) + ratio * np.sin(prev_tetha + current_w * delta_t)],
                          [0, 0, 1]])
            #-----------------------------------------------------------------
            #   predict the new sigma and location
            #-----------------------------------------------------------------
            R_n = np.array([[1,0,0],[0,1,0],[0,0,2]])* self.sigma_n
            P = G @ P_prev @ G.T + V @ R @ V.T +np.eye(P_prev.shape[0]) * self.sigma_n
            x = x_prev.reshape(3, 1) + np.array(
                [-ratio * np.sin(prev_tetha) + ratio * np.sin(prev_tetha + current_w * delta_t),
                 ratio * np.cos(prev_tetha) - ratio * np.cos(prev_tetha + current_w * delta_t),
                 current_w * delta_t]).reshape(3, 1)
            
            #-----------------------------------------------------------------
            #   calculate the Kalman Gain
            #-----------------------------------------------------------------
            #Kalman gain
            if self.is_dead_reckoning and self.times[i] >= self.dead_reckoning_start_sec:
                K = np.zeros((3,2))
            else:
                S = H @ P @ H.T + Q
                K =  P @ H.T @ np.linalg.inv(S)
            
            #-----------------------------------------------------------------
            #   Apply Kalman correction
            #-----------------------------------------------------------------
            x = x.reshape((3, 1)) + K @ (z - (H @ x).reshape((2, -1)))
            P = (np.eye(P.shape[0]) - K @ H) @ P
            
            

            enu_kf[i, :] = x.reshape((3,))
            covs[i] = np.squeeze(P_prev).flatten()

            x_prev = x
            P_prev = P

        return enu_kf, covs
       
       
    


class ExtendedKalmanFilterSLAM:
    
    
    def __init__(self, sigma_x_y_theta, variance_r1_t_r2, variance_r_phi,sigma_n):

        """
        Args:
            variance_x_y_theta: variance in x, y and theta respectively
            variance_r1_t_r2: variance in rotation1, translation and rotation2 respectively
            variance_r_phi: variance in the range and bearing
        """
                
        self.sigma_x_y_theta = sigma_x_y_theta#TODO
        self.variance_r_phi = variance_r_phi#TODO
        self.R_x = np.array([[variance_r1_t_r2[0]**2, 0, 0], [0,variance_r1_t_r2[1]**2,0], [0, 0, variance_r1_t_r2[2]**2]])#TODO
        self.sigma_n =sigma_n
    
    def predict(self, mu_prev, sigma_prev, u, N):
        # Perform the prediction step of the EKF
        # u[0]=translation, u[1]=rotation1, u[2]=rotation2

        delta_trans, delta_rot1, delta_rot2 = u['t'], u['r1'], u['r2']#TODO
        theta_prev = normalize_angle(mu_prev[2])#TODO
        
        F = np.hstack((np.identity(3),np.zeros((3,2*N)))) #TODO
        G_x = np.identity(3) + np.array([[0, 0, -delta_trans*np.sin(theta_prev + delta_rot1)], [0, 0, delta_trans*np.cos(theta_prev + delta_rot1)], [0,0,0]]) #TODO
        G = np.vstack((np.hstack((G_x,np.zeros((3,2*N)))) , np.hstack((np.zeros((2*N,3)),np.identity(2*N)))))#TODO
        V = np.array([[-delta_trans*np.sin(theta_prev + delta_rot1), np.cos(theta_prev + delta_rot1), 0], [delta_trans*np.cos(theta_prev + delta_rot1), np.sin(theta_prev + delta_rot1),0], [1,0,1]])#TODO
        #R_hat_x = np.dot(V,np.dot(self.R_x,V.T)) + np.eye(self.R_x.shape[0]) * self.sigma_n#0.001
        R_hat_x = np.dot(V,np.dot(self.R_x,V.T)) + np.array([[0,0,0],[0,0,0],[0,0,1]]) * self.sigma_n
        mu_est = mu_prev + np.dot(F.T, np.array([ delta_trans * np.cos(theta_prev + delta_rot1), delta_trans * np.sin(theta_prev + delta_rot1) , normalize_angle(delta_rot1 + delta_rot2)]).T)#TODO
        sigma_est = np.dot(G, np.dot(sigma_prev, G.T)) + np.vstack((np.hstack((R_hat_x, np.zeros((3,2*N)))), np.zeros((2*N,2*N+3)))) #TODO
        
        return mu_est, sigma_est
    
    def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
        # Perform filter update (correction) for each odometry-observation pair read from the data file.
        mu = mu_pred.copy()
        sigma = sigma_pred.copy()
        theta = mu[2]
        
        m = len(z["id"])
        Z = np.zeros(2 * m)
        z_hat = np.zeros(2 * m)
        H = None
        
        for idx in range(m):
            j = z["id"][idx] - 1
            r = z["range"][idx]
            phi = z["bearing"][idx]
            
            mu_j_x_idx = 3 + j*2
            mu_j_y_idx = 4 + j*2
            Z_j_x_idx = idx*2
            Z_j_y_idx = 1 + idx*2
            
            if observed_landmarks[j] == False:
                mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
                observed_landmarks[j] = True
                
            Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])
            
            delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
            q = delta.T@delta#delta.dot(delta)
            z_hat[Z_j_x_idx : Z_j_y_idx + 1] = np.array([np.sqrt(q),  normalize_angle(np.arctan2(delta[1],delta[0]) - theta)]) #TODO
            
            I = np.diag(5*[1])
            F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))
            
            Hi = np.dot([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],[delta[1], -delta[0], -q, -delta[1], delta[0]]], F_j)/q#TODO

            if H is None:
                H = Hi.copy()
            else:
                H = np.vstack((H, Hi))
        
        Q = np.zeros((H.shape[0],H.shape[0]))#TODO
        np.fill_diagonal(Q, [np.power(self.variance_r_phi[0],2),np.power(self.variance_r_phi[1],2)] )
        
        S = np.linalg.inv(np.dot(H,np.dot(sigma_pred,H.T)) + Q) #TODO#TODO
        K = np.dot(sigma_pred,np.dot(H.T,S))#TODO
        
        diff =Z - z_hat#TODO
        diff[1::2] = normalize_angles_array(diff[1::2])
        
        mu = mu + K.dot(diff)
        sigma = np.dot((np.identity(2*N+3) - np.dot(K,H)),sigma_pred)#TODO
        
        mu[2] = normalize_angle(mu[2])

        # Remember to normalize the bearings after subtracting!
        # (hint: use the normalize_all_bearings function available in tools)

        # Finish the correction step by computing the new mu and sigma.
        # Normalize theta in the robot pose.

        
        return mu, sigma, observed_landmarks
    
    def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
        # Get the number of landmarks in the map
        N = len(landmarks)
        
        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution


        init_inf_val = 100 #TODO
        
        #mu_arr =[np.hstack(([0.096,0.0101,0.01009],np.zeros(2*N))).T] #TODO
        mu_arr =[np.hstack(([0,0,0],np.zeros(2*N))).T] #TODO
        sigma_prev = np.vstack((np.hstack(([[np.power(self.sigma_x_y_theta[0],2), 0, 0], [0, np.power(self.sigma_x_y_theta[1],2), 0], [0, 0, np.power(self.sigma_x_y_theta[2],2)]],np.zeros((3,2*N)))), np.hstack((np.zeros((2*N,3)),init_inf_val*np.identity(2*N)))))#TODO

        # sigma for analysis graph sigma_x_y_t + select 2 landmarks
        landmark1_ind=3#TODO
        landmark2_ind=4#TODO

        Index=[0,1,2,landmark1_ind,landmark1_ind+1,landmark2_ind,landmark2_ind+1]
        sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index,Index].copy()
        
        observed_landmarks = np.zeros(N, dtype=bool)
        
        sensor_data_count = int(len(sensor_data_noised) / 2)
        frames = []
        
        mu_arr_gt = np.array([[0, 0, 0]])
        
        for idx in range(sensor_data_count):
            mu_prev = mu_arr[-1]
            if idx %99 ==0:
                tt=1
            u = sensor_data_noised[(idx, "odometry")]
            # predict
            mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)
            # update (correct)
            mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)
            
            mu_arr = np.vstack((mu_arr, mu))
            sigma_prev = sigma.copy()
            sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index,Index].copy()))
            
            delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
            delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
            delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]

            calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
            calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)

            theta = delta_r1_gt + delta_r2_gt

            theta_prev = mu_arr_gt[-1,2]
            mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))
            
            frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])
            
            frames.append(frame)
        
        return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2
    
    