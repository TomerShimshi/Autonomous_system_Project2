import os
import numpy as np
from data_preparation import *
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation
import graphs
import random

from kalman_filter import KalmanFilter, ExtendedKalmanFilter , ExtendedKalmanFilterSLAM
import graphs
import random
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation


random.seed(10)#TODO)
np.random.seed(10)#TODO)

font = {'size': 20}
plt.rc('font', **font)
basedir = 'Results'

class ProjectQuestions:
    def __init__(self, dataset):
        """
        Given a Loaded Kitti data set with the following ground truth values: tti dataset and adds noise to GT-gps values
        - lat: latitude [deg]
        - lon: longitude [deg]
        - yaw: heading [rad]
        - vf: forward velocity parallel to earth-surface [m/s]
        - wz: angular rate around z axis [rad/s]
        Builds the following np arrays:
        - enu - lla converted to enu data
        - times - for each frame, how much time has elapsed from the previous frame
        - yaw_vf_wz - yaw, forward velocity and angular change rate
        - enu_noise - enu with Gaussian noise (sigma_xy=3 meters)
        - yaw_vf_wz_noise - yaw_vf_wz with Gaussian noise in vf (sigma 2.0) and wz (sigma 0.2)
        """
        self.dataset = dataset
        self.LLA_GPS_trajectory = build_LLA_GPS_trajectory(self.dataset)
        self.enu, self.times, self.yaw_vf_wz = build_GPS_trajectory(self.dataset)#TODO (hint- use build_GPS_trajectory)

        # add noise to the trajectory
        self.sigma_xy = 3.0 #TODO
        self.sigma_vf = 2.0#TODO
        self.sigma_wz = 0.2#TODO

        #self.enu_noise = np.random.normal(0, self.sigma_xy, self.enu.shape)#TODO
        #self.enu_noise = [sample[:2]+np.random.normal(0, self.sigma_xy, 2) for sample in self.enu ]#np.random.normal(0, self.sigma_xy, self.enu.shape)
        self.enu_noise = self.enu + np.random.normal(0, self.sigma_xy, self.enu.shape)
        #self.yaw_vf_wz_noise=np.random.normal(0, self.sigma_vf, self.enu.shape)#TODO
        self.yaw_vf_wz_noise = self.yaw_vf_wz.copy()
        self.yaw_vf_wz_noise[:, 1] = self.yaw_vf_wz_noise[:, 1] + np.random.normal(0, self.sigma_vf, self.yaw_vf_wz.shape[0])
        self.yaw_vf_wz_noise[:, 2] = self.yaw_vf_wz_noise[:, 2] + np.random.normal(0, self.sigma_wz, self.yaw_vf_wz.shape[0])
    
    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, and apply a Kalman filter to the noisy data (enu).
        """
	
	    # "TODO":	        

        random.seed(10)#TODO)
        np.random.seed(10)#TODO)
        sigma_x_y=  self.sigma_xy
        # Convert GPS trajectory to ENU
        #locations_GT = self.enu#build_GPS_trajectory(self.dataset)
        temp = self.LLA_GPS_trajectory.copy()
        self.LLA_GPS_trajectory[:, 0],self.LLA_GPS_trajectory[:,1] = temp[:,1],temp[:,0]
        graphs.plot_single_graph(self.LLA_GPS_trajectory,'World coordinate (LLA)','lon','lat','World coordinate (LLA)')
        graphs.plot_single_graph(self.enu,'Local coordinate (ENU)','X [m]','Y [m]','Local coordinate (ENU)')
        # plot on the same graph original GT and observations noise.
        graphs.plot_graph_and_scatter(self.enu,self.enu_noise,'original GT and observations noise','X [m]','Y [m]','ENU_GT', 'ENU_noised',)
        locations_GT = self.enu
        
        #first we want to find the best sigma n value:
        maxE_list = []
        RMSE_list = []
        num_of_tests = 20
        for i in range(1,num_of_tests):
            
            sigma_n = i
            kf = KalmanFilter(sigma_xy=sigma_x_y,enu_noise=self.enu_noise,times=self.times,sigma_n=sigma_n,is_dead_reckoning=False)
            enu_kf, covs = kf.run()
                 
            RMSE, maxE = kf.calc_RMSE_maxE(locations_GT,enu_kf)
            maxE_list.append(maxE)
            RMSE_list.append(RMSE)
        
        maxE_array = np.array(maxE_list, ndmin = 2).T
        RMSE_array = np.array(RMSE_list, ndmin = 2).T
        sigma_n_array = np.array([i for i in range(1,num_of_tests,1)], ndmin = 2).T
        maxE_array_sigma_n = np.concatenate((sigma_n_array,maxE_array),axis=1)
        RMSE_array_sigma_n = np.concatenate((sigma_n_array, RMSE_array),axis=1)
        graphs.plot_single_graph(maxE_array_sigma_n, 'maxE vs sigma_n', 'sigma_n','maxE', 'maxE vs sigma_n')
        graphs.plot_single_graph(RMSE_array_sigma_n,'RMSE vs sigma_n','sigma_n','RMSE', 'RMSE vs sigma_n')
       
        sigma_n = np.argmin(maxE_list)+1
        # Apply Kalman filter to the ENU data
        kf = KalmanFilter(sigma_xy=sigma_x_y,enu_noise=self.enu_noise,times=self.times,sigma_n=2.0,is_dead_reckoning=False)
        enu_kf, covs = kf.run()

        # Calculate RMSE and max error between ground truth and filtered trajectory
        rmse, maxE = kf.calc_RMSE_maxE(locations_GT, enu_kf)
        print(f"RMSE: {rmse:.2f} meters, Max Error: {maxE:.2f} meters")
        
        # build_ENU_from_GPS_trajectory
        #graphs.plot_three_graphs( self.enu, enu_kf,self.enu_noise, 'KF results' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        graphs.plot_two_graphs_and_scatter( self.enu, enu_kf,self.enu_noise, 'KF results' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        kf_dmr = KalmanFilter(sigma_xy=sigma_x_y,enu_noise=self.enu_noise,times=self.times,sigma_n=2.0,is_dead_reckoning=True)
        enu_kf_dmr, covs_dmr = kf_dmr.run()

        # Calculate RMSE and max error between ground truth and filtered trajectory
        rmse, maxE = kf.calc_RMSE_maxE(locations_GT, enu_kf_dmr)
        print(f"RMSE: {rmse:.2f} meters, Max Error: {maxE:.2f} meters")
        # Build animation of the estimated path with error ellipses
         # build_animation (hint- graphs.build_animation)

        # animation that shows the covariances of the EKF estimated path

        # this animation that shows the covariances of the dead reckoning estimated path
        X_XY_XY_Y_uncertinty_cov_list = covs[:,[0,2,8,10]]
        
       
        graphs.plot_trajectory_comparison_dead_reckoning(self.enu,enu_kf,enu_kf_dmr)
       
        
        
        
        
        # Plot and analyze the estimated x-y values separately and corresponded
        # sigma value along the trajectory. (e.g. show in same graph xestimated-xGT and
        # 𝜎𝑥 values and explain your results):
        X_Y_estimated_minus_X_Y_GT = enu_kf - locations_GT[:,:2]
        times_array = self.times.reshape((self.times.shape[0],1))

        #X_estimate_minus_X_GT = X_Y_estimated_minus_X_Y_GT[:,0].reshape((X_Y_estimated_minus_X_Y_GT[:,0].shape[0],1))
        x_error = X_Y_estimated_minus_X_Y_GT[:,0]
        sigma_x = X_XY_XY_Y_uncertinty_cov_list[:,0]
        err_cov_x = tuple((x_error,sigma_x))
        y_error = X_Y_estimated_minus_X_Y_GT[:,1]
        sigma_y = X_XY_XY_Y_uncertinty_cov_list[:,3]
        err_cov_y = tuple((y_error,sigma_y))
        #graphs.plot_two_graphs_one_double(X_estimate_minus_X_GT_and_times,sigma_x_with_times,sigma_minus_x_with_times,'X estimated - XGT and sigma_x values', 'Times elapsed [sec]', 'X estimation error [m]' ,'estimation eror', 'estimated 1 sigma interval')
        graphs.plot_error(err_cov_x=err_cov_x,err_cov_y=err_cov_y)

        

        ani = graphs.build_animation(locations_GT[:,:2], enu_kf, enu_kf_dmr, X_XY_XY_Y_uncertinty_cov_list, 'trajectories', 'X [m]', 'Y [m]', 'GT', 'KF_estimat', 'dead reckoning')
        graphs.save_animation(ani, basedir, 'animation of GT KF estimate and dead reckoning')

        print('finished Q1')

        

    def Q2(self):

        """
        That function runs the code of question 2 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, yaw rate, and velocities, and apply a Kalman filter to the noisy data.
        """

        # plot yaw, yaw rate and forward velocity
        random.seed(10)#TODO)
        np.random.seed(10)#TODO)
        
        sigma_n = 0.01
        sigma_theta = 1.5
        
        # build_LLA_GPS_trajectory

        # add_gaussian_noise to u and measurments (locations_gt[:,i], sigma_samples[i])>> Done in the setup

        # plot vf and wz with and without noise
        times_array = self.times.reshape((self.times.shape[0],1))
        yaw_array = np.array(self.yaw_vf_wz[:,0],ndmin = 2).T
        vf_array = np.array(self.yaw_vf_wz[:,1],ndmin = 2).T
        wz_array = np.array(self.yaw_vf_wz[:,2],ndmin = 2).T 
        
        yaw_array_and_times = np.concatenate((times_array,yaw_array),axis = 1)
        vf_array_and_times = np.concatenate((times_array,vf_array),axis = 1)
        wz_array_and_times = np.concatenate((times_array,wz_array), axis = 1)
        
       
        noised_vf_array = np.array(self.yaw_vf_wz_noise[:,1],ndmin = 2).T
        noised_wz_array = np.array(self.yaw_vf_wz_noise[:,2],ndmin = 2).T 
        
        noised_vf_array_and_times = np.concatenate((times_array,noised_vf_array),axis = 1)
        noised_wz_array_and_times = np.concatenate((times_array,noised_wz_array), axis = 1)
        
        graphs.plot_single_graph(yaw_array_and_times,'ground-truth yaw angles', 'time elapsed [sec]','yaw angle [rad]','yaw angles GT' )
        graphs.plot_single_graph(vf_array_and_times,'ground-truth forward velocity ', 'time elapsed [sec]','forward velocity [m/s]','forward velocity GT' )
        graphs.plot_single_graph(wz_array_and_times,'ground-truth yaw rates', 'time elapsed [sec]','yaw rate [rad/s]','yaw rates GT' )

        #plot the noised array compared to GT
        graphs.plot_graph_and_scatter(vf_array_and_times,noised_vf_array_and_times,'original GT and noised forword velocity ','Forword Velocity [m/s]','Elapsed Time [sec]','Vf_GT', 'Vf_noised')
        graphs.plot_graph_and_scatter(wz_array_and_times,noised_wz_array_and_times,'original GT and noised yaw rates','Yaw Rate [rad/s]','Elapsed Time [sec]','Vf_GT', 'Vf_noised',)
        
           
        
        sigma_vf=0
        sigma_wz =0
        
        ekf = ExtendedKalmanFilter(self.enu_noise,self.yaw_vf_wz,self.times,self.sigma_xy,sigma_theta,sigma_vf,sigma_wz,sigma_n=sigma_n,k=2.0)
        locations_ekf, sigma_x_xy_yx_y_t = ekf.run()#locations_noised, times, yaw_vf_wz_noised, do_only_predict=False)
        GT = np.concatenate((self.enu[:,:2],yaw_array),axis= 1)
        
        RMSE, maxE = ekf.calc_RMSE_maxE(GT, locations_ekf)

        # print the maxE and RMSE
        print(f"RMSE: {RMSE:.2f} meters, Max Error: {maxE:.2f} meters of the non noised yaw_vf_wz ")
        # draw the trajectories
        graphs.plot_trajectory_comparison(self.enu, locations_ekf,type='EKF no noise in commands')
        
        #Now we will apply the EKF on the noised yaw_vf_wz
        ekf_noise = ExtendedKalmanFilter(self.enu_noise,self.yaw_vf_wz_noise,self.times,self.sigma_xy,sigma_theta,self.sigma_vf,self.sigma_wz,sigma_n=sigma_n,k=2.0)
        locations_ekf_noised, covs = ekf_noise.run()
        
        RMSE, maxE = ekf.calc_RMSE_maxE(GT, locations_ekf_noised)

        # print the maxE and RMSE
        print(f"RMSE: {RMSE:.2f} meters, Max Error: {maxE:.2f} meters")
        # draw the trajectories
        graphs.plot_trajectory_comparison(self.enu, locations_ekf_noised,type='EKF_with_vf_wz_noise')
        
        ## now we will applay the same, only this time with dead reconing
        ekf_noise_DR = ExtendedKalmanFilter(self.enu_noise,self.yaw_vf_wz_noise,self.times,self.sigma_xy,sigma_theta,self.sigma_vf,self.sigma_wz,sigma_n=sigma_n,k=2.0,is_dead_reckoning=True)
        locations_ekf_noised_dr, sigma_x_xy_yx_y_t_dr = ekf_noise_DR.run()
        
        RMSE, maxE = ekf.calc_RMSE_maxE(GT, locations_ekf_noised_dr)
        # draw the error
        X_Y_estimated_minus_X_Y_GT = GT - locations_ekf_noised
        
        times_array = self.times.reshape((self.times.shape[0],1))

        #X_estimate_minus_X_GT = X_Y_estimated_minus_X_Y_GT[:,0].reshape((X_Y_estimated_minus_X_Y_GT[:,0].shape[0],1))
        x_error = X_Y_estimated_minus_X_Y_GT[:,0]
        sigma_x = covs[:,0]
        err_cov_x = tuple((x_error,sigma_x))
        y_error = X_Y_estimated_minus_X_Y_GT[:,1]
        sigma_y = covs[:,4]
        err_cov_y = tuple((y_error,sigma_y))
        theta_error_normalized = (locations_ekf_noised[: , 2] - self.yaw_vf_wz[: , 0])/  locations_ekf_noised[: , 2].max()
        sigma_theta = covs[:,8]
        err_cov_theta = tuple((theta_error_normalized,sigma_theta))
        
        graphs.plot_error(err_cov_x=err_cov_x,err_cov_y=err_cov_y,err_cov_yaw=err_cov_theta,type='EKF')

        #v.	Plot the estimated error of x-y-θ values separately and corresponded sigma value along the trajectory
        
        X_XY_XY_Y_uncertinty_cov_list = covs[:,[0,1,3,4]]
 
        # build_animation
        ani = graphs.build_animation(GT[:,:2], locations_ekf_noised[:,:2], locations_ekf_noised_dr[:,:2], X_XY_XY_Y_uncertinty_cov_list, 'trajectories', 'X [m]', 'Y [m]', 'GT', 'EKF_estimat', 'dead reckoning')
        graphs.save_animation(ani, basedir, 'ekf_predict')
       
            # animation that shows the covariances of the EKF estimated path

            # this animation that shows the covariances of the dead reckoning estimated path

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")
        print('finished Q2')

    
    def get_odometry(self, sensor_data):
        """
        Args:
            sensor_data: map from a tuple (frame number, type) where type is either ‘odometry’ or ‘sensor’.
            Odometry data is given as a map containing values for ‘r1’, ‘t’ and ‘r2’ – the first angle, the translation and the second angle in the odometry model respectively.
            Sensor data is given as a map containing:
              - ‘id’ – a list of landmark ids (starting at 1, like in the landmarks structure)
              - ‘range’ – list of ranges, in order corresponding to the ids
              - ‘bearing’ – list of bearing angles in radians, in order corresponding to the ids

        Returns:
            numpy array of of dim [num of frames X 3]
            first two components in each row are the x and y in meters
            the third component is the heading in radians
        """
        num_frames = len(sensor_data) // 2
        state = np.array([[0, 0, 0]], dtype=float).reshape(1, 3)
        for i in range(num_frames):
            curr_odometry = sensor_data[i, 'odometry']
            t = np.array([
                curr_odometry['t'] * np.cos(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['t'] * np.sin(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['r1'] + curr_odometry['r2']
            ]).reshape(3, 1)
            new_pos = state[-1, :].reshape(3, 1) + t
            state = np.concatenate([state, new_pos.reshape(1, 3)], axis=0)
        return state
    
        
    def Q3(self):

        """
        Runs the code for question 3 of the project
        Loads the odometry (robot motion) and sensor (landmarks) data supplied with the exercise
        Adds noise to the odometry data r1, trans and r2
        Uses the extended Kalman filter SLAM algorithm with the noisy odometry data to predict the path of the robot and
        the landmarks positions
        """
        random.seed(10)#TODO)
        np.random.seed(10)#TODO)
        
        #Pre-processing
        landmarks = self.dataset.load_landmarks()
        sensor_data_gt = self.dataset.load_sensor_data()
        state = self.get_odometry(sensor_data_gt)      
        sigma_x_y_theta = [3,3, 1] #TODO
        variance_r1_t_r2 = [0.01,0.1,0.01]#TODO
        variance_r_phi = [0.3,0.035]#TODO
        sigma_n =0.1

        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        # plot trajectory 
          #TODO 
        graphs.plot_single_graph(X_Y=state,title='odometry GT trajectory', xlabel='X [m]', ylabel='Y [m]', label='GT trajectory odometry' ) 
        # plot trajectory + noise
          #TODO
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # KalmanFilter
        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi,sigma_n=sigma_n)

        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)
         #RMSE, maxE =calc_RMSE_maxE
        maxE = 0
        e_x = mu_arr_gt[20:,0] - mu_arr[20:,0]
        e_y = mu_arr_gt[20:,1] - mu_arr[20:,1]
        maxE = max(abs(e_x)+abs(e_y))
        RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(mu_arr_gt.shape[0]-20))
        print("RMSE", RMSE, "maxE", maxE)
        
        
        '''
        # KalmanFilter
        best_sigma = 1.0
        min_error = 100
        
        for i in range(1,1001,1):
            sima_n = i/1000.0
            ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi,sigma_n=sima_n)

            frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

             #RMSE, maxE =calc_RMSE_maxE
            maxE = 0
            e_x = mu_arr_gt[20:,0] - mu_arr[20:,0]
            e_y = mu_arr_gt[20:,1] - mu_arr[20:,1]
            maxE = max(abs(e_x)+abs(e_y))
            RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(mu_arr_gt.shape[0]-20))
            if maxE < min_error:
                print("xurrent best is:RMSE", RMSE, "maxE", maxE, 'abd rhe sigma is',sima_n) 
                min_error = maxE
                best_sigma = sima_n
        '''
        
        
        # draw the error for x, y and theta

        # Plot the estimated error ofof x-y-θ -#landmark values separately and corresponded sigma value along the trajectory

        # draw the error
        
        graphs.plot_trajectory_comparison(mu_arr_gt, mu_arr,type='SLAM+EKF')

        graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-x_n", "frame", "error", "x-x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
        graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-y_n", "frame", "error", "y-y_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
        graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "theta-theta_n", 
                                 "frame", "error", "theta-theta_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))
        
        graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]), 
                                 "landmark 1 x-x_n", "frame", "error [m]", "x-x_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
        graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]), 
                                 "landmark 1 y-y_n", "frame", "error [m]", "y-y_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))
        
        graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
                                 "landmark 2 x-x_n", "frame", "error [m]", "x-x_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
        graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
                                 "landmark 2 y-y_n", "frame", "error [m]", "y-y_n", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))
        
        ax.set_xlim([-2, 12])
        ax.set_ylim([-2, 12])
        
        
        graphs.show_graphs()
        # ani.save('im.mp4', metadata={'artist':'me'})
        ani = animation.ArtistAnimation(fig, frames, repeat=False)
        graphs.save_animation(ani, basedir, 'animation of Trajectory of EKF-SLAM')
        print('finished Q3')
    
    def run(self):
        #
        self.Q1()
        self.Q2()
        self.Q3()
        
