# Mapping and perception for autonomous robot , semester B 2022/23
# Roy orfaig & Ben-Zion Bobrovsky
# Project 2 - Robot localization and SLAM!
# Kalman filter, Extended Kalman Filter and EKF-SLAM. 

import os
from data_loader import DataLoader
from project_questions import ProjectQuestions


if __name__ == "__main__":

    basedir = ".\Data\kitti_data"
    date = '2011_09_26'
    drive = '0061' #"TODO" - The recording number I used in the sample during class is (insert your correct record number)

    dat_dir = os.path.join(basedir,"Ex3_data")
    
    dataset = DataLoader(basedir, date, drive, dat_dir)
    if not os.path.exists('Results'):
        os.mkdir('Results')
    project = ProjectQuestions(dataset)
    project.run()