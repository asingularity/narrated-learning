import os
import datetime
import shutil


class SimFolderManager(object):
    def __init__(self, params):
        self.sim_prefix = params['sim_prefix']

        now = datetime.datetime.now().isoformat()
        self.sim_folder_path = params['sim_folders_path'] + '/' + now
        os.makedirs(self.sim_folder_path)

        self.models_save_folder = self.sim_folder_path
        self.plots_save_folder = self.sim_folder_path

        # TODO fix this again or just use git commit id
        file_list = []
        # file_list = ['robot_brain.py',
        #              'robot_model.py',
        #              'robot_environment.py',
        #              'robot_sensors.py',
        #              'run_demo.py',
        #              'run_offline_training.py',
        #              'brain_components.py',
        #              'brain_components_classes/motor_history.py',
        #              'brain_components_classes/autoencoder.py',
        #              'brain_components_classes/states_history.py',
        #              'brain_components_classes/inverse_model.py',
        #              'brain_components_classes/predictor.py',
        #              'sim_folder_manager.py',
        #              'performance_evaluator.py',
        #              'setup.py',
        #              'setup_fast_save_matrix.py',
        #              'task_manager.py',
        #              'visualizer.py',
        #              'install_cython_libraries.sh']

        for filename in file_list:
            shutil.copy2(params['scripts_folder_path'] + '/' + filename, self.sim_folder_path)

    def get_models_save_folder(self):
        return self.models_save_folder

    def get_plots_save_folder(self):
        return self.plots_save_folder
