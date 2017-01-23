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

        file_list = ['robot_brain.py',
                     'robot_model.py',
                     'robot_environment.py',
                     'robot_sensors.py',
                     'run_demo.py',
                     'sim_folder_manager.py',
                     'performance_evaluator.py',
                     'visualizer.py']

        for filename in file_list:
            shutil.copy2(params['scripts_folder_path'] + '/' + filename, self.sim_folder_path)

    def get_models_save_folder(self):
        return self.models_save_folder

    def get_plots_save_folder(self):
        return self.plots_save_folder
