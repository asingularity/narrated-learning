
import cv2
import random
import os
import numpy as np
random.seed(6)
np.random.seed(6)
from multiprocessing import Pool
import datetime
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt


from sim_folder_manager import SimFolderManager
from robot_sensor_classes.video_playback import VideoPlaybackSensor
from robot_preprocess_classes.event_pre_processor import EventPreProcessor
from robot_brain_classes.rate_control_wta import RateControlWTABRain



RF_IM_DIM = 8  # 8, 16, 32, 64
ROOT_DIR = '/srv'


def get_sensors_params():  # override_params

    # 'video_filename': 'seattle-driving-fkps18H3SXY.mp4',

    square_crop = (128 - int(RF_IM_DIM / 2), 128 - int(RF_IM_DIM / 2), RF_IM_DIM)

    params_video_playback = {
        'pickle_im_square_crop': None,
        'pickle_im_square_resize': 256,
        'force_reload_to_pkl': False,
        'gb_per_pkl_file': 1,
        'returned_im_square_crop': square_crop,
        'returned_im_dtype': np.float32,  # only np.float32 supported
        'returned_im_use_color': False,  # only False supported for now
        'video_dir': ROOT_DIR + '/projects/video-downloads',
        #'video_filename': 'sea-turtles-yLuEx-XH3Uc.mp4'
        #'video_filename': 'seattle-driving-fkps18H3SXY.mp4'
        'video_filename': 'sea-turtles-11hr-spxtEt6RaS4.mp4'
    }

    # for pname in override_params.keys():
    #     params_video_playback[pname] = override_params[pname]

    return params_video_playback


def get_pre_proc_params():  # override_params
    params_pre_proc = {
        'brightness_threshold': 1.0/255,
        'im_dim': RF_IM_DIM
    }

    # for pname in override_params.keys():
    #     params_pre_proc[pname] = override_params[pname]

    return params_pre_proc


def get_brain_params():  # override_params
    brain_params = {
        'input_im_dim': RF_IM_DIM,
        'num_rfs': 400,
        'lr': 1.0 / 1000,
        'rate_lr': 1.0 / 1000,
        'apply_rate_control': False,
        'do_raster_plots_every_k_im': None  # or None
    }

    # for pname in override_params.keys():
    #     brain_params[pname] = override_params[pname]

    return brain_params


def run_one_sim(sim_params):

    sim_params['brain_params']['max_time'] = sim_params['max_time']

    robot_brain = RateControlWTABRain(sim_params['brain_params'])
    robot_sensors = VideoPlaybackSensor(sim_params['sensor_params'])
    pre_proc = EventPreProcessor(sim_params['pre_processor_params'])

    sim_folder_manager = SimFolderManager(sim_params['sim_folder_manager_params'])

    robot_brain.set_plots_folder(sim_folder_manager.get_plots_save_folder())

    random.seed(1233)

    for t in range(sim_params['max_time']):
        im = robot_sensors.read_input()
        events_p, events_n = pre_proc.step(input_frame=im)
        robot_brain.process_input(input_events_p=events_p, input_events_n=events_n)

    robot_brain.do_plots()
    ims_list, ims_names_list = robot_brain.get_table_ims()

    #  instead of opencv viz, should save rfs image at end!!
    for k in range(len(ims_list)):
        cv2.imwrite(filename=sim_folder_manager.get_plots_save_folder() + '/' + ims_names_list[k] + ".png",
                    img=(255 * ims_list[k]).astype(np.uint8))

    final_errors_dict = robot_brain.get_final_errors_dict()

    return_thing = (final_errors_dict, sim_folder_manager.get_plots_save_folder())

    return return_thing


def get_sweep_sims(param_set, sim_prefix):

    param_type, param_name, param_values = param_set
    sim_params_list = []

    print()
    print('starting an experiment')
    print()

    now = datetime.datetime.now().isoformat()
    sweep_name = param_name + '_' + now

    sweep_folder = ROOT_DIR + '/projects/NL-sim-sweeps/' + sim_prefix + '/' + sweep_name
    os.makedirs(sweep_folder)

    for param_value in param_values:
        sim_params = {}

        # TODO move this elsewhere and should be longer!!!
        sim_params['max_time'] = 5000000

        sim_params['brain_params'] = get_brain_params()
        sim_params['sensor_params'] = get_sensors_params()
        sim_params['pre_processor_params'] = get_pre_proc_params()

        # then create sim folder manager params
        sim_params['sim_folder_manager_params'] = {
            'sim_prefix': param_name + '_' + str(param_value),
            'sim_folders_path': sweep_folder + '/',
            'scripts_folder_path': ROOT_DIR + '/projects/NL/'
        }

        print('    setting a sim:', param_type, param_name, param_value)

        # override parameter for sweep
        sim_params[param_type][param_name] = param_value

        sim_params_list.append(sim_params)

    return sim_params_list, sweep_folder


def run_several_sweeps():
    print()
    sim_prefix = input('run prefix? >> ')
    print()

    if sim_prefix[-1] != '_':
        sim_prefix += '_'

    if len(sim_prefix) == 0:
        sim_prefix = 'default_'

    now = datetime.datetime.now().isoformat()
    sim_prefix += now

    # define param sets to vary in successive experiments

    # ('pre_processor_params', 'brightness_threshold', [1.0/255, 10.0/255, 20.0/255]),

    param_sets = [('brain_params', 'num_rfs', [200, 400, 800, 1600, 3200]),
                  ('brain_params', 'apply_rate_control', [True, False])]

    # each experiment is running a set of simulations for one of the param sets defined above
    # call them in order

    # should dump all sims from all param sets into same pool.map for optimum efficiency
    all_sim_params_list = []  # params

    # later: instead should it be grid search? i.e. combine all variants of params above?
    sweep_folders = []
    for param_set in param_sets:
        sim_params_list, sweep_folder = get_sweep_sims(param_set=param_set, sim_prefix=sim_prefix)
        all_sim_params_list.extend(sim_params_list)
        sweep_folders.append(sweep_folder)

    # distribute and run simulations using multiprocess
    print()
    print('starting simulations via multiprocess...')
    print()

    # TODO MORE PROCESSES!
    with Pool(processes=18) as pool:

        return_things = pool.map(run_one_sim, all_sim_params_list)

        final_errors_dicts = []
        sim_folder_paths = []

        for k in range(len(return_things)):
            final_errors_dicts.append(return_things[k][0])
            sim_folder_paths.append(return_things[k][1])

        print()
        print('plotting final error values...')
        print()
        k = 0

        fig = plt.figure(figsize=(40, 20))
        ax = fig.add_subplot(1, 1, 1)

        r = -1
        for param_set in param_sets:
            r += 1
            sweep_folder = sweep_folders[r]
            # this is one sweep: one set of plots

            param_type, param_name, param_values = param_set

            plot_x = []
            plot_ys = {}
            error_types = None

            for param_value in param_values:

                final_errors_dict = final_errors_dicts[k]
                error_types = final_errors_dict.keys()
                plot_x.append(param_value)

                for error_type in final_errors_dict.keys():
                    if error_type in plot_ys:
                        plot_ys[error_type].append(final_errors_dict[error_type])
                    else:
                        plot_ys[error_type] = [final_errors_dict[error_type]]

                k += 1

            for error_type in error_types:
                ax.cla()
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)

                ax.plot(np.array(plot_x), np.array(plot_ys[error_type]), color='k', marker='.')
                fig.savefig(sweep_folder + "/" + error_type + ".png", dpi=100)


if __name__ == '__main__':

    run_several_sweeps()
