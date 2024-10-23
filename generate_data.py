import argparse
import multiprocessing
import os
import pathlib
import time
import signal
import psutil
from ctypes import c_int
from tabulate import tabulate
import shutil

from data_generator.data_creation import take_data_without_records
from data_generator import utils
from data_generator import config
from datetime import datetime
from data_generator.carla_interface.run_carla import check_integrity_of_carla_path, \
    launch_carla_server_saifly_and_wait_till_its_up, \
    set_up_world_saifly_and_wait_till_its_setted_up, \
    set_up_traffic_manager_saifly_and_wait_till_its_up


def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--carla_ip',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '--rpc_port',
        default=2000,
        help='Carla RPC port (deafult: 2000)',
        type=int
    )
    argparser.add_argument(
        '--tm_port',
        default=8000,
        help='Traffic Manager port (default: 8000)',
        type=int
    )
    argparser.add_argument(
        '--town',
        default=10,
        help='Witch town to select (default: 10)',
        type=int
    )
    argparser.add_argument(
        '--job_id',
        default=0,
        help='Job ID to use (default: 0)',
        type=int
    )
    argparser.add_argument(
        '--end_of_egg_file',
        help='How the egg file should end to be valid! (default: py3.7-linux-x86_64.egg)',
        default="py3.7-linux-x86_64.egg",
        type=str
    )
    argparser.add_argument(
        '--carla_path',
        help='Path to the Carla Installation!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--num_of_vehicle',
        help='Number of Vehicle to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--num_of_walkers',
        help='Number of Walkers to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--num_of_frames',
        help='Number of Frames to take! (default: 10)',
        required=False,
        default=10,
        type=int
    )
    argparser.add_argument(
        '--dataset_path',
        help=f'Where to save the data! (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets")})',
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets"),
        type=str
    )
    argparser.add_argument(
        '--back_camera',
        help='Set if we want to add a camera in the back!',
        action='store_true'
    )
    argparser.add_argument(
        '--lateral_cameras',
        help='Set if we want to add left and right lateral cameras!',
        action='store_true'
    )
    argparser.add_argument(
        '--show_carla_window',
        help='If ypu want to see the carla window!',
        action='store_true'
    )
    args = argparser.parse_args()
    if args.town not in config.TOWN_DICT:
        error = f"Invalid Town Index! [{args.town}]\n" + \
                "Possible Town Index:\n"
        for key in config.TOWN_DICT:
            error += f"{key} -> {config.TOWN_DICT[key]}\n"
        raise Exception(utils.color_error_string(error))
    return args


def kill_all(carla_server_pid, traffic_manager_pid, data_creation_pid):
    try:
        os.kill(carla_server_pid, signal.SIGKILL)
    except:
        pass
    try:
        os.kill(traffic_manager_pid, signal.SIGKILL)
    except:
        pass
    try:
        os.kill(data_creation_pid, signal.SIGKILL)
    except:
        pass


pids_to_be_killed = []


def run_all(args, where_to_save):
    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_saifly_and_wait_till_its_up(
        rpc_port=args.rpc_port,
        carla_server_pid=carla_server_pid,
        carlaUE4_path=carlaUE4_path,
        logs_path=carla_log_path,
        how_many_seconds_to_wait=100,
        show_carla_window=args.show_carla_window,
    )

    pids_to_be_killed.append(carla_server_pid.value)

    if not carla_was_correctly_started_up:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while starting!"))

    print(utils.color_info_string("(1/3)\tCarla Server is UP!"))

    # (3) SET UP THE WORLD
    world_was_correctly_setted_up = set_up_world_saifly_and_wait_till_its_setted_up(
        carla_ip=args.carla_ip,
        rpc_port=args.rpc_port,
        town_number=args.town,
        carla_server_pid=carla_server_pid
    )

    if not world_was_correctly_setted_up:
        raise utils.NutException(utils.color_error_string(f"Failed to set up world!"))

    print(utils.color_info_string("(2/3)\tWorld was correctly setted up!"))

    # (4) SET UP TRAFFIC MANAGER
    traffic_manager_pid = multiprocessing.Value(c_int)
    carla_is_ok, \
        traffic_manager_is_ok, \
        you_can_tick, \
        traffic_manager_is_up, \
        set_up_traffic_manager_process = set_up_traffic_manager_saifly_and_wait_till_its_up(
        carla_ip=args.carla_ip,
        rpc_port=args.rpc_port,
        tm_port=args.tm_port,
        number_of_vehicles=args.num_of_vehicle,
        number_of_walkers=args.num_of_walkers,
        carla_server_pid=carla_server_pid,
        traffic_manager_pid=traffic_manager_pid,
        logs_path=traffic_manager_log_path
    )

    pids_to_be_killed.append(traffic_manager_pid.value)

    if not carla_is_ok:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Traffic Manager!"))
    if not traffic_manager_is_ok:
        raise utils.NutException(utils.color_error_string(f"Traffic Manager Crashed!"))

    print(utils.color_info_string("(3/3)\tTraffic Manager Setted Up properly!"))

    # (5) LAUNCH DATA CREATION PROCESS
    data_creation_pid = multiprocessing.Value(c_int)
    ego_vehicle_found_event = multiprocessing.Event()
    finished_taking_data_event = multiprocessing.Event()
    data_creation_process = multiprocessing.Process(target=take_data_without_records.take_data_backbone,
                                                    args=(egg_file_path, args.town, args.rpc_port, args.job_id,
                                                          ego_vehicle_found_event, finished_taking_data_event,
                                                          you_can_tick, args.num_of_frames, where_to_save,
                                                          args.back_camera, args.lateral_cameras))
    data_creation_process.start()
    data_creation_pid.value = data_creation_process.pid
    pids_to_be_killed.append(data_creation_pid.value)

    print(utils.get_a_title(f"STARTING TO TAKE DATA [{args.num_of_frames}]", color="green"))
    start_time = time.time()
    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            # kill_all(carla_server_pid, traffic_manager_pid, data_creation_pid)
            raise utils.NutException(utils.color_error_string(f"Carla crashed!"))
        if not set_up_traffic_manager_process.is_alive():
            # kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise utils.NutException(utils.color_error_string(f"Traffic Manager crashed!"))
        if not data_creation_process.is_alive():
            # kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise utils.NutException(utils.color_error_string(f"Data Creation crashed!"))
        if not ego_vehicle_found_event.is_set() and time.time() - start_time > 10:
            # kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise utils.NutException(
                utils.color_error_string(f"Data Creation is not able to find out the Ego Vehicle!"))
        if finished_taking_data_event.is_set():
            break

    print(utils.get_a_title(f"FINISHED TO TAKE DATA [{args.num_of_frames}]", color="green"))

    # (6) CLEANING EVERYTHING
    kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
    return True


if __name__ == "__main__":
    args = get_arguments()
    egg_file_path, carlaUE4_path = check_integrity_of_carla_path(args)
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))

    print(utils.get_a_title("STARTING LAUNCHING ALL THE PROCESS", color="blue"))
    print(utils.color_info_success(f"Find out a valid carla in {egg_file_path}!"))

    # (0) SET UP LOGS AND DATASETFOLDER
    nutfuser_folder_path = pathlib.Path(__file__).parent.resolve()
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_logs_{args.job_id}.log")
    traffic_manager_log_path = os.path.join(nutfuser_folder_path, "logs", f"traffic_manager_logs_{args.job_id}.log")
    datasets_path = args.dataset_path

    if not os.path.isdir(datasets_path):
        try:
            os.mkdir(datasets_path)
        except:
            Exception(utils.color_error_string(f"Unable to create [{datasets_path}] dir!"))


    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    times_that_i_need_to_take_data = 1

    for k in range(times_that_i_need_to_take_data):
        if times_that_i_need_to_take_data > 1:
            to_print = f"# Taking Data Loop number {k + 1}/{times_that_i_need_to_take_data} #"
            print(utils.color_info_string("#" * len(to_print)))
            print(utils.color_info_string(to_print))
            print(utils.color_info_string("#" * len(to_print)))
            args.job_id = k
        # LET'S RUN ALL
        where_to_save = None
        for i in range(config.MAX_NUM_OF_ATTEMPTS):
            if i > 0:
                # It's not the first attempt to we need to remove the previous failed data
                shutil.rmtree(where_to_save, ignore_errors=False, onerror=None)
            now = datetime.now()
            current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
            where_to_save = os.path.join(datasets_path, f"{args.job_id}_{current_time}_{config.TOWN_DICT[args.town]}")
            os.mkdir(where_to_save)
            try:
                print(utils.get_a_title(f"ATTEMPT [{i + 1}/{config.MAX_NUM_OF_ATTEMPTS}]", "blue"))
                if run_all(args, where_to_save):
                    break
            except utils.NutException as e:
                print(e.message)
                args.rpc_port += 1
                for pid in pids_to_be_killed:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except:
                        pass
                pids_to_be_killed = []
            except KeyboardInterrupt:
                for pid in pids_to_be_killed:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except:
                        pass
                exit()
