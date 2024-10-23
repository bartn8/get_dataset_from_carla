import os
import sys
import subprocess
import time
from datetime import datetime
import multiprocessing
import signal
import psutil
from tqdm import tqdm

from data_generator.data_creation.generate_traffic import generate_traffic
from ..utils import color_error_string
from ..config import TOWN_DICT

def check_integrity_of_carla_path(args):
    """
    Check Integrity of the Carla Path
    """
    # (1) Check that the Carla's Path really exists
    if not os.path.isdir(args.carla_path):
        raise Exception(color_error_string(f"The given Carla Path doesn't exist! [{args.carla_path}]"))
    # (2) Check that the egg file is really present and it works: being able to import carla!
    carla_pythonapi_dist_path = os.path.join(args.carla_path, "PythonAPI/carla/dist")
    if not os.path.isdir(carla_pythonapi_dist_path):
        raise Exception(color_error_string(f"The given Carla doen't contains a PythonAPI! [{carla_pythonapi_dist_path}]"))
    egg_files = [file for file in os.listdir(carla_pythonapi_dist_path) if file[-len(args.end_of_egg_file):] == args.end_of_egg_file]
    if len(egg_files) == 0:
        raise Exception(color_error_string(f"The given Carla doen't contains a \"*{args.end_of_egg_file}\" file in \"{carla_pythonapi_dist_path}\""))
    if len(egg_files) > 1:
        raise Exception(color_error_string(f"The given Carla contains to many \"*{args.end_of_egg_file}\" files in \"{carla_pythonapi_dist_path}\"\n" +
                                                  "Set a more restrict search with the \"--end_of_egg_file\" arguments!"))
    egg_file_path = os.path.join(carla_pythonapi_dist_path, egg_files[0])
    # Now that we have a unique egg file we add it to the python path!
    sys.path.append(egg_file_path)
    # (3) Check that the CarlaUE4 executable is present
    carlaUE4_folder = os.path.join(args.carla_path, "CarlaUE4/Binaries/Linux/")
    if not os.path.isdir(carlaUE4_folder):
        raise Exception(color_error_string(f"The folder in wicth I was expecting \"CarlaUE4-Linux-Shipping\" doesn't exists! [{carlaUE4_folder}]"))
    files = os.listdir(carlaUE4_folder)
    if "CarlaUE4-Linux-Shipping" not in files:
        raise Exception(color_error_string(f"I cannot find \"CarlaUE4-Linux-Shipping\" executable in \"{carlaUE4_folder}\"!"))
    carlaUE4_path = os.path.join(carlaUE4_folder, "CarlaUE4-Linux-Shipping")
    return egg_file_path, carlaUE4_path

def launch_carla_server_saifly_and_wait_till_its_up(rpc_port, carla_server_pid, carlaUE4_path, logs_path, how_many_seconds_to_wait, show_carla_window=False):
    def start_up_carla_server(rpc_port, carla_server_pid, carlaUE4_path, logs_path, how_many_seconds_to_wait):
        with open(logs_path, 'r+') as logs_file:
            if not show_carla_window:
                command_as_list = ["/usr/bin/stdbuf", "-o0", carlaUE4_path, "-RenderOffScreen", "-nosound", f"-carla-rpc-port={rpc_port}"]
            else:
                command_as_list = ["/usr/bin/stdbuf", "-o0", carlaUE4_path, "-nosound", f"-carla-rpc-port={rpc_port}"]
            carla_process = subprocess.Popen(
                command_as_list,
                stdout=logs_file,
                stderr=logs_file,
                universal_newlines=True
            )
        carla_server_pid.value = carla_process.pid
        # We will wait Carla to start up!
        while True:
            time.sleep(0.1)
            with open(logs_path, 'r') as logs_file:
                lines = len(logs_file.readlines())
                if lines >= 2:
                    break
            return_code = carla_process.poll()
            if return_code is not None:
                # The Carla process died before starting up!
                exit()

        print("Waiting Carla to Start...", end="", flush=True)
        import carla
        start_time = datetime.now()
        while True:
            try:
                client = carla.Client('localhost', rpc_port)
                client.set_timeout(1.0)
                _ = client.get_world()
                break
            except RuntimeError as e:
                pass
            print("*", end="", flush=True)
            if (datetime.now() - start_time).total_seconds() > how_many_seconds_to_wait:
                break
        print()

    # FIRST OF ALL KILL ALL CARLA SERVER RUNNING
    for proc in psutil.process_iter():
        if "CarlaUE4-Linux-Shipping" in proc.name():
            os.kill(proc.pid, signal.SIGKILL)
    if not os.path.isdir(os.path.dirname(logs_path)):
        try:
            os.mkdir(os.path.dirname(logs_path))
        except:
            Exception(color_error_string(f"Unable to find out log dir! [{os.path.dirname(logs_path)}]"))
    if os.path.isfile(logs_path):
        os.remove(logs_path)
    with open(logs_path, 'w') as _:
        pass
    
    check_carla_process = multiprocessing.Process(target=start_up_carla_server, args=(rpc_port, carla_server_pid, carlaUE4_path, logs_path, how_many_seconds_to_wait))
    check_carla_process.start()
    # Let's wait till Carla Server is Up!
    while True:
        if not check_carla_process.is_alive():
            check_carla_process.join()
            if not psutil.pid_exists(carla_server_pid.value):
                return False
            else:
                return True

def set_up_world_saifly_and_wait_till_its_setted_up(carla_ip, rpc_port, town_number, carla_server_pid):
    def set_up_world(carla_ip, rpc_port, town_number, world_setted_up):
        try:
            import carla
        except:
            raise Exception(color_error_string(f"Not able to import Carla!"))
        client = carla.Client(carla_ip, rpc_port)
        client.set_timeout(1000.0)
        client.load_world(TOWN_DICT[town_number])
        world_setted_up.set()

    world_setted_up = multiprocessing.Event()
    set_up_world_process = multiprocessing.Process(target=set_up_world, args=(carla_ip, rpc_port, town_number, world_setted_up))
    set_up_world_process.start()

    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            set_up_world_process.kill()
            return False
        if not set_up_world_process.is_alive():
            set_up_world_process.join()
            if world_setted_up.is_set():
                return True
            else:
                os.kill(carla_server_pid.value, signal.SIGKILL)
                return False

def set_up_traffic_manager_saifly_and_wait_till_its_up(carla_ip, rpc_port, tm_port, number_of_vehicles, number_of_walkers, carla_server_pid, traffic_manager_pid, logs_path, hero=True):
    you_can_tick = multiprocessing.Event()
    traffic_manager_is_up = multiprocessing.Event()
    set_up_traffic_manager_process = multiprocessing.Process(target=generate_traffic, args=(carla_ip, rpc_port, tm_port, number_of_vehicles, number_of_walkers, traffic_manager_is_up, you_can_tick, logs_path, hero))
    set_up_traffic_manager_process.start()

    traffic_manager_pid.value = set_up_traffic_manager_process.pid

    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            set_up_traffic_manager_process.kill()
            return False, True, you_can_tick, traffic_manager_is_up, set_up_traffic_manager_process     # Means Carla Crashed!
        if not set_up_traffic_manager_process.is_alive():
            set_up_traffic_manager_process.join()
            os.kill(carla_server_pid.value, signal.SIGKILL)
            return True, False, you_can_tick, traffic_manager_is_up, set_up_traffic_manager_process   # Means Traffic Manager Crashed!
        if traffic_manager_is_up.is_set():
            return True, True, you_can_tick, traffic_manager_is_up, set_up_traffic_manager_process     # Means everything good!
