import sys
import os
import math
import signal
import time
import numpy as np
import cv2
from tqdm import tqdm
import json

from ..config import CARLA_FPS, AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE, IMAGE_W, IMAGE_H
from ..utils import lidar_to_histogram_features, color_info_string
from .weather import get_a_random_weather

STARTING_FRAME = None
PATHS = {}
ALREADY_OBTAINED_DATA_FROM_SENSOR = []


def take_data_backbone(carla_egg_path, town_id, rpc_port, job_id, ego_vehicle_found_event, finished_taking_data_event,
                       you_can_tick_event, how_many_frames, where_to_save, back_camera, lateral_cameras):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    # Setup witch cameras is present
    cameras_indexes = [0]
    if back_camera:
        cameras_indexes.append(2)
    if lateral_cameras:
        cameras_indexes.append(1)
        cameras_indexes.append(3)

    lidar_indexes = [16, 32, 64, 128]

    global ALREADY_OBTAINED_DATA_FROM_SENSOR
    ALREADY_OBTAINED_DATA_FROM_SENSOR = {f"rgb_{i}": False for i in cameras_indexes}
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({f"depth_{i}": False for i in cameras_indexes}, **ALREADY_OBTAINED_DATA_FROM_SENSOR)
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({f"normals_{i}": False for i in cameras_indexes}, **ALREADY_OBTAINED_DATA_FROM_SENSOR)
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({f"semantic_{i}": False for i in cameras_indexes}, **ALREADY_OBTAINED_DATA_FROM_SENSOR)
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({f"lidar_{i}": False for i in lidar_indexes}, **ALREADY_OBTAINED_DATA_FROM_SENSOR)

    # Connect the client and set up bp library
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1. / CARLA_FPS
    # In this case, the simulator will take 20 steps (1/0.05) to recreate one second of
    # the simulated world.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    # fixed_delta_seconds <= max_substep_delta_time * max_substeps
    # In order to have an optimal physical simulation,
    # the substep delta time should at least be below 0.01666 and ideally below 0.01.
    world.apply_settings(settings)
    bp_lib = world.get_blueprint_library()

    # Search the CAR
    hero = None
    while hero is None:
        print("Waiting for the ego vehicle...")
        possible_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in possible_vehicles:
            if vehicle.attributes['role_name'] == 'hero':
                print("Ego vehicle found")
                hero = vehicle
                break
        time.sleep(1)
    ego_vehicle_found_event.set()


    # I will set a random weather
    a_random_weather, weather_dict = get_a_random_weather()
    world.set_weather(a_random_weather)

    # LIDAR callback
    def lidar_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))            
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            np.savez_compressed(os.path.join(PATHS[f"lidar_{number}"], f"{saved_frame}.npz"), lidar_data_raw=lidar_data_raw)
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"lidar_{number}"] = True

    # CAMERAS callback
    def rgb_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"rgb_{number}"], f"{saved_frame}.jpg"), rgb)
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"rgb_{number}"] = True

    # DEPTH callback
    def depth_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            depth = depth[:, :, :3]
            depth = depth[:, :, ::-1]
            R, G, B = depth[:, :, 0], depth[:, :, 1], depth[:, :, 2]
            depth_normalized = ((R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0))
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"depth_{number}"], f"{saved_frame}.png"), (65535.0*depth_normalized).astype(np.uint16))
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"depth_{number}"] = True

    # NORMALS callback
    def normals_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            normals = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            normals = normals[:, :, :3]
            normals = normals[:, :, ::-1]

            normals[:, :, 2] = ((normals[:, :, 2] / 255.0) * 127.0 + 128.0).astype(normals.dtype)

            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"normals_{number}"], f"{saved_frame}.png"), cv2.cvtColor(normals, cv2.COLOR_RGB2BGR))
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"normals_{number}"] = True

    # SEMANTIC SEGMENTATION callback
    def semantic_segmentation_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            semantic_map = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            semantic_map = semantic_map[:, :, :3]
            semantic_map = semantic_map[:, :, ::-1]
            semantic_map = semantic_map[:, :, 0]            

            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"semantic_{number}"], f"{saved_frame}.png"), cv2.cvtColor(semantic_map, cv2.COLOR_GRAY2BGR))
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"semantic_{number}"] = True

    # LIDAR
    def calculate_lidar_points_per_second(lidar_freq, lidar_channels, h_angle_res_degree):
        return round(lidar_freq * lidar_channels * (360.0 / h_angle_res_degree))

    lidar_freq_list = [CARLA_FPS] * 4
    lidar_channels_list = [16.0, 32.0, 64.0, 128.0]
    h_angle_res_degree_list = [0.18] * 4

    assert len(lidar_indexes) == len(lidar_freq_list) == len(lidar_channels_list) == len(h_angle_res_degree_list)

    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('noise_stddev', '0.0')
    lidar_bp.set_attribute('upper_fov', '31.0')
    lidar_bp.set_attribute('lower_fov', '-16.0')
    # lidar_bp.set_attribute('channels', f'{lidar_channels:.1f}')
    # lidar_bp.set_attribute('rotation_frequency', f'{lidar_freq:.1f}')
    # lidar_bp.set_attribute('points_per_second', str(calculate_lidar_points_per_second(lidar_freq, lidar_channels, h_angle_res_degree)))

    lidar_init_trans = carla.Transform(
        carla.Location(x=1.0, y=0, z=2.0),
        carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
    )

    camera_fov = 90.0

    # RGB CAMERAS
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("fov", str(camera_fov))
    camera_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    camera_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    # DEPTH CAMERAS
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("fov", str(camera_fov))
    depth_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    depth_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    # NORMAL CAMERAS
    normal_bp = bp_lib.find("sensor.camera.normals")
    normal_bp.set_attribute("fov", str(camera_fov))
    normal_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    normal_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    # SEMANTIC SEGMENTATION CAMERAS
    semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    semantic_bp.set_attribute("fov", str(camera_fov))
    semantic_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    semantic_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    K_camera = np.eye(3)
    K_camera[0, 2] = IMAGE_W / 2
    K_camera[1, 2] = IMAGE_H / 2
    K_camera[0, 0] = K_camera[1, 1] = IMAGE_W / (2.0 * math.tan(math.radians(camera_fov) / 2.0))

    np.savetxt(os.path.join(where_to_save, "K_camera.txt"), K_camera)

    transformations = []

    # Obvious CAMERAS
    transformations.append(carla.Transform(carla.Location(x=1.0, y=+0.0, z=2.0),
                                           carla.Rotation(pitch=0.0, roll=0, yaw=0)))
    transformations.append(carla.Transform(carla.Location(x=0.0, y=-1.0, z=2.0),
                                           carla.Rotation(pitch=-5.0, roll=0, yaw=90)))
    transformations.append(carla.Transform(carla.Location(x=-1.0, y=+0.0, z=2.0),
                                           carla.Rotation(pitch=0.0, roll=0, yaw=180)))
    transformations.append(carla.Transform(carla.Location(x=+0.0, y=1.0, z=2.0),
                                           carla.Rotation(pitch=-5.0, roll=0, yaw=270)))
    
    #TODO: get extrinsics from Carla

    sensors = {}

    for i in range(len(lidar_indexes)):
        lidar_freq = lidar_freq_list[i]
        lidar_channels = lidar_channels_list[i]
        h_angle_res_degree = h_angle_res_degree_list[i]

        lidar_bp.set_attribute('channels', f'{lidar_channels:.1f}')
        lidar_bp.set_attribute('rotation_frequency', f'{lidar_freq:.1f}')
        lidar_bp.set_attribute('points_per_second', str(calculate_lidar_points_per_second(lidar_freq, lidar_channels, h_angle_res_degree)))

        sensors[f"lidar_{lidar_indexes[i]}"] = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)

    for i in cameras_indexes:
        sensors[f"rgb_{i}"] = world.spawn_actor(camera_bp, transformations[i], attach_to=hero)
        sensors[f"depth_{i}"] = world.spawn_actor(depth_bp, transformations[i], attach_to=hero)
        sensors[f"normals_{i}"] = world.spawn_actor(normal_bp, transformations[i], attach_to=hero)
        sensors[f"semantic_{i}"] = world.spawn_actor(semantic_bp, transformations[i], attach_to=hero)

    # Connect Sensor and Callbacks
    for i in lidar_indexes:
        sensors[f"lidar_{i}"].listen(lambda scan, j=i: lidar_callback(scan, j))

    for i in cameras_indexes:
        sensors[f"rgb_{i}"].listen(lambda image, j=i: rgb_callback(image, j))
        sensors[f"depth_{i}"].listen(lambda depth, j=i: depth_callback(depth, j))
        sensors[f"normals_{i}"].listen(lambda normals, j=i: normals_callback(normals, j))
        sensors[f"semantic_{i}"].listen(lambda semantic, j=i: semantic_segmentation_callback(semantic, j))

    lidar_folders_name = [f"lidar_{i}" for i in lidar_indexes]
    rgb_folders_name = [f"rgb_{i}" for i in cameras_indexes]
    depth_folders_name = [f"depth_{i}" for i in cameras_indexes]
    normal_folders_name = [f"normals_{i}" for i in cameras_indexes]
    semantic_folders_name = [f"semantic_{i}" for i in cameras_indexes]

    global PATHS
    
    for i in range(len(lidar_indexes)):
        PATHS[f"lidar_{lidar_indexes[i]}"] = os.path.join(where_to_save, lidar_folders_name[i])

    for i in range(len(cameras_indexes)):
        PATHS[f"rgb_{cameras_indexes[i]}"] = os.path.join(where_to_save, rgb_folders_name[i])
        PATHS[f"depth_{cameras_indexes[i]}"] = os.path.join(where_to_save, depth_folders_name[i])
        PATHS[f"normals_{cameras_indexes[i]}"] = os.path.join(where_to_save, normal_folders_name[i])
        PATHS[f"semantic_{cameras_indexes[i]}"] = os.path.join(where_to_save, semantic_folders_name[i])

    for key_path in PATHS:
        os.mkdir(PATHS[key_path])

    def cntrl_c(_, __):
        for i in lidar_indexes:
            sensors[f"lidar_{i}"].stop()
            sensors[f"lidar_{i}"].destroy()
        for i in cameras_indexes:
            sensors[f"rgb_{i}"].stop()
            sensors[f"rgb_{i}"].destroy()
            sensors[f"depth_{i}"].stop()
            sensors[f"depth_{i}"].destroy()
            sensors[f"normals_{i}"].stop()
            sensors[f"normals_{i}"].destroy()
            sensors[f"semantic_{i}"].stop()
            sensors[f"semantic_{i}"].destroy()
        exit()


    signal.signal(signal.SIGINT, cntrl_c)

    # Let's Run Some Carla's Step to let everything be set up
    global DISABLE_ALL_SENSORS
    global KEEP_GPS
    global STARTING_FRAME
    KEEP_GPS = False
    DISABLE_ALL_SENSORS = True
    for _ in tqdm(range(10), desc=color_info_string("Warming Up...")):
        you_can_tick_event.set()
        world_snapshot = world.wait_for_tick()
        STARTING_FRAME = world_snapshot.frame
        time.sleep(0.3)

    time.sleep(3)
    STARTING_FRAME += 1
    DISABLE_ALL_SENSORS = False

    you_can_tick_event.set()
    for _ in tqdm(range(how_many_frames * AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE),
                  desc=color_info_string("Taking data...")):
        world_snapshot = world.wait_for_tick()
        if (world_snapshot.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            while True:
                if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR.values()) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR):
                    break
            # print("Obtained all the sensors data! B")
        you_can_tick_event.set()
        for key in ALREADY_OBTAINED_DATA_FROM_SENSOR:
            ALREADY_OBTAINED_DATA_FROM_SENSOR[key] = False

    finished_taking_data_event.set()
