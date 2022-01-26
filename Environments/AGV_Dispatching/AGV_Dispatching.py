import csv
import datetime
import enum
import math
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
import pyautogui
from PIL import Image
from absl import logging

from Environments.AGV_Dispatching.Maps import get_map_params
from Environments.multiagentenv import MultiAgentEnv
from utils.DBManager import MariaManager


class DispatchingAttributes(enum.IntEnum):
    IDLE = 0
    DISTANCE = 1
    WAITINGTIME = 2
    TRAVELINGTIME = 3
    LINEBALANCING = 4


def VTE_is_on_process():
    counter = 0
    processes = ["Pinokio.exe", "Pinokio.ACS.exe"]
    # processes = ["Pinokio.exe"]
    for proc in psutil.process_iter():
        try:
            # 프로세스 이름, PID값 가져오기
            proc_name = proc.name()

            if proc_name in processes:
                counter += 1

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):  # 예외처리
            pass

    if counter == len(processes):
        return True
    else:
        return False


def VTE_kill_process():
    processes = ["Pinokio.exe", "Pinokio.ACS.exe"]
    for proc in psutil.process_iter():
        try:
            # 프로세스 이름, PID값 가져오기
            proc_name = proc.name()
            proc_id = proc.pid

            if proc_name in processes:
                parent_pid = proc_id  # PID
                parent = psutil.Process(parent_pid)  # PID 찾기
                for child in parent.children(recursive=True):  # 자식-부모 종료
                    child.kill()
                parent.kill()

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):  # 예외처리
            pass


def VTE_launch():
    try:
        VTE_kill_process()

        od = os.curdir
        os.chdir(r'D:\MnS\Pinokio.V2\Pinokio.VTE\Pinokio.VTE\bin\Debug')
        subprocess.Popen('Pinokio.exe',
                         shell=True, stdin=None, stdout=None, stderr=None,
                         close_fds=True)
        time.sleep(3)
        # png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-026.png")
        # rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
        pyautogui.moveTo(478, 144)
        time.sleep(0.2)
        pyautogui.click()
        pyautogui.click()

        SW_HIDE = 0
        info = subprocess.STARTUPINFO()
        info.dwFlags = subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = SW_HIDE

        os.chdir(r'D:\MnS\Pinokio.V2\Pinokio.ACS\Pinokio.ACS\bin\Debug')
        p2 = subprocess.Popen('Pinokio.ACS.exe',
                              stdin=None, stdout=None, stderr=None,
                              close_fds=True, startupinfo=info)
        time.sleep(5)

        # png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-025.png")
        # rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
        pyautogui.moveTo(167, 92)
        time.sleep(0.2)
        pyautogui.click()

        time.sleep(1)

        # png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-028.png")
        # rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
        pyautogui.moveTo(931, 17)  # 영상처리를 위한 캡처로 포지션 하드코딩함
        time.sleep(0.2)
        pyautogui.click()

        os.chdir(od)

        time.sleep(1)

        date = time.strftime('%m-%d-%Y %H%M%S', time.localtime(time.time()))
        filename = 'D:/MnS/Pinokio.RL/results/dummy/' + date + '.png'
        # 좌상
        # 1039
        # 198
        # 좌하
        # 1039
        # 687
        # 우하
        # 1590
        # 687
        # 우상
        # 1590
        # 198
        init_image = pyautogui.screenshot(filename, region=(1039, 198, 1590 - 1039, 687 - 198))
        return init_image
    except pyautogui.FailSafeException:
        return None


class AGVBasedFeature(MultiAgentEnv):
    def __init__(self,
                 map_name="",
                 end_time=432,
                 continuing_episode=False,
                 order_view=50,
                 target_production=500,
                 reward_scale=True,
                 state_last_action=True,
                 state_timestep_number=False,
                 state_normalize=8,
                 obs_instead_of_state=False,
                 seed=None,
                 heuristic_ai=False,
                 heuristic_rest=False,
                 replay_dir="",
                 replay_prefix="",
                 debug=False,
                 history=False,
                 ):
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        map_params['database'] = self.map_name
        self.DBMS = MariaManager(config=map_params)
        self.MapCoordinate = self.__getDataFrame(tb='map_node')
        self.LinkInfo = self.__getDataFrame(tb='map_node_link')
        self.Machines = self.__getDataFrame(tb='simulation_agv_info')
        self.MachineInfo = dict()

        self.n_agents = len(self.Machines)
        self.n_orders = order_view
        self.n_volumes = end_time
        self.episode_limit = map_params["limit"]
        self.score = 0
        self.start_time = time.time()

        # Observations and state
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        self.obs_instead_of_state = obs_instead_of_state

        # reward
        self.reward_scale = reward_scale
        self.target_production = target_production
        self.now_production = 0
        # n actions
        self.n_actions = len(DispatchingAttributes)

        # multi agent setting
        self.agents = {}
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.timeouts = 0
        self._run_config = None
        self._controller = None

        # History setting
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.history = {'Enable': history}
        if self.history['Enable'] is True:
            date = time.strftime('%m-%d-%Y %H%M%S', time.localtime(time.time()))
            self.history['Path'] = 'D:/MnS/Pinokio.RL/results/AGV_Dispatching/action_history' + date + '/'
            if not os.path.exists(self.history['Path']):
                os.makedirs(self.history['Path'])
            self.history['Data'] = {'Selected strategy': [],
                                    'Order ID': [],
                                    'Dispatched AGV': [],
                                    'Registered time': [],
                                    'Assigned time': [],
                                    'Complete orders': [],
                                    'AGV mileage': [],
                                    'Reward': [],
                                    }
            self.history['Action index'] = ['IDLE', 'DISTANCE', 'WAITING TIME', 'TRAVELING TIME', 'LINE BALANCING']
        self.debug = debug

        # heuristics
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_ai = False  # 일단 기능 꺼놓는다.
        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        # Utility
        # self.spfa_root = 0
        self.continuing_episode = continuing_episode
        self.spfa_dist_set = dict()

        self.green_channel = np.zeros((100850, 100850), dtype=np.uint8)

        links = self.LinkInfo.loc[self.LinkInfo.orientation == 'side', ['base_node', 'link_node']]
        for idx, link in links.iterrows():
            base_node = link.base_node
            link_node = link.link_node
            base_coord = self.MapCoordinate.loc[
                base_node == self.MapCoordinate.node_id, ['x_coordinate', 'y_coordinate']]
            link_coord = self.MapCoordinate.loc[
                link_node == self.MapCoordinate.node_id, ['x_coordinate', 'y_coordinate']]
            min_y = min([int(base_coord.y_coordinate), int(link_coord.y_coordinate)])
            max_y = max([int(base_coord.y_coordinate), int(link_coord.y_coordinate)])
            min_x = min([int(base_coord.x_coordinate), int(link_coord.x_coordinate)])
            max_x = max([int(base_coord.x_coordinate), int(link_coord.x_coordinate)])
            if (max_y - min_y) < (max_x - min_x):
                self.green_channel[max_y, range(min_x, max_x)] = 127
            else:
                self.green_channel[range(min_y, max_y), max_x] = 127

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions_int = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for index, action in enumerate(actions_int):
            if not self.get_avail_agent(index):
                continue
            if not self.heuristic_ai:
                strategy = self.get_agent_action(index, action)
            else:
                strategy, action_num = self.get_agent_action_heuristic(index, action)
                actions[index] = action_num

            if strategy is not None:
                # Send action request
                agent = self.get_agent_by_id(index)
                assign = ["selected_agv_id", "status", "assigned_date"]
                assign_time = time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))
                self.DBMS.DML.update(tb='mcs_order', columns=assign,
                                     values=[str(agent['AGV_ID']), str(1), '"' + str(assign_time) + '"'],
                                     cond='uid = ' + str(strategy['job id']))
                agent["Busy"] = 1

                if self.history['Enable'] is True:
                    data = self.history['Data']
                    data['Selected strategy'].append(self.history['Action index'][action])
                    data['Order ID'].append(strategy['job id'])
                    data['Dispatched AGV'].append(agent['AGV_ID'])
                    data['Registered time'].append(strategy['registered time'])
                    data['Assigned time'].append(assign_time)
                    data['Complete orders'].append(self.now_production)
                    data['AGV mileage'].append(agent['Mileage'])

        # reward
        view = self.DBMS.DML.select_all(tb='mcs_order', cond="completed_date != 'None'")
        columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
        production = pd.DataFrame(view, columns=columns)
        self.now_production = production.shape[0]
        reward = 1.0 - math.exp(-4 * (float(self.now_production) / self.target_production))  # Team reward

        mile_max = 0.0
        mile_min = math.inf
        for i in range(self.n_agents):
            temp_agent = self.agents[i]
            if mile_max < temp_agent['Mileage']:
                mile_max = temp_agent['Mileage']
            if mile_min > temp_agent['Mileage']:
                mile_min = temp_agent['Mileage']

        if not mile_min == 0.0 and mile_max == 0.0:
            reward = math.exp(-2 * 1 - (mile_min / mile_max))  # Agent Mile reward

        self.score += reward
        if self.history['Enable'] is True:
            data = self.history['Data']
            data['Reward'] = reward

        # terminated check
        terminated = False
        deadlock_time = time.time()
        while True and not terminated:
            # Update units
            self.__AGV_update()
            if VTE_is_on_process() is False:
                terminated = True
            if time.time() - self.start_time > self.n_volumes:
                terminated = True
            if time.time() - deadlock_time > 20:
                if self.history['Enable'] is True:
                    date = time.strftime('%m-%d-%Y %H%M%S', time.localtime(time.time()))
                    filename = self.history['Path'] + date + '.png'
                    pyautogui.screenshot(filename, region=(1039, 198, 1590 - 1039, 687 - 198))
                terminated = True
                reward = 0

            if terminated:
                view = self.DBMS.DML.select_all(tb='mcs_order', cond="completed_date != 'None'")
                columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
                production = pd.DataFrame(view, columns=columns)
                self.now_production = production.shape[0]

                if self.history['Enable'] is True:
                    data = self.history['Data']
                    output = pd.DataFrame(data)
                    filename = self.history['Path'] + 'action_history(' + str(self._episode_count) + ').csv'
                    output.to_csv(filename, sep=',')
                filename = self.history['Path'] + 'Result.csv'

                if os.path.isfile(filename):
                    f = open(filename, 'a', newline='')
                else:
                    f = open(filename, 'w', newline='')

                wr = csv.writer(f)
                wr.writerow([production.shape[0], self.score])
                f.close()

                self._episode_count += 1
                break

            pass_flag = False
            for index, _ in enumerate(actions_int):
                if self.get_avail_agent(index):
                    pass_flag = True
                    break
            if not pass_flag:
                continue

            view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
            columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
            idle_order_set = pd.DataFrame(view, columns=columns)
            orders = idle_order_set.head(3)

            if orders.shape[0] == 0:
                time.sleep(0.1)
            else:
                break

        return reward, terminated, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]

        # 224x224
        init_image = pyautogui.screenshot('temp.png', region=(1039, 198, 1590 - 1039, 687 - 198))
        view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
        columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
        idle_order_set = pd.DataFrame(view, columns=columns)
        from_nodes = idle_order_set.loc[:, ['from_node', 'uid']]
        min_y = 18715
        base_y = 49435 - 18715
        min_x = 64500
        base_x = 100850 - 64500
        current_date = datetime.datetime.now()
        red_channel = np.zeros((100850, 100850), dtype=np.uint8)
        for idx, iter_node in from_nodes.iterrows():
            coord = self.MapCoordinate.loc[int(iter_node.from_node) == self.MapCoordinate.node_id,
                                           ['x_coordinate', 'y_coordinate']]
            reg_date = idle_order_set.loc[iter_node.uid == idle_order_set.uid, 'registration_date']

            date_diff = current_date - reg_date
            sec = date_diff[idx].seconds
            red_channel[int(coord.y_coordinate), int(coord.x_coordinate)] = int(math.exp(-0.1 * sec) * 255.0)

        blue_channel = np.zeros((100850, 100850), dtype=np.uint8)
        for idx, val in enumerate(self.Machines.agv_id.tolist()):
            machine = self.MachineInfo[str(val)]
            names = ['MOVE', 'MOVE_NEXT']
            for n in range(len(names)):
                node = machine.loc[machine.addressName == names[n], 'addressValue']
                converted = node.values[0]
                if converted != '':
                    coord = self.MapCoordinate.loc[int(converted) == self.MapCoordinate.node_id,
                                                   ['x_coordinate', 'y_coordinate']]
                    blue_channel[int(coord.y_coordinate), int(coord.x_coordinate)] = 128 + (127 * n)
        img_b, img_g, img_r = init_image

        Merged = Image.fromarray(blue_channel)
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """

        # load N-order info
        view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
        columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
        idle_order_set = pd.DataFrame(view, columns=columns)
        top_N_order = idle_order_set.head(self.n_orders)

        # agent attributes - Id, Minimum dist among orders, Maximum dist among orders, Mileage
        agent = self.get_agent_by_id(agent_id)
        agent_feats = np.zeros(5, dtype=np.float32)
        agent_feats[0] = agent['AGV_ID']
        agent_feats[1] = math.inf
        agent_feats[2] = 0
        agent_feats[3] = agent['Mileage']
        agent_feats[4] = agent['Busy']

        # extract 5 attributes Id, fromX, fromY, toX, toY
        order_feats = np.zeros((self.n_orders, 5), dtype=np.float32)
        dummy_count = 0
        for i in top_N_order.index:
            from_node = top_N_order.loc[i, 'from_node']
            to_node = top_N_order.loc[i, 'to_node']
            from_cdi = self.MapCoordinate.loc[self.MapCoordinate.node_id == int(from_node),
                                              ['x_coordinate', 'y_coordinate']]
            to_cdi = self.MapCoordinate.loc[self.MapCoordinate.node_id == int(to_node),
                                            ['x_coordinate', 'y_coordinate']]

            order_feats[dummy_count, 0] = top_N_order.loc[i, 'uid']
            order_feats[dummy_count, 1] = from_cdi.iat[0, 0]
            order_feats[dummy_count, 2] = from_cdi.iat[0, 1]
            order_feats[dummy_count, 3] = to_cdi.iat[0, 0]
            order_feats[dummy_count, 4] = to_cdi.iat[0, 1]

            for identify, machine in self.MachineInfo.items():
                cur_node = machine.loc[machine.addressName == 'CUR_NODE', 'addressValue']
                distance = self.__getDistance(cur_node.item(), from_node)
                if agent_feats[1] >= distance:
                    agent_feats[1] = distance
                if agent_feats[2] <= distance:
                    agent_feats[2] = distance

            dummy_count += 1
        for i in range(self.n_orders - dummy_count):
            order_feats[dummy_count, 0] = 0
            order_feats[dummy_count, 1] = 0
            order_feats[dummy_count, 2] = 0
            order_feats[dummy_count, 3] = 0
            order_feats[dummy_count, 4] = 0

        agent_obs = np.concatenate(
            (
                agent_feats.flatten(),
                order_feats.flatten(),
            )
        )

        return agent_obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        rtn = 5 + self.n_orders * 5
        return rtn

    def get_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
        return None

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents
        return None

    def get_avail_actions(self):
        # cannot choose no-op when alive
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        agent = self.get_agent_by_id(agent_id)
        avail_actions = [1] * self.n_actions
        if agent["Busy"]:
            for attribute in DispatchingAttributes:
                if not attribute == DispatchingAttributes.IDLE:
                    avail_actions[attribute] = 0
        else:
            avail_actions[DispatchingAttributes.IDLE] = 0
        return avail_actions

    def get_avail_agent(self, agent_id):
        agent = self.get_agent_by_id(agent_id)
        if agent["Busy"] == 1:
            return False
        else:
            return True

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        if self.continuing_episode is True:
            return
        self.init_agents()
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.now_production = 0
        self.score = 0

        if self.history['Enable'] is True:
            self.history['Data'] = {'Selected strategy': [],
                                    'Order ID': [],
                                    'Dispatched AGV': [],
                                    'Registered time': [],
                                    'Assigned time': [],
                                    'Complete orders': [],
                                    'AGV mileage': [],
                                    'Reward': [],
                                    }

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        # observation, state 초기화

        launch_check = VTE_launch()
        if launch_check is None:
            VTE_kill_process()
            return None
        self.start_time = time.time()
        self.__AGV_update()
        while True:
            view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
            columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
            idle_order_set = pd.DataFrame(view, columns=columns)
            orders = idle_order_set.head(3)

            if orders.shape[0] < 1:
                time.sleep(1)
            else:
                break
        return None

    def render(self):
        # Auto render
        return None

    def close(self):
        VTE_kill_process()

    def seed(self):
        return self._seed

    def save_replay(self):
        """Save a replay."""
        # prefix = self.replay_prefix or self.map_name
        # replay_dir = self.replay_dir or ""
        # replay_path = self._run_config.save_replay(
        #     self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        # logging.info("Replay saved at: %s" % replay_path)
        return None

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def init_agents(self):
        """Initialise the units."""
        self.agents = {}
        machine_id = self.Machines.agv_id.tolist()
        for i in range(self.n_agents):
            self.agents[i] = {"AGV_ID": machine_id[i], "Mileage": 0, "Busy": 0}

        return None

    def __AGV_update(self):
        machine_id = self.Machines.agv_id.tolist()
        agv_info = self.__getDataFrame(tb='tb_agv_control')
        busy_agvs = self.DBMS.DML.select(tb='mcs_order', col='selected_agv_id',
                                         cond="completed_date = 'None' AND assigned_date != 'None'")
        busy_agvs = pd.DataFrame(busy_agvs).values
        for idx, val in enumerate(machine_id):
            agent = self.agents[idx]
            self.MachineInfo[str(val)] = agv_info.loc[agv_info.agvid == val, :]

            if agent['AGV_ID'] in busy_agvs:
                agent["Busy"] = 1
            else:
                agent["Busy"] = 0

    def get_agent_by_id(self, index):
        """Get agent by ID."""
        return self.agents[index]

    def get_agent_action(self, index, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(index)
        if avail_actions[action] != 1:
            agent = self.get_agent_by_id(index)
            test = agent['Busy']
        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(index, action)

        agent = self.get_agent_by_id(index)
        view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
        columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
        idle_order_set = pd.DataFrame(view, columns=columns)
        top_N_order = idle_order_set.head(self.n_orders)
        if top_N_order.shape[0] == 0:
            return None

        min_dist = math.inf
        max_dist = 0
        strategies = {'Shortest': {'job id': 0, 'distance': 0, 'registered time': ''},
                      'Minimum waiting': {'job id': 0, 'distance': 0, 'registered time': ''},
                      'Minimum traveling': {'job id': 0, 'distance': 0, 'registered time': ''},
                      'Line balancing': {'job id': 0, 'distance': 0, 'registered time': ''},
                      }
        for i in top_N_order.index:
            from_node = top_N_order.loc[i, 'from_node']
            to_node = top_N_order.loc[i, 'to_node']

            for identify, machine in self.MachineInfo.items():
                cur_node = machine.loc[machine.addressName == 'CUR_NODE', 'addressValue']
                distance = self.__getDistance(cur_node.item(), from_node)
                if i == 0:
                    strategies['Minimum waiting'] = {'job id': top_N_order.loc[i, 'uid'],
                                                     'distance': distance,
                                                     'registered time': top_N_order.loc[i, 'registration_date']}
                    strategies['Minimum traveling'] = {'job id': top_N_order.loc[i, 'uid'],
                                                       'distance': distance,
                                                       'registered time': top_N_order.loc[i, 'registration_date']}
                if min_dist >= distance:
                    if distance < 1.0:
                        continue

                    strategies['Shortest'] = {'job id': top_N_order.loc[i, 'uid'],
                                              'distance': distance,
                                              'registered time': top_N_order.loc[i, 'registration_date']}
                    min_dist = distance
                if max_dist <= distance:
                    strategies['Line balancing'] = {'job id': top_N_order.loc[i, 'uid'],
                                                    'distance': distance,
                                                    'registered time': top_N_order.loc[i, 'registration_date']}
                    max_dist = distance
        if action == DispatchingAttributes.IDLE:
            action = random.choice(range(1, len(DispatchingAttributes)))

        if action == DispatchingAttributes.DISTANCE:
            strategy = strategies['Shortest']
            if self.debug:
                logging.debug("Agent {} strategy: Minimum distance".format(index))
        elif action == DispatchingAttributes.WAITINGTIME:
            strategy = strategies['Minimum waiting']
            if self.debug:
                logging.debug("Agent {} strategy: Minimum waiting time".format(index))
        elif action == DispatchingAttributes.TRAVELINGTIME:
            strategy = strategies['Minimum traveling']
            if self.debug:
                logging.debug("Agent {} strategy: Minimum traveling time".format(index))
        else:
            strategy = strategies['Line balancing']
            if self.debug:
                logging.debug("Agent {} strategy: line balancing".format(index))

        agent['Mileage'] += strategy['distance']
        return strategy

    def get_agent_action_heuristic(self, index, action):
        # 안쓸래
        return None, None

    def __getDataFrame(self, tb):
        view = self.DBMS.DML.select_all(tb=tb)
        columns = self.DBMS.DML.get_columns(tb=tb, on_tuple=False)
        return pd.DataFrame(view, columns=columns)

    def __getDistance(self, begin, terminal):
        if begin in self.spfa_dist_set and terminal in self.spfa_dist_set:
            begin_dist = self.spfa_dist_set[begin]
            terminal_dist = self.spfa_dist_set[terminal]
            dist = abs(terminal_dist - begin_dist)

        else:
            _, dist = self.__getSPFARouting(begin, terminal)

        return dist

    def __getSPFARouting(self, begin, terminal):
        visit = {begin: 0}
        route = {begin: [begin]}
        q = [begin]
        node = begin

        self.spfa_root = begin
        self.spfa_dist_set = visit

        while len(q) != 0:
            candidates = self.LinkInfo.loc[self.LinkInfo.base_node == int(node),
                                           ['link_id', 'link_node', 'weight']]

            for idx in candidates.index:
                temp_node = candidates.loc[idx, 'link_node']
                temp_node = str(temp_node)
                distance = candidates.loc[idx, 'weight']

                if temp_node in visit:
                    if distance + visit[node] > visit[temp_node]:
                        continue

                q.append(temp_node)
                visit[temp_node] = distance + visit[node]
                route[temp_node] = route[node] + [temp_node]

            node = q.pop()

        return route[terminal], visit[terminal]


class AGVBasedImage(AGVBasedFeature):
    def reset(self):
        """ Returns initial observations and states"""
        if self.continuing_episode is True:
            return
        self.init_agents()
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.now_production = 0
        self.score = 0

        if self.history['Enable'] is True:
            self.history['Data'] = {'Selected strategy': [],
                                    'Order ID': [],
                                    'Dispatched AGV': [],
                                    'Registered time': [],
                                    'Assigned time': [],
                                    'Complete orders': [],
                                    'AGV mileage': [],
                                    'Reward': [],
                                    }

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        # observation, state 초기화

        init_state = VTE_launch()
        if init_state is None:
            VTE_kill_process()
            return None
        self.start_time = time.time()
        self.__AGV_update()
        while True:
            view = self.DBMS.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
            columns = self.DBMS.DML.get_columns(tb='mcs_order', on_tuple=False)
            idle_order_set = pd.DataFrame(view, columns=columns)
            orders = idle_order_set.head(3)

            if orders.shape[0] < 1:
                time.sleep(1)
            else:
                break
        return init_state

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

# class JobBased(gym.Env):
#     def __init__(self, config: EnvContext):
#
#         self.score = 0
#         self.__MaxCandidate = 5
#         self.DB = MariaManager(config=config)
#         self.MapCoordinate = self.__getDataFrame(tb='tb_base_map_node')
#         self.LinkInfo = self.__getDataFrame(tb='tb_base_map_link')
#         self.__BISectionAugment__()
#         self.machines = self.__getDataFrame(tb='simulation_agv_info')
#         self.MachineInfo = dict()
#
#         # self.action_space = Discrete(len(self.machines))
#         self.action_space = Dict({'Machine': Discrete(len(self.machines)), 'Order': Discrete(5)})
#         # self.action_space = Box(low=[0, 0], high=[len(self.machines), 5], shape=(2,), dtype=np.int32)
#
#         self.observation_space = self.__GetObservationSpaces__()
#         self.obs = self.__GetInitObservation__()
#
#         self.__update()
#
#     def reset(self):
#         # Infinity AGV Environment (non-reset)
#
#         wait_flag = True
#         while wait_flag:
#             view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
#             columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
#             idle_order_set = pd.DataFrame(view, columns=columns)
#             top5_order = idle_order_set.head(5)
#
#             if top5_order.shape[0] != 0:
#                 wait_flag = False
#             else:
#                 time.sleep(0.5)
#
#         self.__update()
#         return self.__convert_obs()
#
#     def step(self, action):
#         reward = 0
#         done = False
#         assign = ["selected_agv_id", "status", "assigned_date"]
#         order = self.obs['Order-' + str(action['Order'])]
#         agv = self.obs['AGV-' + str(action['Machine'] + 1)]
#         if agv['Busy'] == 0:
#             assign_time = time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))
#
#             self.DB.DML.update(tb='mcs_order', columns=assign,
#                                values=[str(agv['AGV_ID']), str(1), '"' + str(assign_time) + '"'],
#                                cond='uid = ' + str(order['Id']))
#
#             agv['Busy'] = 1
#             if agv['Recommend'] == action['Order']:
#                 reward = 1
#             wait_flag = True
#             while wait_flag:
#                 view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
#                 columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
#                 idle_order_set = pd.DataFrame(view, columns=columns)
#                 top5_order = idle_order_set.head(5)
#
#                 if top5_order.shape[0] != 0:
#                     wait_flag = False
#                 else:
#                     time.sleep(0.5)
#         else:
#             reward = -1
#         self.__update()
#         return self.__convert_obs(), reward, done, {}
#
#     def render(self, mode="AGV"):
#         # mode-Centered rendering
#         mode = 'Field'
#
#     def __getDataFrame(self, tb):
#         view = self.DB.DML.select_all(tb=tb)
#         columns = self.DB.DML.get_columns(tb=tb, on_tuple=False)
#         return pd.DataFrame(view, columns=columns)
#
#     def __AGV_update(self):
#         machine_id = self.machines.agv_id.tolist()
#         agv_info = self.__getDataFrame(tb='tb_agv_control')
#         for val in machine_id:
#             self.MachineInfo[str(val)] = agv_info.loc[agv_info.agvId == val, :]
#             machineinfo = self.MachineInfo[str(val)]
#             busy = machineinfo.loc[machineinfo.addressName == 'Busy', 'addressValue']
#             if busy.item() != '1':
#                 machine = self.obs['AGV-' + str(val)]
#                 machine['Busy'] = 0
#
#     def __update(self):
#         view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
#         columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
#         idle_order_set = pd.DataFrame(view, columns=columns)
#         top5_order = idle_order_set.head(5)
#
#         self.__AGV_update()
#         machine_id = self.machines.agv_id.tolist()
#
#         dummy_count = 0
#         for i in top5_order.index:
#             from_node = top5_order.loc[i, 'from_node']
#             to_node = top5_order.loc[i, 'to_node']
#             from_cdi = self.MapCoordinate.loc[self.MapCoordinate.nodeName == from_node, ['x', 'y']]
#             to_cdi = self.MapCoordinate.loc[self.MapCoordinate.nodeName == to_node, ['x', 'y']]
#
#             order = self.obs['Order-' + str(dummy_count)]
#             order['Id'] = top5_order.loc[i, 'uid']
#             order['fromX'] = from_cdi.iat[0, 0]
#             order['fromY'] = from_cdi.iat[0, 1]
#             order['toX'] = to_cdi.iat[0, 0]
#             order['toY'] = to_cdi.iat[0, 1]
#
#             dist_list = []
#             for identify, machine in self.MachineInfo.items():
#                 cur_node = machine.loc[machine.addressName == 'CUR_NODE', 'addressValue']
#                 _, distance = self.__getSPFARouting__(cur_node.item(), from_node)
#                 dist_list.append(distance)
#             agv_idx = np.argmin(dist_list)
#
#             machine = self.obs['AGV-' + str(machine_id[agv_idx])]
#             if machine['Distance to job'] >= dist_list[agv_idx]:
#                 machine['Recommend'] = dummy_count
#                 machine['Distance to job'] = dist_list[agv_idx]
#
#             dummy_count += 1
#
#         # print(self.__precheck__(self.__convert_obs()))
#
#     def __convert_obs(self):
#         lim = ['Priority', 'Busy', 'Recommend']
#
#         temp = copy.deepcopy(self.obs)
#         for _, target in temp.items():
#             for key, value in target.items():
#                 if key not in lim:
#                     if not isinstance(value, list):
#                         convert = [value]
#                     else:
#                         convert = value
#                     target[key] = np.array(convert)
#
#         return temp
#
#     def precheck(self, sample):
#         if self.observation_space.contains(sample):
#             return True, []
#         else:
#             return False, self.observation_space.sample()
#
#     def __GetInitObservation__(self):
#         observation = dict()
#         machine_id = self.machines.agv_id.tolist()
#         for index in machine_id:
#             value = {'AGV_ID': index - 1,
#                      'Busy': 0,
#                      # 'CurrentX': 0.00,
#                      # 'CurrentY': 0.00,
#                      'Distance to job': INF,
#                      # 'Expected Left distance': 0.00,
#                      'Recommend': 0}
#             observation['AGV-' + str(index)] = OrderedDict(value)
#
#         for index in range(self.__MaxCandidate):
#             key = 'Order-' + str(index)
#             value = {'Id': 0, 'fromX': 0.00, 'fromY': 0.00, 'toX': 0.00, 'toY': 0.00}  # , 'Priority': 0}
#             observation[key] = OrderedDict(value)
#         return OrderedDict(observation)
#
#     def __GetObservationSpaces__(self):
#         observation = dict()
#         for index in range(self.__MaxCandidate):
#             key = 'Order-' + str(index)
#             value = {'Id': Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
#                      'fromX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      'fromY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      'toX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      'toY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64)}
#             # 'Priority': Discrete(self.__MaxCandidate + 1)}
#             observation[key] = Dict(value)
#
#         machine_id = self.machines.agv_id.tolist()
#         for index in machine_id:
#             value = {'AGV_ID': Box(low=0, high=max(machine_id), shape=(1,), dtype=np.int32),
#                      'Busy': Discrete(2),
#                      # 'CurrentX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      # 'CurrentY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      'Distance to job': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      # 'Expected Left distance': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
#                      'Recommend': Discrete(5)}
#             observation['AGV-' + str(index)] = Dict(value)
#         return Dict(observation)
#
#     def __BISectionAugment__(self):
#         bisection = self.LinkInfo.loc[self.LinkInfo.direction == 'BI', :]
#         augsection = bisection.copy()
#         name = augsection.linkName
#         reverse_name = copy.deepcopy(name)
#         reverse_name = reverse_name.tolist()
#         id = augsection.id.copy()
#         id = id.tolist()
#         to_node = bisection.fromNode.copy()
#         from_node = bisection.toNode.copy()
#
#         for idx, primal in enumerate(reverse_name):
#             reverse = 'link'
#             primal = primal[4:]
#             sep = primal.split('_')
#             reverse += sep[1] + '_' + sep[0]
#             augsection.loc[id[idx] - 1, ['id', 'linkName', 'fromNode', 'toNode']] = \
#                 [id[idx] + 500, reverse, from_node.iloc[idx], to_node.iloc[idx]]
#
#         temp1 = self.LinkInfo[self.LinkInfo['direction'] == 'BI']
#         temp2 = self.LinkInfo[self.LinkInfo['direction'] != 'BI']
#         self.LinkInfo = temp1.append(augsection, ignore_index=True).append(temp2, ignore_index=True)
#         self.__getAllDistance__()
#
#     def __getAllDistance__(self):
#         edges = []
#         for idx in self.LinkInfo.index:
#             from_node = self.LinkInfo.loc[idx, 'fromNode']
#             to_node = self.LinkInfo.loc[idx, 'toNode']
#             link_type = self.LinkInfo.loc[idx, 'linkType']
#             radius = self.LinkInfo.loc[idx, 'radius']
#             name = self.LinkInfo.loc[idx, 'linkName']
#             dist = self.__getDistance__(from_node, to_node, link_type, radius)
#             edges.append(dist)
#         self.LinkInfo['distance'] = edges
#
#     def __getSPFARouting__(self, begin, terminal):
#         visit = {begin: 0}
#         route = {begin: [begin]}
#         q = [begin]
#         node = begin
#         while len(q) != 0:
#             candidates = self.LinkInfo.loc[self.LinkInfo.fromNode == node,
#                                            ['linkName', 'toNode', 'linkType', 'radius', 'distance']]
#
#             for idx in candidates.index:
#                 temp_node = candidates.loc[idx, 'toNode']
#                 distance = candidates.loc[idx, 'distance']
#
#                 if temp_node in visit:
#                     if distance + visit[node] > visit[temp_node]:
#                         continue
#
#                 q.append(temp_node)
#                 visit[temp_node] = distance + visit[node]
#                 route[temp_node] = route[node] + [temp_node]
#
#             node = q.pop()
#
#         return route[terminal], visit[terminal]
#
#     def __getDistance__(self, a, b, link_type, radius=0):
#         if link_type == 'CURVED':
#             dist = math.pi * 0.5 * radius
#         else:
#             from_coordinate = self.MapCoordinate.loc[self.MapCoordinate.nodeName == a, ['x', 'y']]
#             to_coordinate = self.MapCoordinate.loc[self.MapCoordinate.nodeName == b, ['x', 'y']]
#             dist = np.linalg.norm([to_coordinate.iat[0, 0] - from_coordinate.iat[0, 0],
#                                    to_coordinate.iat[0, 1] - from_coordinate.iat[0, 1]])
#         return dist
