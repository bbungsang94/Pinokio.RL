from Environments.multiagentenv import MultiAgentEnv
from Environments.AGV_Dispatching.Maps import get_map_params


def kill_process():
    import psutil

    for proc in psutil.process_iter():
        try:
            # 프로세스 이름, PID값 가져오기
            proc_name = proc.name()
            proc_id = proc.pid

            if proc_name == "Pinokio.exe":
                parent_pid = proc_id  # PID
                parent = psutil.Process(parent_pid)  # PID 찾기
                for child in parent.children(recursive=True):  # 자식-부모 종료
                    child.kill()
                parent.kill()

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):  # 예외처리
            pass



class AGVBased(MultiAgentEnv):
    def __init__(self,
                 map_name="acs_dda_simulator_210324",
                 continuing_episode=False,
                 reward_alpha=10,
                 reward_beta=0,
                 reward_theta=0.5,
                 state_last_action=True,
                 state_timestep_number=False,
                 state_normalize=8,
                 seed=None,
                 heuristic_ai=False,
                 heuristic_rest=False,
                 debug=False,

                 ):
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_volumes = map_params["n_volumes"]
        self.episode_limit = map_params["limit"]

        # Observations and state
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number

        # n actions
        self.n_actions = 3

        # multi agent setting
        self.agents = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.timeouts = 0

    def launch(self):
        import os
        import subprocess
        import pyautogui
        import time
        from PIL import Image

        kill_process()

        od = os.curdir
        os.chdir(r'D:\MnS\Pinokio.V2\Pinokio.VTE\Pinokio.VTE\bin\Debug')
        p1 = subprocess.Popen('Pinokio.exe',
                              shell=True, stdin=None, stdout=None, stderr=None,
                              close_fds=True)
        time.sleep(3)
        png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-026.png")
        rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
        pyautogui.moveTo(rtn)
        pyautogui.click()
        pyautogui.click()

        time.sleep(3)
        png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-025.png")
        rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
        pyautogui.moveTo(rtn)
        pyautogui.click()

        return None

    def step(self, actions):
        """ Returns reward, terminated, info """
        return None

    def get_obs(self):
        """ Returns all agent observations in a list """
        return None

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return None

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return None

    def get_state(self):
        return None

    def get_state_size(self):
        """ Returns the shape of the state"""
        return None

    def get_avail_actions(self):
        return None

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return None

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return None

    def reset(self):
        """ Returns initial observations and states"""
        return None

    def render(self):
        return None

    def close(self):
        return None

    def seed(self):
        return None

    def save_replay(self):
        return None

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


class VTE(gym.Env):
    def __init__(self, config: EnvContext):

        self.score = 0
        self.__MaxCandidate = 5
        self.DB = MariaManager(config=config)
        self.MapCoordinate = self.__getDataFrame(tb='tb_base_map_node')
        self.LinkInfo = self.__getDataFrame(tb='tb_base_map_link')
        self.__BISectionAugment__()
        self.machines = self.__getDataFrame(tb='simulation_agv_info')
        self.MachineInfo = dict()

        # self.action_space = Discrete(len(self.machines))
        self.action_space = Dict({'Machine': Discrete(len(self.machines)), 'Order': Discrete(5)})
        # self.action_space = Box(low=[0, 0], high=[len(self.machines), 5], shape=(2,), dtype=np.int32)

        self.observation_space = self.__GetObservationSpaces__()
        self.obs = self.__GetInitObservation__()

        self.__update()

    def reset(self):
        # Infinity AGV Environment (non-reset)

        wait_flag = True
        while wait_flag:
            view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
            columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
            idle_order_set = pd.DataFrame(view, columns=columns)
            top5_order = idle_order_set.head(5)

            if top5_order.shape[0] != 0:
                wait_flag = False
            else:
                time.sleep(0.5)

        self.__update()
        return self.__convert_obs()

    def step(self, action):
        reward = 0
        done = False
        assign = ["selected_agv_id", "status", "assigned_date"]
        order = self.obs['Order-' + str(action['Order'])]
        agv = self.obs['AGV-' + str(action['Machine'] + 1)]
        if agv['Busy'] == 0:
            assign_time = time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))

            self.DB.DML.update(tb='mcs_order', columns=assign,
                               values=[str(agv['AGV_ID']), str(1), '"' + str(assign_time) + '"'],
                               cond='uid = ' + str(order['Id']))

            agv['Busy'] = 1
            if agv['Recommend'] == action['Order']:
                reward = 1
            wait_flag = True
            while wait_flag:
                view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
                columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
                idle_order_set = pd.DataFrame(view, columns=columns)
                top5_order = idle_order_set.head(5)

                if top5_order.shape[0] != 0:
                    wait_flag = False
                else:
                    time.sleep(0.5)
        else:
            reward = -1
        self.__update()
        return self.__convert_obs(), reward, done, {}

    def render(self, mode="AGV"):
        # mode-Centered rendering
        mode = 'Field'

    def __getDataFrame(self, tb):
        view = self.DB.DML.select_all(tb=tb)
        columns = self.DB.DML.get_columns(tb=tb, on_tuple=False)
        return pd.DataFrame(view, columns=columns)

    def __AGV_update(self):
        machine_id = self.machines.agv_id.tolist()
        agv_info = self.__getDataFrame(tb='tb_agv_control')
        for val in machine_id:
            self.MachineInfo[str(val)] = agv_info.loc[agv_info.agvId == val, :]
            machineinfo = self.MachineInfo[str(val)]
            busy = machineinfo.loc[machineinfo.addressName == 'Busy', 'addressValue']
            if busy.item() != '1':
                machine = self.obs['AGV-' + str(val)]
                machine['Busy'] = 0

    def __update(self):
        view = self.DB.DML.select_all(tb='mcs_order', cond='selected_agv_id = 0')
        columns = self.DB.DML.get_columns(tb='mcs_order', on_tuple=False)
        idle_order_set = pd.DataFrame(view, columns=columns)
        top5_order = idle_order_set.head(5)

        self.__AGV_update()
        machine_id = self.machines.agv_id.tolist()

        dummy_count = 0
        for i in top5_order.index:
            from_node = top5_order.loc[i, 'from_node']
            to_node = top5_order.loc[i, 'to_node']
            from_cdi = self.MapCoordinate.loc[self.MapCoordinate.nodeName == from_node, ['x', 'y']]
            to_cdi = self.MapCoordinate.loc[self.MapCoordinate.nodeName == to_node, ['x', 'y']]

            order = self.obs['Order-' + str(dummy_count)]
            order['Id'] = top5_order.loc[i, 'uid']
            order['fromX'] = from_cdi.iat[0, 0]
            order['fromY'] = from_cdi.iat[0, 1]
            order['toX'] = to_cdi.iat[0, 0]
            order['toY'] = to_cdi.iat[0, 1]

            dist_list = []
            for identify, machine in self.MachineInfo.items():
                cur_node = machine.loc[machine.addressName == 'CUR_NODE', 'addressValue']
                _, distance = self.__getSPFARouting__(cur_node.item(), from_node)
                dist_list.append(distance)
            agv_idx = np.argmin(dist_list)

            machine = self.obs['AGV-' + str(machine_id[agv_idx])]
            if machine['Distance to job'] >= dist_list[agv_idx]:
                machine['Recommend'] = dummy_count
                machine['Distance to job'] = dist_list[agv_idx]

            dummy_count += 1

        # print(self.__precheck__(self.__convert_obs()))

    def __convert_obs(self):
        lim = ['Priority', 'Busy', 'Recommend']

        temp = copy.deepcopy(self.obs)
        for _, target in temp.items():
            for key, value in target.items():
                if key not in lim:
                    if not isinstance(value, list):
                        convert = [value]
                    else:
                        convert = value
                    target[key] = np.array(convert)

        return temp

    def precheck(self, sample):
        if self.observation_space.contains(sample):
            return True, []
        else:
            return False, self.observation_space.sample()

    def __GetInitObservation__(self):
        observation = dict()
        machine_id = self.machines.agv_id.tolist()
        for index in machine_id:
            value = {'AGV_ID': index - 1,
                     'Busy': 0,
                     # 'CurrentX': 0.00,
                     # 'CurrentY': 0.00,
                     'Distance to job': INF,
                     # 'Expected Left distance': 0.00,
                     'Recommend': 0}
            observation['AGV-' + str(index)] = OrderedDict(value)

        for index in range(self.__MaxCandidate):
            key = 'Order-' + str(index)
            value = {'Id': 0, 'fromX': 0.00, 'fromY': 0.00, 'toX': 0.00, 'toY': 0.00}  # , 'Priority': 0}
            observation[key] = OrderedDict(value)
        return OrderedDict(observation)

    def __GetObservationSpaces__(self):
        observation = dict()
        for index in range(self.__MaxCandidate):
            key = 'Order-' + str(index)
            value = {'Id': Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
                     'fromX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     'fromY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     'toX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     'toY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64)}
            # 'Priority': Discrete(self.__MaxCandidate + 1)}
            observation[key] = Dict(value)

        machine_id = self.machines.agv_id.tolist()
        for index in machine_id:
            value = {'AGV_ID': Box(low=0, high=max(machine_id), shape=(1,), dtype=np.int32),
                     'Busy': Discrete(2),
                     # 'CurrentX': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     # 'CurrentY': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     'Distance to job': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     # 'Expected Left distance': Box(low=0.0, high=INF, shape=(1,), dtype=np.float64),
                     'Recommend': Discrete(5)}
            observation['AGV-' + str(index)] = Dict(value)
        return Dict(observation)

    def __BISectionAugment__(self):
        bisection = self.LinkInfo.loc[self.LinkInfo.direction == 'BI', :]
        augsection = bisection.copy()
        name = augsection.linkName
        reverse_name = copy.deepcopy(name)
        reverse_name = reverse_name.tolist()
        id = augsection.id.copy()
        id = id.tolist()
        to_node = bisection.fromNode.copy()
        from_node = bisection.toNode.copy()

        for idx, primal in enumerate(reverse_name):
            reverse = 'link'
            primal = primal[4:]
            sep = primal.split('_')
            reverse += sep[1] + '_' + sep[0]
            augsection.loc[id[idx] - 1, ['id', 'linkName', 'fromNode', 'toNode']] = \
                [id[idx] + 500, reverse, from_node.iloc[idx], to_node.iloc[idx]]

        temp1 = self.LinkInfo[self.LinkInfo['direction'] == 'BI']
        temp2 = self.LinkInfo[self.LinkInfo['direction'] != 'BI']
        self.LinkInfo = temp1.append(augsection, ignore_index=True).append(temp2, ignore_index=True)
        self.__getAllDistance__()

    def __getAllDistance__(self):
        edges = []
        for idx in self.LinkInfo.index:
            from_node = self.LinkInfo.loc[idx, 'fromNode']
            to_node = self.LinkInfo.loc[idx, 'toNode']
            link_type = self.LinkInfo.loc[idx, 'linkType']
            radius = self.LinkInfo.loc[idx, 'radius']
            name = self.LinkInfo.loc[idx, 'linkName']
            dist = self.__getDistance__(from_node, to_node, link_type, radius)
            edges.append(dist)
        self.LinkInfo['distance'] = edges

    def __getSPFARouting__(self, begin, terminal):
        visit = {begin: 0}
        route = {begin: [begin]}
        q = [begin]
        node = begin
        while len(q) != 0:
            candidates = self.LinkInfo.loc[self.LinkInfo.fromNode == node,
                                           ['linkName', 'toNode', 'linkType', 'radius', 'distance']]

            for idx in candidates.index:
                temp_node = candidates.loc[idx, 'toNode']
                distance = candidates.loc[idx, 'distance']

                if temp_node in visit:
                    if distance + visit[node] > visit[temp_node]:
                        continue

                q.append(temp_node)
                visit[temp_node] = distance + visit[node]
                route[temp_node] = route[node] + [temp_node]

            node = q.pop()

        return route[terminal], visit[terminal]

    def __getDistance__(self, a, b, link_type, radius=0):
        if link_type == 'CURVED':
            dist = math.pi * 0.5 * radius
        else:
            from_coordinate = self.MapCoordinate.loc[self.MapCoordinate.nodeName == a, ['x', 'y']]
            to_coordinate = self.MapCoordinate.loc[self.MapCoordinate.nodeName == b, ['x', 'y']]
            dist = np.linalg.norm([to_coordinate.iat[0, 0] - from_coordinate.iat[0, 0],
                                   to_coordinate.iat[0, 1] - from_coordinate.iat[0, 1]])
        return dist


def testcase():
    db_config = dict()
    db_config['host'] = '127.0.0.1'
    db_config['user'] = 'root'
    db_config['password'] = '151212kyhASH@'
    db_config['port'] = int(3306)
    db_config['database'] = 'lv2meblayer4'

    Env = VTE(db_config)
    sample = Env.reset()
    print(Env.precheck(sample))
    action = {'Machine': 1, 'Order': 2}
    sample, _, _, _ = Env.step(action)
    print(Env.precheck(sample))


if __name__ == '__main__':
    testcase()
