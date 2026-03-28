import cityflow


class TrafficEnv:
    def __init__(self, config_file="config.json", tl_id="intersection_1_1"):
        self.config_file = config_file
        self.tl_id = tl_id
        self.engine = None

    
    def reset(self):
        self.engine = cityflow.Engine(self.config_file, thread_num=1)
        self.current_phase = 0
        return self.get_state()

    # def get_state(self):
    #     lane_vehicle_count = self.engine.get_lane_vehicle_count()
    #     lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()

    #     return {
    #         "lane_vehicle_count": lane_vehicle_count,
    #         "lane_waiting_vehicle_count": lane_waiting_vehicle_count,
    #     }
    
    # def get_state(self):
    #     lane_vehicle_count = self.engine.get_lane_vehicle_count()
    #     lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()

    #     total_vehicles = sum(lane_vehicle_count.values())
    #     total_waiting = sum(lane_waiting_vehicle_count.values())

    #     return [total_vehicles, total_waiting] # will make DQN easier to handle the state as a vector instead of a dict

    def get_state(self):
        lane_vehicle_count = self.engine.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()

        incoming_road_prefixes = [
            "road_0_1_0",  # west -> center
            "road_1_0_1",  # south -> center
            "road_2_1_2",  # east -> center
            "road_1_2_3",  # north -> center
        ]

        vehicle_features = []
        waiting_features = []

        for prefix in incoming_road_prefixes:
            road_vehicle_total = 0
            road_waiting_total = 0

            for lane_id, count in lane_vehicle_count.items():
                if lane_id.startswith(prefix):
                    road_vehicle_total += count

            for lane_id, count in lane_waiting_vehicle_count.items():
                if lane_id.startswith(prefix):
                    road_waiting_total += count

            vehicle_features.append(road_vehicle_total)
            waiting_features.append(road_waiting_total)

        total_vehicles = sum(vehicle_features)
        total_waiting = sum(waiting_features)

        current_phase = self.current_phase

        return vehicle_features + waiting_features + [total_vehicles, total_waiting, current_phase]
    # above: sum by incoming road group
    # [
    # west_vehicle_count,
    # south_vehicle_count,
    # east_vehicle_count,
    # north_vehicle_count,
    # west_waiting_count,
    # south_waiting_count,
    # east_waiting_count,
    # north_waiting_count,
    # total_vehicles,
    # total_waiting,
    # current_phase
    # ]


    # def compute_reward(self):
    #     waiting = self.engine.get_lane_waiting_vehicle_count()
    #     total_waiting = sum(waiting.values())
    #     return -total_waiting  # gives reward=0
    
    # def compute_reward(self):
    #     lane_vehicle_count = self.engine.get_lane_vehicle_count()
    #     total_vehicles = sum(lane_vehicle_count.values())
    #     return -total_vehicles / 10.0
    
    def compute_reward(self):
        lane_vehicle_count = self.engine.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()

        total_vehicles = sum(lane_vehicle_count.values())
        total_waiting = sum(lane_waiting_vehicle_count.values())

        reward = -(total_waiting + 0.5 * total_vehicles) / 10.0
        return reward
    
    def step(self, action):
        self.current_phase = action
        self.engine.set_tl_phase(self.tl_id, action)
        self.engine.next_step()

        next_state = self.get_state()
        reward = self.compute_reward()
        done = False

        return next_state, reward, done