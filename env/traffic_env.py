import cityflow


class TrafficEnv:
    def __init__(self, config_file="config.json", tl_id="intersection_1_1"):
        self.config_file = config_file
        self.tl_id = tl_id
        self.engine = None
        self.current_phase = 0
        self.action_duration = 10   # key change: each action lasts 10 sim steps

        self.incoming_road_prefixes = [
            "road_0_1_0",  # west -> center
            "road_1_0_1",  # south -> center
            "road_2_1_2",  # east -> center
            "road_1_2_3",  # north -> center
        ]

        self.prev_total_waiting = 0.0

    def reset(self):
        self.engine = cityflow.Engine(self.config_file, thread_num=1)
        self.current_phase = 0
        self.prev_total_waiting = self.get_total_waiting()
        return self.get_state()

    def get_total_waiting(self):
        lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()
        return float(sum(lane_waiting_vehicle_count.values()))

    def get_total_vehicles(self):
        lane_vehicle_count = self.engine.get_lane_vehicle_count()
        return float(sum(lane_vehicle_count.values()))

    def get_state(self):
        lane_vehicle_count = self.engine.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.engine.get_lane_waiting_vehicle_count()

        vehicle_features = []
        waiting_features = []

        for prefix in self.incoming_road_prefixes:
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

        return vehicle_features + waiting_features + [total_vehicles, total_waiting, self.current_phase]

    def compute_reward(self):
        current_waiting = self.get_total_waiting()
        current_vehicles = self.get_total_vehicles()

        # Reward improvement, not just absolute congestion
        reward = (self.prev_total_waiting - current_waiting) - 0.05 * current_vehicles

        # clip to stabilize training
        reward = max(min(reward, 10.0), -10.0)

        self.prev_total_waiting = current_waiting
        return reward

    def step(self, action):
        self.current_phase = action
        self.engine.set_tl_phase(self.tl_id, action)

        # let the action have visible effect
        for _ in range(self.action_duration):
            self.engine.next_step()

        next_state = self.get_state()
        reward = self.compute_reward()
        done = False

        return next_state, reward, done