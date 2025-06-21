import sys
import numpy as np

from transformation import r2rpy
from snapbot_env import SnapbotGymClass

class RunningSideways(SnapbotGymClass):
    """ 
        Snapbot gym for running sideways to target
        - Measured based on the time taken to reach the designated goal coordinate (shorter time results in a higher score)
        - Minimize time taken to reach the lateral goal
    """
    def __init__(self, env, HZ=50, history_total_sec=2.0, history_intv_sec=0.1, VERBOSE=True, target_x=0.0, target_y=50.0):
        """
            Initialize
            Args:
                target_x: x축 목표 좌표 (기본값: 0.0)
                target_y: y축 목표 좌표 (기본값: 2.0)
        """

        # 목표 좌표 설정
        self.target_x = target_x  # x축 목표
        self.target_y = target_y  # y축 목표
        self.target_distance = np.sqrt(target_x**2 + target_y**2)  # 목표까지의 거리
        
        # call parent class initialization
        super().__init__(env, HZ, history_total_sec, history_intv_sec, VERBOSE)
        self.name = 'RunningSideways'
        
        # 리워드 계수들
        self.k_speed = 20.0      # y_forward 보상 계수
        self.k_progress = 15.0   # 진행도 보상 계수
        self.k_accuracy = 10.0   # 정확도 보상 계수
        self.k_stability = 1.0  # 안정성 보상 계수
        self.k_time_penalty = -0.1  # 시간 페널티 (시간이 지날수록 보상 감소)
        
        self.k_x_penalty = -5.0 
        self.k_heading = 10.0 
        self.k_lane = -20.0
        self.k_collision = -10.0
        self.k_survive = 0.01
        self.k_rollover = -10.0  
        
        self.start_y = 0.0   
        self.max_progress = 0.0
        self.reached_target = False 
        self.start_time = 0.0 
        self.completion_time = np.inf 

        
    def step(self,a,max_time=np.inf):
        """
            Step forward
        """
        # Increse tick
        self.tick = self.tick + 1
        
        # Previous torso position and yaw angle in degree
        p_torso_prev       = self.env.get_p_body('torso')
        R_torso_prev       = self.env.get_R_body('torso')
        yaw_torso_deg_prev = np.degrees(r2rpy(R_torso_prev)[2])
        
        # Run simulation for 'mujoco_nstep' steps
        self.env.step(ctrl=a,nstep=self.mujoco_nstep)
        
        # Current torso position and yaw angle in degree
        p_torso_curr       = self.env.get_p_body('torso')
        R_torso_curr       = self.env.get_R_body('torso')
        yaw_torso_deg_curr = np.degrees(r2rpy(R_torso_curr)[2])
        
        # Compute the done signal
        ROLLOVER = (np.dot(R_torso_curr[:,2],np.array([0,0,1]))<0.0)
        if (self.get_sim_time() >= max_time) or ROLLOVER:
            d = True
        else:
            d = False
        
        y_diff = p_torso_curr[1] - p_torso_prev[1]  # y-directional displacement
        y_velocity = y_diff / self.dt
        
        x_diff = p_torso_curr[0] - p_torso_prev[0]
        x_velocity = x_diff / self.dt
        r_x_penalty = self.k_x_penalty * abs(x_velocity)
        
        current_y = p_torso_curr[1]
        current_x = p_torso_curr[0]
        distance_to_target = np.sqrt((self.target_x - current_x)**2 + (self.target_y - current_y)**2)
        r_accuracy = self.k_accuracy * (1.0 - min(distance_to_target / self.target_distance, 1.0))
        
        progress = (current_y - self.start_y) / self.target_y
        delta = progress - self.max_progress
        if delta > 0:
            r_progress = self.k_progress * delta
            self.max_progress = progress
        else:
            r_progress = 0.0
        
        target_reached_threshold = 0.2
        if not self.reached_target and distance_to_target < target_reached_threshold:
            self.reached_target = True
            self.completion_time = self.get_sim_time() - self.start_time
            r_target_reached = 50.0
        else:
            r_target_reached = 0.0
        
        current_time = self.get_sim_time() - self.start_time
        r_time_penalty = self.k_time_penalty * current_time
        
        torso_height = p_torso_curr[2]
        target_height = 0.15 
        height_error = abs(torso_height - target_height)
        r_stability = self.k_stability * (1.0 - min(height_error / target_height, 1.0))
        
        # self-collision check
        p_contacts,f_contacts,geom1s,geom2s,_,_ = self.env.get_contact_info(must_exclude_prefix='floor')
        if len(geom1s) > 0:
            SELF_COLLISION = 1
            r_collision = self.k_collision
        else:
            SELF_COLLISION = 0
            r_collision = 0.0
            
        # survival reward (전복 시 페널티, 그렇지 않으면 작은 보상)
        r_survive = self.k_rollover if ROLLOVER else self.k_survive
        
        # heading reward (앞쪽 방향을 향하도록) - 사이드워크를 위해 수정
        heading_vec = R_torso_curr[:,0]  # x direction (로봇이 바라보는 방향)
        target_heading = np.array([1, 0, 0])
        r_heading = self.k_heading * np.dot(heading_vec, target_heading)
        r_sideways_movement = self.k_speed * max(y_velocity, 0.0)
        
        # lane keeping (x 방향에서 목표 x 위치로 유도)
        epsilon = 0.05
        x_dev = abs(current_x - self.target_x)
        r_lane = self.k_lane * ((x_dev - epsilon)**2) if x_dev > epsilon else 0.0

        # compute total reward
        r = np.array(
            r_x_penalty +
            r_accuracy +
            r_progress +
            r_target_reached +
            r_time_penalty +
            r_stability +
            r_collision + 
            r_survive +
            r_heading +
            r_sideways_movement +
            r_lane
        )
        # clipping
        r = np.clip(r, -8000, 8000)
        
        # accumulate state history
        self.accumulate_state_history()
        # get observation
        o_prime = self.get_observation()
        # info
        info = {}
        # return
        return o_prime, r, d, info
    
    def reset(self):
        # call parent reset
        o = super().reset()
        
        # reset sideways running specific variables
        self.start_y = 0.0
        self.max_progress = 0.0
        self.reached_target = False
        self.start_time = self.get_sim_time()
        self.completion_time = np.inf

        return o
