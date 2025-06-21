
import sys
import numpy as np

from transformation import r2rpy
from snapbot_env import SnapbotGymClass

class StandingLongJump(SnapbotGymClass):
    def __init__(self, env, HZ=50, history_total_sec=2.0, history_intv_sec=0.1, VERBOSE=True):
        # call parent class initialization
        super().__init__(env, HZ, history_total_sec, history_intv_sec, VERBOSE)
        self.name += 'StandingLongJump'

        # Contact and jump state variables
        self.prev_contact = True
        self.in_air = False
        self.is_crouched = False
        self.ground_height = None
        self.min_crouch_height = float('inf')
        
        # Jump metrics
        self.air_time = 0.0
        self.air_start_x = 0.0
        self.jump_start_x = 0.0
        self.jump_distance = 0.0
        self.max_jump_distance = 0.0
        self.jump_peak_height = 0.0
        
        # Thresholds and optimal values
        self.optimal_crouch_height = 0.1
        self.optimal_jump_angle = np.pi/4
        self.crouch_threshold = 0.1
        
        # Reward coefficients
        self.k_preparation = 15.0    # 웅크리기 보상
        self.k_extension = 20.0      # 폄 동작 보상
        self.k_takeoff = 30.0        # 이륙 순간 보상
        self.k_airtime = 15.0        # 체공 시간 보상
        self.k_jump_dist = 50.0      # 점프 거리 보상
        self.k_landing_pose = 20.0   # 착지 자세 보상
        self.k_angle = 15.0          # 점프 각도 보상

    def get_state(self):
        """
            Get state (33)
            : Current state consists of 
                1) current joint position (8)
                2) current joint velocity (8)
                3) torso rotation (9)
                4) torso height (1)
                5) torso y value (1)
                6) contact info (8)
        """
        # Joint position
        qpos = self.env.data.qpos[self.env.ctrl_qpos_idxs] # joint position
        # Joint velocity
        qvel = self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        # Torso rotation matrix flattened
        R_torso_flat = self.env.get_R_body(body_name='torso').reshape((-1)) # torso rotation
        # Torso height
        p_torso = self.env.get_p_body(body_name='torso') # torso position
        torso_height = np.array(p_torso[2]).reshape((-1))
        # Torso y value
        torso_y_value = np.array(p_torso[1]).reshape((-1))
        # Contact information
        contact_info = np.zeros(self.env.n_sensor)
        contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
        contact_info[contact_idxs] = 1.0 # 1 means contact occurred
        # Concatenate information
        state = np.concatenate([
            qpos,
            qvel/10.0, # scale
            R_torso_flat,
            torso_height,
            torso_y_value,
            contact_info
        ])
        return state
    
    def get_observation(self):
        """
            Get observation 
        """
        
        # Sparsely accumulated history vector 
        state_history_sparse = self.state_history[self.history_ticks,:]
        
        # Concatenate information
        obs = np.concatenate([
            state_history_sparse
        ])
        return obs.flatten()


    def step(self,a,max_time=np.inf):
        """
            Step forward
        """
        # Increase tick
        self.tick = self.tick + 1
        
        # Previous torso position and yaw angle in degree
        p_torso_prev = self.env.get_p_body('torso')
        R_torso_prev = self.env.get_R_body('torso')
        yaw_torso_deg_prev = np.degrees(r2rpy(R_torso_prev)[2])
        
        # Run simulation for 'mujoco_nstep' steps
        self.env.step(ctrl=a,nstep=self.mujoco_nstep)
        
        # Current torso position and yaw angle in degree
        p_torso_curr = self.env.get_p_body('torso')
        R_torso_curr = self.env.get_R_body('torso')
        yaw_torso_deg_curr = np.degrees(r2rpy(R_torso_curr)[2])
        
        # Initialize ground height if not set
        if self.ground_height is None:
            self.ground_height = p_torso_curr[2]
        
        # Compute velocities
        velocity = (p_torso_curr - p_torso_prev) / self.dt
        vertical_velocity = velocity[2]
        horizontal_velocity = velocity[0]
        
        # Compute the done signal
        ROLLOVER = (np.dot(R_torso_curr[:,2],np.array([0,0,1]))<0.0)
        if (self.get_sim_time() >= max_time) or ROLLOVER:
            d = True
        else:
            d = False
        
        # Check contact status
        sensors = self.env.get_sensor_values(self.env.sensor_names)
        contacts_curr = (sensors > 0.2)
        has_contact = np.any(contacts_curr)
        
        # Initialize rewards
        r_preparation = 0.0    # 웅크리기 보상
        r_extension = 0.0      # 폄 동작 보상
        r_takeoff = 0.0        # 이륙 보상
        r_jump = 0.0          # 점프 관련 보상
        r_landing_pose = 0.0   # 착지 자세 보상
        r_angle = 0.0         # 점프 각도 보상
        
        # 1. 웅크리기 단계
        if has_contact and not self.in_air:
            current_height = p_torso_curr[2]
            if current_height < self.min_crouch_height:
                self.min_crouch_height = current_height
                crouch_depth = self.ground_height - current_height
                r_preparation = self.k_preparation * crouch_depth
                
                if crouch_depth > self.crouch_threshold:
                    self.is_crouched = True
            
            # 웅크린 상태에서 펴는 동작 보상
            elif self.is_crouched and vertical_velocity > 0:
                r_extension = self.k_extension * vertical_velocity
                
                if (current_height - self.ground_height) > -self.crouch_threshold:
                    self.is_crouched = False
        
        # 2. 이륙 감지 및 점프 시작
        if self.prev_contact and not has_contact:
            self.in_air = True
            self.air_time = 0.0
            self.air_start_x = p_torso_prev[0]
            self.jump_start_x = p_torso_prev[0]
            self.jump_distance = 0.0
            self.jump_peak_height = p_torso_curr[2]
            
            # 이륙 순간의 보상
            r_takeoff = self.k_takeoff * (
                max(vertical_velocity, 0.0) * 0.5 +    # 수직 속도
                max(horizontal_velocity, 0.0) * 0.5    # 수평 속도
            )
            
            self.is_crouched = False
        
        # 3. 공중에서의 보상
        if self.in_air:
            # 체공 시간 보상
            self.air_time += self.dt
            r_jump += self.k_airtime * self.dt
            
            # 거리 증가 보상
            self.jump_distance = p_torso_curr[0] - self.jump_start_x
            r_jump += self.k_jump_dist * max(self.jump_distance - self.max_jump_distance, 0.0)
            
            # 점프 각도 보상
            jump_angle = np.arctan2(velocity[2], velocity[0])
            angle_diff = abs(jump_angle - self.optimal_jump_angle)
            r_angle = self.k_angle * (1.0 - min(angle_diff/(np.pi/2), 1.0))
            
            # 최고 높이 갱신
            if p_torso_curr[2] > self.jump_peak_height:
                self.jump_peak_height = p_torso_curr[2]
        
        # 4. 착지 감지 및 보상
        if self.in_air and has_contact:
            # 최종 점프 거리에 대한 보상
            final_jump_distance = p_torso_curr[0] - self.jump_start_x
            r_jump += self.k_jump_dist * max(final_jump_distance, 0.0) * 2.0
            
            # 최대 기록 갱신 보상
            if final_jump_distance > self.max_jump_distance:
                r_jump += self.k_jump_dist * final_jump_distance
                self.max_jump_distance = final_jump_distance
            
            # 착지 자세 보상
            z_alignment = np.dot(R_torso_curr[:,2], np.array([0,0,1]))
            r_landing_pose = self.k_landing_pose * z_alignment
            
            # 상태 리셋
            self.in_air = False
            self.min_crouch_height = float('inf')
        
        # check self-collision (excluding 'floor')
        p_contacts,f_contacts,geom1s,geom2s,_,_ = self.env.get_contact_info(must_exclude_prefix='floor')
        if len(geom1s) > 0: # self-collision occurred
            r_collision = -10.0
        else:
            r_collision = 0.0
        
        # heading reward
        heading_vec = R_torso_curr[:,0]
        r_heading = 0.01*np.dot(heading_vec,np.array([1,0,0]))
        if r_heading < 0.0:
            r_heading = r_heading*100.0
            
        # lane deviation penalty
        lane_deviation = p_torso_curr[1]
        r_lane = -np.abs(lane_deviation)*0.1
        
        # survival reward
        r_survive = -40.0 if ROLLOVER else 0.05 \
            + (0.02 * vertical_velocity if vertical_velocity > 0 else 0)
        
        
        # 모든 보상을 합쳐서 최종 보상 계산
        r = np.array(
            r_collision +      # 충돌 패널티
            r_survive +        # 생존 보상
            r_heading +        # 방향 보상
            r_lane +          # 차선 유지 보상
            r_preparation +    # 웅크리기 보상
            r_extension +      # 폄 동작 보상
            r_takeoff +        # 이륙 보상
            r_jump +          # 점프 관련 보상
            r_landing_pose +   # 착지 자세 보상
            r_angle           # 점프 각도 보상
        )
        r = np.clip(r, -8000, 8000)
        
        # save contact info for next step
        self.prev_contact = has_contact
        
        # accumulate state history
        self.accumulate_state_history()
        # get observation
        o_prime = self.get_observation()
        # info
        info = {
            'jump_distance': self.jump_distance,
            'max_jump_distance': self.max_jump_distance,
            'air_time': self.air_time,
            'in_air': self.in_air,
            'jump_start_x': self.jump_start_x,
            'jump_peak_height': self.jump_peak_height,
            'is_crouched': self.is_crouched,
            'r_preparation': r_preparation,
            'r_extension': r_extension,
            'r_takeoff': r_takeoff,
            'r_jump': r_jump,
            'r_landing_pose': r_landing_pose,
            'r_angle': r_angle
        }
        return o_prime, r, d, info
    
    def reset(self):
        """
            Reset environment and initialize variables
        """
        # call parent reset
        o = super().reset()
        
        # long jump specific variables
        self.prev_contact = True
        self.in_air = False
        self.is_crouched = False
        self.ground_height = None
        self.min_crouch_height = float('inf')
        
        self.air_time = 0.0
        self.air_start_x = 0.0
        self.jump_start_x = 0.0
        self.jump_distance = 0.0
        self.jump_peak_height = 0.0
        # max_jump_distance는 에피소드 전체에서 유지 (최고 기록)
        
        return o
