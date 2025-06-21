import sys
import numpy as np

from transformation import r2rpy
from snapbot_env import SnapbotGymClass

class HighJump(SnapbotGymClass):
    def __init__(self, env, HZ=50, history_total_sec=2.0, history_intv_sec=0.1, VERBOSE=True):
        # call parent class initialization
        super().__init__(env, HZ, history_total_sec, history_intv_sec, VERBOSE)
        self.name = 'high_jump'
        
        self.prev_contact = True    # 이전 스텝에서의 접촉 상태
        self.in_air = False         # 현재 공중에 있는지
        self.ground_height = None   # 바닥 높이 (초기 torso 높이로 설정)
        self.min_crouch_height = float('inf')  # 웅크렸을 때 최저 높이
        self.jump_peak_height = 0.0 # 현재 점프의 최대 높이
        self.is_crouched = False    # 웅크린 상태인지 여부
        self.crouch_threshold = 0.1 # 웅크린 것으로 간주할 높이 차이 임계값
        
        # Reward 계수들
        self.k_preparation = 10.0   # 웅크리기 보상 계수
        self.k_extension = 20.0     # 폄 동작 보상 계수
        self.k_takeoff = 40.0       # 점프 보상 계수
        self.k_height = 50.0        # 높이 보상 계수
        self.k_airtime = 30.0       # 공중 시간 보상 계수
        self.k_completion = 10.0    # 착지 완료 보상 계수

        # 공중 시간 관련 변수
        self.airtime_start = 0.0
        self.prev_height = 0.0      # 이전 높이 저장

    def get_state(self):
        """
            Get state (33) - 기존과 동일하게 유지
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
    
    def step(self,a,max_time=np.inf):
        """
            Step forward with High Jump specific rewards
        """
        # Increase tick
        self.tick = self.tick + 1
        
        # Previous torso position
        p_torso_prev = self.env.get_p_body('torso')
        
        # Run simulation for 'mujoco_nstep' steps
        self.env.step(ctrl=a,nstep=self.mujoco_nstep)
        
        # Current torso position
        p_torso_curr = self.env.get_p_body('torso')
        
        # Initialize ground height
        if self.ground_height is None:
            self.ground_height = p_torso_curr[2]
        
        # Check contact status
        sensors = self.env.get_sensor_values(self.env.sensor_names)
        contacts_curr = (sensors > 0.2)
        has_contact = np.any(contacts_curr)
        
        # Compute the done signal
        ROLLOVER = (np.dot(self.env.get_R_body('torso')[:,2],np.array([0,0,1]))<0.0)
        if (self.get_sim_time() >= max_time) or ROLLOVER:
            d = True
        else:
            d = False
        
        # 현재 상대 높이 계산
        current_height = p_torso_curr[2]
        relative_height = current_height - self.ground_height
        
        # 수직 속도 계산
        vertical_velocity = (p_torso_curr[2] - p_torso_prev[2]) / self.dt
        
        # 기본 보상 초기화
        r_preparation = 0.0    # 웅크리기 보상
        r_extension = 0.0      # 폄 동작 보상 (새로 추가)
        r_jump = 0.0          # 점프 보상
        r_landing = 0.0       # 착지 보상
        r_airtime = 0.0       # 공중 시간 보상
        r_completion = 0.0    # 완료 보상
        
        # 1. 웅크리기 단계 (상대 높이 사용)
        if has_contact and not self.in_air:
            if current_height < self.min_crouch_height:
                self.min_crouch_height = current_height
                crouch_depth = self.ground_height - current_height
                r_preparation = self.k_preparation * crouch_depth
                
                # 웅크린 상태 감지
                if crouch_depth > self.crouch_threshold:
                    self.is_crouched = True
            
            # 웅크린 상태에서 펴는 동작 보상
            elif self.is_crouched and vertical_velocity > 0:
                extension_speed = vertical_velocity
                r_extension = self.k_extension * extension_speed
                
                # 충분히 폈다면 웅크린 상태 해제
                if relative_height > -self.crouch_threshold:
                    self.is_crouched = False
        
        # 2. 점프 단계 (이륙 감지)
        if self.prev_contact and not has_contact:
            self.in_air = True
            self.jump_peak_height = current_height
            r_jump = self.k_takeoff * (max(vertical_velocity, 0.0) ** 2)
            self.airtime_start = self.get_sim_time()
            self.is_crouched = False  # 점프 시작시 웅크린 상태 초기화
        
        # 3. 공중에 있는 동안의 보상
        if self.in_air:
            # 공중 시간 보상 (매 스텝마다)
            current_airtime = self.get_sim_time() - self.airtime_start
            r_airtime = self.k_airtime * (current_airtime / 10.0)  # 정규화
            
            # 최고 높이 갱신 및 보상
            if current_height > self.jump_peak_height:
                height_improvement = current_height - self.jump_peak_height
                r_airtime += self.k_height * height_improvement  # 높이 증가에 대한 즉각적 보상
                self.jump_peak_height = current_height
        
        # 4. 착지 단계
        if self.in_air and has_contact:
            # 최종 점프 높이 계산 (상대 높이)
            final_jump_height = self.jump_peak_height - self.ground_height
            
            # 착지 보상 (제곱항 추가)
            r_landing = self.k_height * (final_jump_height + 0.5 * final_jump_height ** 2)
            
            # 완료 보상 (성공적인 착지)
            r_completion = self.k_completion * (
                final_jump_height +  # 높이 기여도
                0.3 * final_jump_height ** 2 +  # 높이에 대한 추가 보상
                (self.get_sim_time() - self.airtime_start)  # 공중 시간 기여도
            )
            
            # 상태 리셋
            self.in_air = False
            self.min_crouch_height = float('inf')
        
        # 접촉 상태 업데이트
        self.prev_contact = has_contact
        self.prev_height = current_height

        # check self-collision (excluding 'floor')
        p_contacts, f_contacts, geom1s, geom2s, _, _ =\
             self.env.get_contact_info(must_exclude_prefix='floor')
        if len(geom1s) > 0: # self-collision occurred
            SELF_COLLISION = 1
            r_collision    = -10.0
        else:
            SELF_COLLISION = 0
            r_collision    = 0.0
        
        # survival reward
        r_survive = -40.0 if ROLLOVER else 0.05 \
            + (0.02 * vertical_velocity if vertical_velocity > 0 else 0)
        
        # compute total reward
        r = np.array(
            r_collision +
            r_survive +
            r_preparation + 
            r_extension + 
            r_jump +
            r_landing +
            r_airtime +
            r_completion
        )
        r = np.clip(r, -8000, 8000)
        
        # accumulate state history
        self.accumulate_state_history()
        # get observation
        o_prime = self.get_observation()

        info = {}
        return o_prime, r, d, info
    
    def reset(self):
        """
            Reset with High Jump specific variables
        """
        # Reset base class
        super().reset()
        
        # Reset High Jump specific variables
        self.prev_contact = True
        self.in_air = False
        self.ground_height = None
        self.min_crouch_height = float('inf')
        self.jump_peak_height = 0.0
        self.airtime_start = 0.0
        self.prev_height = 0.0
        self.is_crouched = False    # 웅크린 상태 초기화
        
        # Get observation
        o = self.get_observation()
        return o
