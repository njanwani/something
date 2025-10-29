import numpy as np


class Scenario:
    def __init__(self, keyframes, z_height):
        self.keyframes = keyframes
        self.z_height = z_height
    
    @classmethod
    def yaw_to_quat(cls, yaw):
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    @classmethod
    def quat_slerp(cls, q1, q2, s):
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            q = q1 + s * (q2 - q1)
            return q / np.linalg.norm(q)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * s
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = s0 * q1 + s1 * q2
        return q / np.linalg.norm(q)
    
    @classmethod
    def smooth_lerp(cls, v0, v1, s):
        """Smooth interpolation between two values."""
        s = np.clip(s, 0.0, 1.0)
        s = 3 * s**2 - 2 * s**3  # smoothstep
        return (1 - s) * v0 + s * v1
    
    def interpolate_pose(self, t):
        for i in range(len(self.keyframes) - 1):
            t0, pos0, yaw0 = self.keyframes[i]
            t1, pos1, yaw1 = self.keyframes[i + 1]
            if t0 <= t <= t1:
                s = (t - t0) / (t1 - t0)
                pos = (1 - s) * pos0 + s * pos1
                q0 = Scenario.yaw_to_quat(yaw0)
                q1 = Scenario.yaw_to_quat(yaw1)
                quat = Scenario.quat_slerp(q0, q1, s)
                pos = np.append(pos, self.z_height)
                return pos, quat
        pos, yaw = self.keyframes[-1][1], self.keyframes[-1][2]
        pos.append(self.z_height)
        return pos, Scenario.yaw_to_quat(yaw)
    
    
class Wave(Scenario):
    # -----------------------------------------------------
    # Default arm poses (from pointing code)
    # -----------------------------------------------------
    left_default = dict(
        shoulder1=0.855,
        shoulder2=-0.611,
        shoulder3=-0.244,
        elbow=-1.75
    )

    right_default = dict(
        shoulder1=0.75,
        shoulder2=-0.558,
        shoulder3=-0.489,
        elbow=-1.75
    )

    # Wave lifted pose
    wave_pose = dict(
        shoulder1=-1.26,
        shoulder2=-0.157,
        shoulder3=0.96,
        elbow=-0.7
    )
    def __init__(
        self,
        move_duration      = 2.0,
        turn_duration      = 0.5,
        pause_duration     = 3.0,
        speed_scale        = 4.0,
        turn_buffer        = 0.2,
        pre_wave_duration  = 0.3,
        post_wave_duration = 0.3,
        wave_duration      = 2.0,
        wave_freq          = 4.0,
        wave_amp           = 0.5,
        z_height           = 1.28,
    ):
        move_duration /= speed_scale
        turn_duration /= speed_scale
        pause_duration /= speed_scale
        keyframes = []
        t = 0.0

        keyframes.append((t, np.array([1.5, -3.0]), np.pi / 2))
        t += move_duration
        
        keyframes.append((t, np.array([1.5, 0.0]), np.pi / 2))
        t += turn_duration
        
        keyframes.append((t, np.array([1.5, 0.0]), -np.pi))
        t += pause_duration
        
        keyframes.append((t, np.array([1.5, 0.0]), -np.pi))
        t += turn_duration
        
        keyframes.append((t, np.array([1.5, 0.0]), np.pi / 2))
        t += move_duration
        
        keyframes.append((t, np.array([1.5, 1.5]), np.pi / 2))
        t += turn_duration
        
        keyframes.append((t, np.array([1.5, 1.5]), -np.pi))
        t += 2 * move_duration
        
        keyframes.append((t, np.array([-6.0, 1.5]), -np.pi))
        super().__init__(keyframes, z_height)
        
        
        # Right arm joints
        self.shoulder1_r = 22
        self.shoulder2_r = 23
        self.shoulder3_r = 24
        self.elbow_r     = 25

        # Left arm joints
        self.shoulder1_l = 26
        self.shoulder2_l = 27
        self.shoulder3_l = 28
        self.elbow_l     = 29
        
        turn_buffer = turn_buffer / speed_scale           # short delay after turn before waving
        self.pre_wave_duration = pre_wave_duration / speed_scale     # smooth lift before waving
        self.post_wave_duration = post_wave_duration / speed_scale    # smooth return after waving
        wave_duration = wave_duration / speed_scale         # duration of actual waving

        self.wave_freq = wave_freq                            # waves per second
        self.wave_amp = wave_amp                             # radians amplitude

        # Start waving *after* turning north and buffer delay
        self.wave_start = move_duration + turn_duration + turn_buffer
        self.wave_end = self.wave_start + wave_duration                
    
    def wave_motion(self, t, qpos):
        """Animate right arm wave during wave window with smooth transitions."""
        # Left arm always at default
        qpos[self.shoulder1_l] = Wave.left_default["shoulder1"]
        qpos[self.shoulder2_l] = Wave.left_default["shoulder2"]
        qpos[self.shoulder3_l] = Wave.left_default["shoulder3"]
        qpos[self.elbow_l] =     Wave.left_default["elbow"]
        # ---- Pre-wave transition ----
        if self.wave_start - self.pre_wave_duration <= t < self.wave_start:
            s = (t - (self.wave_start - self.pre_wave_duration)) / self.pre_wave_duration
            for k, v in Wave.wave_pose.items():
                qpos[self.__getattribute__(k + '_r')] = self.smooth_lerp(Wave.right_default[k], v, s)

        # ---- Waving ----
        elif self.wave_start <= t <= self.wave_end:
            phase = 2 * np.pi * self.wave_freq * (t - self.wave_start)
            qpos[self.shoulder1_r] = Wave.wave_pose["shoulder1"]
            qpos[self.shoulder2_r] = Wave.wave_pose["shoulder2"]
            qpos[self.shoulder3_r] = Wave.wave_pose["shoulder3"]
            elbow_min, elbow_max = -1.4, -0.2
            qpos[self.elbow_r] = (elbow_max + elbow_min) / 2 + ((elbow_max - elbow_min) / 2) * np.sin(phase)

        # ---- Post-wave return ----
        elif self.wave_end < t <= self.wave_end + self.post_wave_duration:
            s = (t - self.wave_end) / self.post_wave_duration
            for k, v in Wave.wave_pose.items():
                qpos[self.__getattribute__(k + '_r')] = self.smooth_lerp(v, Wave.right_default[k], s)

        # ---- Default (not waving) ----
        else:
            qpos[self.shoulder1_r] = Wave.right_default["shoulder1"]
            qpos[self.shoulder2_r] = Wave.right_default["shoulder2"]
            qpos[self.shoulder3_r] = Wave.right_default["shoulder3"]
            qpos[self.elbow_r]     = Wave.right_default["elbow"]

        return qpos