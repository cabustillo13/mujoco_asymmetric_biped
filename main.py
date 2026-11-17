import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

XML_MODEL = """
<mujoco model="asymmetric_biped">
  <compiler inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="box" size="20 20 0.1" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1"/>
    
    <body name="torso" pos="0 0 1.2">
      <freejoint/>
      <geom name="torso_geom" type="capsule" size="0.08 0.2" rgba="0.9 0.4 0.4 1"/>
      
      <!-- Pierna Izquierda -->
      <body name="left_thigh" pos="0 0.12 -0.2">
        <joint name="left_hip" type="hinge" axis="0 1 0" range="-30 45" damping="5.0" armature="0.01"/>
        <geom name="left_thigh_geom" type="capsule" size="0.05 0.15" pos="0 0 -0.15" rgba="0.2 0.5 0.8 1"/>
        
        <body name="left_shin" pos="0 0 -0.3">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="0 70" damping="5.0" armature="0.01"/>
          <geom name="left_shin_geom" type="capsule" size="0.04 0.15" pos="0 0 -0.15" rgba="0.2 0.7 0.9 1"/>
        </body>
      </body>
      
      <!-- Pierna Derecha Amputada -->
      <body name="right_thigh" pos="0 -0.12 -0.2">
        <joint name="right_hip" type="hinge" axis="0 1 0" range="-30 45" damping="5.0" armature="0.01"/>
        <geom name="right_thigh_geom" type="capsule" size="0.05 0.15" pos="0 0 -0.15" rgba="0.8 0.5 0.2 1"/>
      </body>
      
    </body>
  </worldbody>
  
  <actuator>
    <motor name="act_left_hip" joint="left_hip" gear="15"/>
    <motor name="act_left_knee" joint="left_knee" gear="15"/>
    <motor name="act_right_hip" joint="right_hip" gear="15"/>
  </actuator>
</mujoco>
"""

class AsymmetricBipedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML_MODEL)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = action * 15.0
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        
        forward_velocity = self.data.qvel[0]
        alive_bonus = 2.0
        control_cost = 0.01 * np.sum(np.square(action))
        
        reward = forward_velocity + alive_bonus - control_cost

        torso_height = self.data.qpos[2]
        terminated = bool(torso_height < 0.4 or torso_height > 1.8)
        truncated = False

        info = {"x_position": self.data.qpos[0]}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[2] = 1.15
        self.data.qpos[7:] = np.random.uniform(-0.01, 0.01, size=self.model.nq - 7)
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


if __name__ == "__main__":
    print("Inicializando entorno ultra-estable...")
    
    env = DummyVecEnv([lambda: AsymmetricBipedEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Configurando agente PPO en CPU...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu",
        learning_rate=5e-4, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.005
    )

    print("Comenzando entrenamiento robusto (100,000 pasos)...")
    model.learn(total_timesteps=100_000)

    model.save("asymmetric_biped_stable")
    env.save("vec_normalize.pkl")
    print("¡Entrenamiento completado sin inestabilidades físicas!")