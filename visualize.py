import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from main import AsymmetricBipedEnv

def main():
    print("Cargando modelo y entorno para visualización...")
    
    env_raw = AsymmetricBipedEnv(render_mode=None)
    env = DummyVecEnv([lambda: env_raw])
    
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
    except FileNotFoundError:
        print("Error: No se encontró 'vec_normalize.pkl'. Entrena el modelo primero ejecutando main.py")
        return
    
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load("asymmetric_biped_stable", device="cpu")
    except FileNotFoundError:
        print("Error: No se encontró 'asymmetric_biped_stable.zip'. Entrena el modelo primero.")
        return

    print("\n--- ABRIENDO VENTANA DE MUJOCO ---")
    print("Presiona Ctrl+C en la terminal para cerrar la simulación.")

    obs = env.reset()
    
    with mujoco.viewer.launch_passive(env_raw.model, env_raw.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            
            viewer.sync()
            
            if terminated[0]:
                obs = env.reset()
                
            time.sleep(0.02)

if __name__ == "__main__":
    main()