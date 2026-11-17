import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from main import AsymmetricBipedEnv

def main():
    print("Preparando entorno para grabación extendida...")
    
    env_raw = AsymmetricBipedEnv(render_mode=None)
    env = DummyVecEnv([lambda: env_raw])
    
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
    except FileNotFoundError:
        print("Error: No se encontró 'vec_normalize.pkl'. Entrena el modelo primero.")
        return
    
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load("asymmetric_biped_stable", device="cpu")
    except FileNotFoundError:
        print("Error: No se encontró el modelo entrenado.")
        return

    width, height = 640, 480
    renderer = mujoco.Renderer(env_raw.model, height=height, width=width)

    obs = env.reset()
    frames = []
    
    # Aumentamos los pasos a 1500 (aprox. 30 segundos de video fluido)
    max_steps = 1500
    print(f"Grabando video extendido de {max_steps} pasos...")
    
    cam = mujoco.MjvCamera()
    cam.azimuth = 90      # Ángulo lateral
    cam.elevation = -15   # Ligeramente inclinada hacia abajo
    cam.distance = 2.8    # Distancia de zoom

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        
        # Cámara de seguimiento anclada al torso
        torso_pos = env_raw.data.body("torso").xpos
        
        # Mover el punto de mira de la cámara junto con el robot
        cam.lookat[0] = torso_pos[0]
        cam.lookat[1] = torso_pos[1]
        cam.lookat[2] = torso_pos[2]
        
        renderer.update_scene(env_raw.data, camera=cam)
        frames.append(renderer.render())
        
        # Si se cae, en lugar de romper el script, reiniciamos el entorno para que siga la secuencia
        if terminated[0]:
            obs = env.reset()

    print(f"Guardando video largo en 'asymmetric_runner_long.mp4'...")
    imageio.mimsave("asymmetric_runner_long.mp4", frames, fps=50)
    print("¡Video extendido generado con éxito!")

if __name__ == "__main__":
    main()