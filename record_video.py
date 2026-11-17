import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from main import AsymmetricBipedEnv

def main():
    print("Preparando entorno para grabación con cámara de seguimiento...")
    
    env_raw = AsymmetricBipedEnv(render_mode=None)
    env = DummyVecEnv([lambda: env_raw])
    
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
    except FileNotFoundError:
        print("Error: No se encontró 'vec_normalize.pkl'. Ejecuta main.py primero.")
        return
    
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load("asymmetric_biped_stable", device="cpu")
    except FileNotFoundError:
        print("Error: No se encontró el modelo entrenado.")
        return

    # Resolución ajustada al búfer por defecto de MuJoCo (640x480)
    width, height = 640, 480
    renderer = mujoco.Renderer(env_raw.model, height=height, width=width)

    obs = env.reset()
    frames = []
    
    print("Grabando episodio con primer plano...")
    max_steps = 500
    
    # Configuración de la cámara de seguimiento
    cam = mujoco.MjvCamera()
    cam.azimuth = 90      # Ángulo lateral
    cam.elevation = -15   # Ligeramente inclinada hacia abajo
    cam.distance = 2.5    # Distancia de zoom

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        
        # Obtener la posición actual del torso del robot
        torso_pos = env_raw.data.body("torso").xpos
        
        # Mover el punto de mira de la cámara junto con el robot
        cam.lookat[0] = torso_pos[0]
        cam.lookat[1] = torso_pos[1]
        cam.lookat[2] = torso_pos[2]
        
        renderer.update_scene(env_raw.data, camera=cam)
        
        img = renderer.render()
        frames.append(img)
        
        if terminated[0]:
            print(f"El robot se cayó en el paso {step_idx}.")
            break

    print(f"Guardando video con primer plano en 'asymmetric_runner_close.mp4'...")
    imageio.mimsave("asymmetric_runner_close.mp4", frames, fps=10)
    print("¡Video generado con éxito! El sujeto ahora está en primer plano.")

if __name__ == "__main__":
    main()