import csv
import os
import tkinter as tk
import threading
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stable_baselines3 import DQN
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


def export_metrics_to_csv(file_path, controller_name, metrics):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Controller", *metrics.keys()])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Controller": controller_name, **metrics})


def calculate_metrics(angle_log, target=0, tolerance=0.05):
    mse = np.mean((np.array(angle_log) - target) ** 2)
    overshoot = np.max(np.abs(angle_log))  # Max deviation from vertical

    # Rise time: time to reach 90% of target deviation (for step input, here pole angle → 0)
    try:
        rise_time_idx = next(i for i, val in enumerate(np.abs(angle_log)) if val < 0.1)
    except StopIteration:
        rise_time_idx = -1

    # Settling time: time when pole remains within ±tolerance
    settled = False
    for i in range(len(angle_log)):
        window = angle_log[i:]
        if all(abs(x - target) < tolerance for x in window):
            settling_time_idx = i
            settled = True
            break
    settling_time_idx = settling_time_idx if settled else -1

    return {
        "MSE": round(mse, 5),
        "Overshoot": round(overshoot, 4),
        "Rise Time (steps)": rise_time_idx,
        "Settling Time (steps)": settling_time_idx if settled else "Unsettled"
    }

class ControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RL vs PID: Inverted Pendulum Comparison")
        self.root.geometry("900x700")

        # Create plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Pole Angle Over Time")
        self.ax.set_xlabel("Timestep")
        self.ax.set_ylabel("Pole Angle (radians)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.toolbar.pack()


        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.rl_button = tk.Button(button_frame, text="Run RL Controller", command=self.run_rl)
        self.rl_button.grid(row=0, column=0, padx=10)

        self.pid_button = tk.Button(button_frame, text="Run PID Controller", command=self.run_pid)
        self.pid_button.grid(row=0, column=1, padx=10)

        self.reset_button = tk.Button(button_frame, text="Reset Graph", command=self.reset_graph)
        self.reset_button.grid(row=0, column=2, padx=10)

        self.basic_pid_button = tk.Button(button_frame, text="Run Basic PID", command=self.run_basic_pid)
        self.basic_pid_button.grid(row=0, column=3, padx=10)


        # Toggle disturbance injection
        self.inject_disturbance = tk.BooleanVar(value=False)
        self.toggle_button = tk.Checkbutton(
            root,
            text="Inject Disturbance",
            variable=self.inject_disturbance,
            onvalue=True,
            offvalue=False
        )
        self.toggle_button.pack(pady=5)


        # Metrics output
        # Create a frame to hold text and scrollbar
        scroll_frame = tk.Frame(self.root)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget for metrics
        self.metrics_label = tk.Text(scroll_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=8, font=("Courier", 10))
        self.metrics_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.metrics_label.yview)


    def reset_graph(self):
        self.ax.clear()
        self.ax.set_title("Pole Angle Over Time")
        self.ax.set_xlabel("Timestep")
        self.ax.set_ylabel("Pole Angle (radians)")
        self.ax.grid(True)
        self.canvas.draw()
        self.metrics_label.delete("1.0", tk.END)



    def run_pid(self):
        def pid_thread():
            # PD gains for pole balancing (θ, θ̇)
            Kp_theta = 60.0
            Kd_theta = 10.0

            # PD gains for cart velocity control
            Kp_cart = 2.5

            env = gym.make("CartPole-v1", render_mode=None)
            obs, _ = env.reset()
            angles = []
            inject = self.inject_disturbance.get()
            disturb_step = 300  # You can change the timing as needed



            for step in range(500):
                x = obs[0]
                x_dot = obs[1]
                theta = obs[2]
                theta_dot = obs[3]

                # ---- PD 1: Pole Angle Correction ----
                # Want to reduce pole angle toward 0
                desired_cart_velocity = (
                    Kp_theta * theta + Kd_theta * theta_dot
                )


                # ---- PD 2: Cart Velocity Tracking ----
                # Try to make ẋ follow the desired cart velocity
                cart_error = desired_cart_velocity - x_dot
                control = Kp_cart * cart_error

                # Clamp output to avoid runaway
                control = np.clip(control, -10, 10)

                # Decide action
                action = 1 if control > 0 else 0

                # Step the environment
                obs, _, terminated, truncated, _ = env.step(action)

                # Inject a disturbance at the chosen timestep
                if inject and step == disturb_step:
                    print("⚠️ Injecting disturbance at step", step)
                    obs = list(obs)
                    obs[1] += 1.0
                    obs[2] += 0.1
                    obs = tuple(obs)


                # Update from new obs for termination check
                x = obs[0]
                theta = obs[2]
                angles.append(theta)

                print(
                    f"Step {step:3d} | θ = {theta:.3f}, θ̇ = {theta_dot:.3f}, x = {x:.3f}, ẋ = {x_dot:.3f}, "
                    f"desired_ẋ = {desired_cart_velocity:.3f}, control = {control:.2f}, action = {action}"
                )

                # Termination reason
                if abs(theta) > 0.209:
                    print(f"❌ Terminated: Pole angle out of bounds ({theta:.3f} rad)")
                elif abs(x) > 2.4:
                    print(f"❌ Terminated: Cart out of bounds ({x:.3f} m)")

                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    break

            env.close()
            self.ax.plot(angles, label="PID (2-PD)")
            if inject:
                self.ax.axvline(x=disturb_step, color='red', linestyle='--', label='Disturbance')
            self.ax.legend()
            self.canvas.draw()

            # Metrics
            metrics = calculate_metrics(angles)
            self.display_metrics("PID (2-PD)", metrics)
            export_metrics_to_csv("logs/performance_metrics.csv", "PID (2-PD)", metrics)



        # ✅ START the thread!
        threading.Thread(target=pid_thread).start()

    def run_basic_pid(self):
        def basic_pid_thread():
            Kp = 100.0
            Ki = 0.0
            Kd = 20.0
            integral = 0
            previous_error = 0

            env = gym.make("CartPole-v1", render_mode=None)
            obs, _ = env.reset()
            angles = []
            inject = self.inject_disturbance.get()
            disturb_step = 100

            for step in range(500):
                theta = obs[2]
                theta_dot = obs[3]

                # PID on theta only
                error = -theta
                integral += error
                derivative = error - previous_error
                previous_error = error

                control = Kp * error + Ki * integral + Kd * derivative
                control = np.clip(control, -10, 10)
                action = 1 if control > 0 else 0

                obs, _, terminated, truncated, _ = env.step(action)

                # Disturbance logic
                if inject and step == disturb_step:
                    print("⚠️ Basic PID disturbance injected at step", step)
                    obs = list(obs)
                    obs[1] += 1.0
                    obs[2] += 0.1
                    obs = tuple(obs)

                angles.append(obs[2])
                if terminated or truncated:
                    break

            env.close()
            self.ax.plot(angles, label="PID (θ only)")
            if inject:
                self.ax.axvline(x=disturb_step, color='red', linestyle='--', label='Disturbance')
            self.ax.legend()
            self.canvas.draw()

            metrics = calculate_metrics(angles)
            self.display_metrics("PID (θ only)", metrics)

        threading.Thread(target=basic_pid_thread).start()




    def run_rl(self):
        def rl_thread():
            model = DQN.load("models/best_model")
            env = gym.make("CartPole-v1", render_mode=None)
            obs, _ = env.reset()
            angles = []
            inject = self.inject_disturbance.get()
            disturb_step = 300  # You can change the timing as needed


            for step in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                if inject and step == disturb_step:
                    print("⚠️ RL Disturbance injected at step", step)
                    obs = list(obs)
                    obs[1] += 1.0
                    obs[2] += 0.1
                    obs = tuple(obs)


                angles.append(obs[2])

                if terminated or truncated:
                    break

            env.close()
            self.ax.plot(angles, label="RL")
            if inject:
                self.ax.axvline(x=disturb_step, color='red', linestyle='--', label='Disturbance')

            self.ax.legend()
            self.canvas.draw()

            metrics = calculate_metrics(angles)
            self.display_metrics("RL", metrics)
            export_metrics_to_csv("logs/performance_metrics.csv", "RL", metrics)

        threading.Thread(target=rl_thread).start()


    def display_metrics(self, controller_type, metrics):
        text = f"{controller_type} Metrics:\n"
        for key, val in metrics.items():
            text += f"  {key}: {val}\n"
        text += "\n"
        self.metrics_label.insert(tk.END, text)
        self.metrics_label.see(tk.END)



if __name__ == "__main__":
    root = tk.Tk()
    app = ControllerApp(root)
    root.mainloop()
