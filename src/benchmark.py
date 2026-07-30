"""Automated Benchmark for Embodied Sorting System."""

import sys
import time
import csv
import argparse
from pathlib import Path

# 确保能找到 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import SortingEnv
from src.vision import Vision
from src.controller import Controller
from src.planner import Planner

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark for Sorting System")
    parser.add_argument("--mode", type=str, choices=["random", "nearest", "servo"], default="nearest",
                        help="测试模式: random, nearest, servo")
    parser.add_argument("--episodes", type=int, default=5,
                        help="跑多少轮测试")
    # 🎓 同样加入开关
    parser.add_argument("--noise", action="store_true", help="开启相机传感器高斯噪声干扰")
    args, _ = parser.parse_known_args()
    return args

def run_benchmark(mode="nearest", num_episodes=5, add_noise=False):
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 文件名区分是否加了 noise
    noise_str = "_noise" if add_noise else ""
    csv_file = data_dir / f"benchmark_{mode}{noise_str}_results.csv"
    
    print(f"Initializing Benchmark in [{mode.upper()}] mode (Noise: {add_noise})...")
    env = SortingEnv(gui=True, add_noise=add_noise) 
    vision = Vision()
    controller = Controller(env)
    
    all_rates = []
    all_times = []
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Mode", "Noise", "Total_Blocks", "Success_Count", "Success_Rate(%)", "Time_Cost(s)", "Actions_Taken"])
        
        for ep in range(1, num_episodes + 1):
            n_blocks = int(env.rng.integers(4, 7))
            blocks = env.reset(n_blocks=n_blocks)
            planner = Planner(env, vision, controller)
            
            planner.set_mode(mode)
            start_time = time.time()
            actions_taken = 0
            step_count = 0
            max_steps = 2500 
            
            print(f"\n[{time.strftime('%H:%M:%S')}] --- Starting Episode {ep}/{num_episodes} ({mode}) with {n_blocks} blocks ---")
            
            while planner.state != "FINISH" and actions_taken < 15 and step_count < max_steps:
                prev_state = planner.state
                planner.step()
                step_count += 1
                
                if prev_state in ["PICK_AND_PLACE", "VS_DELIVER"] and planner.state == "RETURN_HOME":
                    actions_taken += 1
            
            time_cost = time.time() - start_time
            ok, total = env.count_success()
            rate = (ok / total) * 100 if total > 0 else 0
            
            if step_count >= max_steps:
                print(f"[⚠️ WARNING] Episode {ep} 触发超时安全锁！强制结算。")
                
            print(f"[{time.strftime('%H:%M:%S')}] Episode {ep} Finished: {ok}/{total} ({rate:.1f}%) in {time_cost:.1f}s")
            
            writer.writerow([ep, mode, add_noise, total, ok, f"{rate:.1f}", f"{time_cost:.1f}", actions_taken])
            all_rates.append(rate)
            all_times.append(time_cost)
            
            time.sleep(1)
            
    env.close()
    
    avg_rate = sum(all_rates) / len(all_rates) if all_rates else 0
    avg_time = sum(all_times) / len(all_times) if all_times else 0
    print("\n" + "="*45)
    print(" 🏆 BENCHMARK COMPLETE 🏆 ")
    print(f" Mode: {mode.upper()} | Noise: {add_noise}")
    print(f" Total Episodes: {num_episodes}")
    print(f" Average Success Rate: {avg_rate:.1f}%")
    print(f" Average Time Cost: {avg_time:.1f}s")
    print(f" Results saved to: {csv_file}")
    print("="*45)

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(mode=args.mode, num_episodes=args.episodes, add_noise=args.noise)