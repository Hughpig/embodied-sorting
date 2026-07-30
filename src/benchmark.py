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
    parser.add_argument("--mode", type=str, choices=["random", "nearest"], default="nearest",
                        help="测试模式: random(随机) 或 nearest(最近优先)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="跑多少轮测试")
    args, _ = parser.parse_known_args()
    return args

def run_benchmark(mode="nearest", num_episodes=5):
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    # 根据模式生成不同的 csv 文件名，方便做数据对比！
    csv_file = data_dir / f"benchmark_{mode}_results.csv"
    
    print(f"Initializing Benchmark Environment in [{mode.upper()}] mode...")
    env = SortingEnv(gui=True) 
    vision = Vision()
    controller = Controller(env)
    
    all_rates = []
    all_times = []
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Mode", "Total_Blocks", "Success_Count", "Success_Rate(%)", "Time_Cost(s)", "Actions_Taken"])
        
        for ep in range(1, num_episodes + 1):
            n_blocks = int(env.rng.integers(4, 7))
            blocks = env.reset(n_blocks=n_blocks)
            planner = Planner(env, vision, controller)
            
            # ==========================================
            # 🎓 核心修复：给 Planner 注入运行模式！
            # ==========================================
            planner.set_mode(mode)
            
            start_time = time.time()
            actions_taken = 0
            
            print(f"\n[{time.strftime('%H:%M:%S')}] --- Starting Episode {ep}/{num_episodes} ({mode}) with {n_blocks} blocks ---")
            
            while planner.state != "FINISH" and actions_taken < 15:
                prev_state = planner.state
                planner.step()
                
                if prev_state == "PICK_AND_PLACE" and planner.state == "RETURN_HOME":
                    actions_taken += 1
            
            time_cost = time.time() - start_time
            ok, total = env.count_success()
            rate = (ok / total) * 100 if total > 0 else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Episode {ep} Finished: {ok}/{total} ({rate:.1f}%) in {time_cost:.1f}s")
            
            writer.writerow([ep, mode, total, ok, f"{rate:.1f}", f"{time_cost:.1f}", actions_taken])
            all_rates.append(rate)
            all_times.append(time_cost)
            
            time.sleep(1)
            
    env.close()
    
    avg_rate = sum(all_rates) / len(all_rates) if all_rates else 0
    avg_time = sum(all_times) / len(all_times) if all_times else 0
    print("\n" + "="*45)
    print(" 🏆 BENCHMARK COMPLETE 🏆 ")
    print(f" Mode: {mode.upper()}")
    print(f" Total Episodes: {num_episodes}")
    print(f" Average Success Rate: {avg_rate:.1f}%")
    print(f" Average Time Cost: {avg_time:.1f}s")
    print(f" Results saved to: {csv_file}")
    print("="*45)

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(mode=args.mode, num_episodes=args.episodes)