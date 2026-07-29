"""Automated Benchmark for Embodied Sorting System."""

import sys
import time
import csv
from pathlib import Path

# 确保能找到 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import SortingEnv
from src.vision import Vision
from src.controller import Controller
from src.planner import Planner

def run_benchmark(num_episodes=5):
    # 1. 自动创建 /data 文件夹
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / "benchmark_results.csv"
    
    # 2. 启动仿真环境
    print("Initializing Benchmark Environment...")
    env = SortingEnv(gui=True)  # 保持 GUI 打开，你可以端着咖啡欣赏它干活
    vision = Vision()
    controller = Controller(env)
    
    all_rates = []
    
    # 3. 准备写入 CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头（这直接就是你报告里的表格结构！）
        writer.writerow(["Episode", "Total_Blocks", "Success_Count", "Success_Rate(%)", "Time_Cost(s)", "Actions_Taken"])
        
        for ep in range(1, num_episodes + 1):
            # 随机生成 4 到 6 个方块
            n_blocks = int(env.rng.integers(4, 7))
            blocks = env.reset(n_blocks=n_blocks)
            planner = Planner(env, vision, controller)
            
            start_time = time.time()
            actions_taken = 0
            
            print(f"\n[{time.strftime('%H:%M:%S')}] --- Starting Episode {ep}/{num_episodes} with {n_blocks} blocks ---")
            
            # 运行状态机直到 FINISH 或触发安全锁 (最多允许抓取 15 次，防止死循环)
            while planner.state != "FINISH" and actions_taken < 15:
                prev_state = planner.state
                planner.step()
                
                # 统计抓取动作次数
                if prev_state == "PICK_AND_PLACE" and planner.state == "RETURN_HOME":
                    actions_taken += 1
            
            # 统计时间与成功率
            time_cost = time.time() - start_time
            ok, total = env.count_success()
            rate = (ok / total) * 100 if total > 0 else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Episode {ep} Finished: {ok}/{total} ({rate:.1f}%) in {time_cost:.1f}s")
            
            # 写入一行数据
            writer.writerow([ep, total, ok, f"{rate:.1f}", f"{time_cost:.1f}", actions_taken])
            all_rates.append(rate)
            
            # 跑完一轮休息一秒
            time.sleep(1)
            
    env.close()
    
    # 4. 输出最终总结
    avg_rate = sum(all_rates) / len(all_rates) if all_rates else 0
    print("\n" + "="*40)
    print(" 🏆 BENCHMARK COMPLETE 🏆 ")
    print(f" Total Episodes: {num_episodes}")
    print(f" Average Success Rate: {avg_rate:.1f}%")
    print(f" Benchmark results successfully saved to: {csv_file}")
    print("="*40)

if __name__ == "__main__":
    # 默认跑 5 轮，如果为了写报告生成漂亮的数据，可以改成 10
    run_benchmark(num_episodes=5)