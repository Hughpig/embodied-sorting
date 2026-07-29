import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import SortingEnv
from src.vision import Vision
from src.controller import Controller
from src.planner import Planner

def main() -> None:
    env = SortingEnv(gui=True)
    vision = Vision()
    controller = Controller(env)
    planner = Planner(env, vision, controller)

    blocks = env.reset(n_blocks=4)

    print("\n" + "="*30)
    print("Starting sorting demo...")
    print("Reset blocks:", [(b.color, b.body_id) for b in blocks])
    print("="*30 + "\n")

    step_count = 0
    max_steps = 100  # 防止无限死循环的安全锁

    # 让主循环根据状态机运行，直到所有方块分拣完毕
    while planner.state != "FINISH" and step_count < max_steps:
        state = planner.step()
        
        # 为了输出日志更清晰，只在特定状态打印检测数量
        if state == "SELECT":
            print(f"State: {state} | Valid Objects to process: {len(planner.detections)}")
        elif state not in ["SCAN"]:
            print(f"State: {state}")
            
        step_count += 1

    ok, total = env.count_success()
    print("\n" + "="*30)
    print(f"Mission Finished!")
    print(f"Success: {ok}/{total} ({ok/total*100:.1f}%)")
    print("="*30)

    # 结束前停留几秒，方便录制视频
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()