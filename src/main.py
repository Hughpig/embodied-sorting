import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import SortingEnv
from src.vision import Vision
from src.controller import Controller
from src.planner import Planner
from src.parser import CommandParser

def parse_args():
    parser = argparse.ArgumentParser(description="Embodied AI Sorting System")
    parser.add_argument("--mode", type=str, choices=["manual", "random", "nearest", "servo"], default=None)
    parser.add_argument("--cmd", type=str, default="")
    # 🎓 增加一键启动噪声模拟的参数开关！
    parser.add_argument("--noise", action="store_true", help="开启相机传感器高斯噪声干扰")
    parser.add_argument("--show-vis", action="store_true", help="显示 OpenCV AI 视觉监控与数据标注窗口")
    args, unknown = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_args()
    
    # 传入 noise 开关
    env = SortingEnv(gui=True, add_noise=args.noise)
    vision = Vision()
    controller = Controller(env)
    planner = Planner(env, vision, controller)
    nlp_parser = CommandParser()

    if args.show_vis:
        planner.show_vision = True

    blocks = env.reset(n_blocks=6)

    print("\n" + "="*50)
    print(" 🤖 多模态智能分拣系统 (Multi-Modal Sorting System)")
    if args.noise:
        print(" ⚠️ 警告：传感器受到高斯噪声严重干扰！")
    print(" 桌面上存在的方块:", [b.color for b in blocks])
    print("="*50)

    if args.mode is None:
        print("请选择分拣模式：")
        print(" [1] 自然语言指令模式 (manual)")
        print(" [2] 随机抓取模式 (random)")
        print(" [3] 贪心算法：最近优先模式 (nearest)")
        print(" [4] 🔥 纯视觉动态伺服跟踪模式 (servo)")
        choice = input(">> 请输入模式编号 (1/2/3/4): ").strip()
        
        if choice == '1': args.mode = 'manual'
        elif choice == '2': args.mode = 'random'
        elif choice == '4': args.mode = 'servo'
        else: args.mode = 'nearest'

    start_time = time.time()
    
    if args.mode == 'manual':
        cmd_text = args.cmd if args.cmd else input("\n>> 请用自然语言下达指令: ")
        if not cmd_text.strip(): cmd_text = "all"
        tasks = ["red", "green", "blue"] * 2 if cmd_text.lower() == "all" else nlp_parser.parse(cmd_text)
        planner.set_mode("language", tasks)
    elif args.mode == 'random':
        planner.set_mode("random")
    elif args.mode == 'servo':
        planner.set_mode("servo")
    else:
        planner.set_mode("nearest")

    step_count = 0
    max_steps = 1500 
    while planner.state != "FINISH" and step_count < max_steps:
        planner.step()
        step_count += 1

    time_cost = time.time() - start_time
    ok, total = env.count_success()
    
    print("\n" + "="*30)
    print(f"Mission Finished in {time_cost:.1f} seconds!")
    print(f"Success: {ok}/{total} ({ok/total*100:.1f}%)")
    print("="*30)

    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()