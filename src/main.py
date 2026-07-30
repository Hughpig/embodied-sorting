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
    parser.add_argument("--mode", type=str, choices=["manual", "random", "nearest"], default=None,
                        help="运行模式: manual(人工语言调度), random(随机策略), nearest(最近优先策略)")
    parser.add_argument("--cmd", type=str, default="",
                        help="语言指令 (仅在 --mode manual 时生效)")
    # 使用 parse_known_args 避免与 PyBullet 底层的 argv 冲突
    args, unknown = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_args()
    
    env = SortingEnv(gui=True)
    vision = Vision()
    controller = Controller(env)
    planner = Planner(env, vision, controller)
    nlp_parser = CommandParser()

    blocks = env.reset(n_blocks=6)

    print("\n" + "="*50)
    print(" 🤖 多模态智能分拣系统 (Multi-Modal Sorting System)")
    print(" 桌面上存在的方块:", [b.color for b in blocks])
    print("="*50)

    # 如果没有传参数，保留原本的交互式菜单作为 Fallback
    if args.mode is None:
        print("请选择分拣模式：")
        print(" [1] 自然语言指令模式 (manual)")
        print(" [2] 随机抓取模式 (random)")
        print(" [3] 贪心算法：最近优先模式 (nearest)")
        choice = input(">> 请输入模式编号 (1/2/3): ").strip()
        
        if choice == '1':
            args.mode = 'manual'
        elif choice == '2':
            args.mode = 'random'
        else:
            args.mode = 'nearest'

    # 根据参数/选择执行不同的模式
    start_time = time.time()
    
    if args.mode == 'manual':
        # 如果命令行里带了 --cmd，直接用；否则要用户输入
        cmd_text = args.cmd if args.cmd else input("\n>> 请用自然语言下达指令: ")
        if not cmd_text.strip():
            cmd_text = "all"  # 默认兜底
            
        if cmd_text.lower() == "all":
            tasks = ["red", "green", "blue"] * 2
        else:
            tasks = nlp_parser.parse(cmd_text)
            
        planner.set_mode("language", tasks)
        
    elif args.mode == 'random':
        planner.set_mode("random")
        
    elif args.mode == 'nearest':
        planner.set_mode("nearest")

    # 开始执行主循环
    step_count = 0
    max_steps = 300 
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