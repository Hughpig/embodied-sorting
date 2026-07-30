"""Natural Language Command Parser."""
import re

class CommandParser:
    def __init__(self):
        # 词汇表：支持中英文混用、同义词映射
        self.color_map = {
            "红": "red", "red": "red",
            "绿": "green", "green": "green",
            "蓝": "blue", "blue": "blue"
        }
        
    def parse(self, command: str) -> list:
        """
        输入: "先把红色的分拣了，再抓蓝方块，最后处理绿色"
        输出: ["red", "blue", "green"]
        """
        task_queue = []
        
        # 将用户的句子按标点符号或者顺承词稍微切分（其实直接正则全局搜也可以，为了严谨我们顺序提取）
        pattern = re.compile("|".join(self.color_map.keys()), re.IGNORECASE)
        matches = pattern.findall(command)
        
        for match in matches:
            color = self.color_map[match.lower()]
            # 去重：防止用户说“红色的红方块”导致重复生成两个红色任务
            if not task_queue or task_queue[-1] != color:
                task_queue.append(color)
                
        return task_queue