#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hao Tang (Oct 2025)
# Purpose: Randomize MOOSE input parameters and generate new .i files

import random
import re
from pathlib import Path

# ====== 1️⃣ 用户配置部分 ======

# 模板文件路径
TEMPLATE_FILE = "template.i"

# 输出目录
OUTPUT_DIR = Path("generated_inputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 要随机化的参数及其范围 (最小值, 最大值)
PARAM_RANGES = {
    "factorL": (0.01, 0.5),
    "factor_f1": (1e-3, 1e-1),
    "factor_f2": (1e-3, 1e-1),
    "k": (1e-12, 1e-9),
    "S1": (1e6, 1e8),
    "S2": (0.5, 2.0)
}

# 要生成多少个随机输入文件
NUM_CASES = 5


# ====== 2️⃣ 随机替换函数 ======

def randomize_parameters(template_text: str, param_ranges: dict) -> tuple[str, dict]:
    """根据给定范围随机生成参数值，并替换模板文本"""
    new_values = {}
    text = template_text

    for param, (low, high) in param_ranges.items():
        val = random.uniform(low, high)
        new_values[param] = val

        # 使用正则替换 constant_expressions = 'xxx'
        pattern = rf"(constant_expressions\s*=\s*')([0-9.eE-]+)(')(?=.*{param})"
        # 若 pattern 不匹配（因为 MOOSE 格式有时把 param 放在不同行），可以使用全局替换策略
        if not re.search(pattern, text):
            # 替换 constant_expressions = '旧值' 出现在 param 的块里
            block_pattern = rf"({param}['\s]*[\n\r]+[\s\S]*?constant_expressions\s*=\s*')([0-9.eE-]+)(')"
            text = re.sub(block_pattern, rf"\g<1>{val:.6g}\g<3>", text)
        else:
            text = re.sub(pattern, rf"\g<1>{val:.6g}\g<3>", text)

    return text, new_values


# ====== 3️⃣ 主流程 ======

def main():
    template = Path(TEMPLATE_FILE).read_text()

    for i in range(1, NUM_CASES + 1):
        new_text, values = randomize_parameters(template, PARAM_RANGES)
        outfile = OUTPUT_DIR / f"case_{i:03d}.i"
        outfile.write_text(new_text)

        print(f"[+] Generated {outfile.name} with parameters:")
        for k, v in values.items():
            print(f"    {k:10s} = {v:.4e}")
        print("-" * 40)


if __name__ == "__main__":
    main()
