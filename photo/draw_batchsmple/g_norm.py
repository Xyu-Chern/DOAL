import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re

# --- 配置参数 ---
BASE_FOLDER = '.' 
OUTPUT_FOLDER = 'g_norm' 
OUTPUT_FILENAME = 'all_envs_grad_norm_blue_gray_final_refinement.pdf' # 更改文件名

# CSV 读取配置
CSV_FILE_NAME = 'batch.csv' # <--- 请在这里设置您的CSV文件名
CSV_SEP = ','
STEP_COL = 'Step'
TARGET_METRIC = 'g_norm_mean' # 目标度量

# 绘图配置
TITLE_FONTSIZE = 14
AXIS_FONTSIZE = 12
LEGEND_FONTSIZE = 12  # 🌟 增大图例字体大小 🌟
LINE_WIDTH = 1.5
# 确保 0 值向上平移的关键参数：Y轴底部间距比例
Y_PADDING_RATIO_BOTTOM = 0.1 # 增大底部间距比例，确保 0 轴明显上移
Y_PADDING_RATIO_TOP = 0.1 # 顶部间距比例
X_PADDING_RATIO = 0.02 # X 轴间距比例

# ----------------------------------------------------
# 🌟 颜色方案配置 (不变) 🌟
# ----------------------------------------------------
BLUE_GRAY_PALETTE = [
    '#0047AB',  # 1. 深蓝 (Dark Blue)
    '#4682B4',  # 2. 浅蓝/钢青 (Light Blue)
    '#808080',  # 3. 中灰 (Medium Gray)
    '#1E90FF',  # 4. 道奇蓝 (Dodge Blue - 强调)
    '#A9A9A9',  # 5. 深灰色 (Dark Gray)
    '#ADD8E6',  # 6. 浅蓝 (Light Cyan)
    '#696969',  # 7. 暗灰 (Dim Gray)
    '#4169E1',  # 8. 皇室蓝 (Royal Blue)
    '#C0C0C0',  # 9. 银灰 (Silver Gray)
    '#B0C4DE',  # 10. 浅钢青 (Light Steel Blue)
]
# ----------------------------------------------------

# ----------------------------------------------------
# 1-3. 数据读取和处理 (不变)
# ----------------------------------------------------
csv_path = os.path.join(BASE_FOLDER, CSV_FILE_NAME)

if not os.path.exists(csv_path):
    print(f"错误：找不到指定的 CSV 文件 '{CSV_FILE_NAME}'。请检查文件名和路径。")
    exit()

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

try:
    df = pd.read_csv(csv_path, sep=CSV_SEP, encoding='utf-8', quotechar='"', header=0, skip_blank_lines=True)
    df.columns = df.columns.str.strip() 

    if STEP_COL not in df.columns:
        print(f"错误：CSV 文件中未找到必需的列 '{STEP_COL}'。")
        exit()

except Exception as e:
    print(f"警告：读取文件 {CSV_FILE_NAME} 时发生错误: {e}")
    exit()

data_to_plot = {}
all_mean_values = []
max_steps_global = 0

for col in df.columns:
    if TARGET_METRIC in col:
        parts = col.split(' - ')
        if len(parts) == 2:
            env_name = parts[0].strip()
            metric_path = parts[1].strip()
            
            if metric_path.endswith(TARGET_METRIC):
                steps = pd.to_numeric(df[STEP_COL], errors='coerce').values
                means = pd.to_numeric(df[col], errors='coerce').values
                
                valid_indices = ~np.isnan(steps) & ~np.isnan(means)
                steps = steps[valid_indices]
                means = means[valid_indices]
                
                if steps.size > 0:
                    data_to_plot[env_name] = {'steps': steps, 'means': means}
                    max_steps_global = max(max_steps_global, steps.max())
                    all_mean_values.extend(means)


if not data_to_plot:
    print(f"没有找到包含 '{TARGET_METRIC}' 的有效数据列。")
    exit()

# ----------------------------------------------------
# 4. 绘制所有曲线到同一张图上
# ----------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# 🌟 关闭默认网格线 🌟
# ax.grid(True, linestyle='--', alpha=0.6) 

color_index = 0
sorted_env_names = sorted(data_to_plot.keys())
all_handles = []
all_labels = []

for env_name in sorted_env_names:
    data = data_to_plot[env_name]
    steps = data['steps']
    means = data['means']
    
    current_color = BLUE_GRAY_PALETTE[color_index % len(BLUE_GRAY_PALETTE)]
    color_index += 1
    
    line, = ax.plot(
        steps, 
        means, 
        label=env_name, 
        linewidth=LINE_WIDTH,
        color=current_color,
        zorder=3
    )
    all_handles.append(line)
    all_labels.append(env_name)

# ----------------------------------------------------
# 5. 配置图表外观 (关键修改部分)
# ----------------------------------------------------

# --- Y轴限制和间距：实现 0 刻度向上平移 ---
y_data_min = np.min(all_mean_values) if all_mean_values else 0
y_data_max = np.max(all_mean_values) if all_mean_values else 1

y_range = y_data_max - y_data_min if y_data_max > y_data_min else y_data_max
if y_range == 0:
    y_range = y_data_max * 0.1 or 0.1 
    
# 确保 y_data_min 不小于 0
y_data_min = max(0, y_data_min)

# 🌟 纵坐标下限设置：通过增大底部间距比例，强制 0 轴上移 🌟
# Y 轴底部必须低于 0 才能让 0 刻度上移
y_bottom = 0 - y_range * Y_PADDING_RATIO_BOTTOM 
y_top = y_data_max + y_range * Y_PADDING_RATIO_TOP

# 保证 y_top 至少大于 0
y_top = max(y_top, y_data_max * 1.05) 

ax.set_ylim(bottom=y_bottom, top=y_top)

# --- X轴限制和间距 ---
x_range = max_steps_global
x_pad = x_range * X_PADDING_RATIO
x_right = max_steps_global + x_pad
x_left = 0 - x_pad

ax.set_xlim(left=x_left, right=x_right)


# 🌟 添加四条虚线 🌟
# 1. Y=0 虚线
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, zorder=2, alpha=0.7, label='Y=0 Reference')
# 2. X=0 虚线
ax.axvline(0, color='black', linestyle='--', linewidth=0.8, zorder=2, alpha=0.7, label='X=0 Reference')
# 3. Y轴最大值参考虚线
ax.axhline(y_data_max, color='gray', linestyle=':', linewidth=0.8, zorder=2, alpha=0.6, label='Max Value')
# 4. X轴最大值参考虚线
ax.axvline(max_steps_global, color='gray', linestyle=':', linewidth=0.8, zorder=2, alpha=0.6, label='Max Steps')


# 轴标签和标题
ax.set_xlabel(r"Steps", fontsize=AXIS_FONTSIZE) 
ax.set_ylabel(r"Gradient Norm Mean", fontsize=AXIS_FONTSIZE) 
ax.set_title(r"Comparison of Gradient Norm Mean Across Environments", fontsize=TITLE_FONTSIZE)

# 🌟 图例配置：增大字体，移除环境名称前缀 🌟
# 创建新的标签列表，移除 "env_name:" 前缀
cleaned_labels = []
for label in all_labels:
    # 移除可能存在的环境名称前缀（如 "env_name:"）
    if ':' in label:
        cleaned_label = label.split(':', 1)[1].strip()  # 只分割一次，取后半部分
    else:
        cleaned_label = label
    cleaned_labels.append(cleaned_label)

ax.legend(
    all_handles,
    cleaned_labels,  # 🌟 使用清理后的标签 🌟
    loc='upper right',
    fontsize=LEGEND_FONTSIZE,  # 🌟 使用增大的字体大小 🌟
    frameon=True,
    title="Environments"
)

# 布局调整和保存
plt.tight_layout()
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"\n========================================================")
print(f"成功生成 Grad Norm 比较图：{output_path}")
print("========================================================")