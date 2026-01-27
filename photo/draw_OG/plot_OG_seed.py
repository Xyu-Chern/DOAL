import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re

# --- 配置参数 ---
BASE_FOLDER = '.' 
OUTPUT_BASE_FOLDER = 'OG_seed' 

# ----------------------------------------------------
# 🌟 字体大小放大 2 倍 🌟
# ----------------------------------------------------
TITLE_FONTSIZE = 12 * 2  # 24
AXIS_FONTSIZE = 10 * 2   # 20
PADDING_RATIO = 0.05     # 对称padding

# CSV 读取配置
CSV_SEP = ','       
CSV_QUOTECHAR = '"' 

# ------------------------------------------------------------------
# 🌟 颜色和标记配置：根据模型名称进行分配 🌟
# ------------------------------------------------------------------

# 1. 颜色配置 (修改：带D的用蓝色，其余统一用灰色)
BLUE_COLOR = '#598BE7'  # 带 'D' 的模型用蓝色
GRAY_COLOR = '#808080'  # 不带 'D' 的模型统一用灰色

# 2. 标记配置：定义算法家族和对应标记
MARKER_MAPPING = {
    'FQL': '^',       # 三形 (用于 IFQL, DIFQL)
    'MFQL': 's',      # 方形 (用于 MFQL, DMFQL)
    'FLOW': 'o',      # 圆形 (用于 TRIGFLOW, DTRIGFLOW, TRIGFLOWQL)
    'REBRAC': 'v',    # 倒三角 (新增：用于 MFREBRAC, DMFREBRAC)
}

# 3. 备用标记和索引
USED_MARKERS = set(['^', 's', 'o', 'v']) # FQL, MFQL, FLOW, REBRAC 族已用
BACKUP_MARKERS = [m for m in ['D', 'p', 'h', '*'] if m not in USED_MARKERS]


# ----------------------------------------------------
# 1. 设置路径和查找环境文件夹
# ----------------------------------------------------
if not os.path.exists(BASE_FOLDER):
    print(f"错误：找不到基础文件夹 '{BASE_FOLDER}'。请确保该文件夹存在。")
    exit()

all_subdirs = [d for d in glob.glob(os.path.join(BASE_FOLDER, '*')) if os.path.isdir(d)]
env_folders = [d for d in all_subdirs if os.path.basename(d) != OUTPUT_BASE_FOLDER]

if not env_folders:
    print(f"错误：在 '{BASE_FOLDER}' 中未找到任何子文件夹作为环境。")
    exit()

if not os.path.exists(OUTPUT_BASE_FOLDER):
    os.makedirs(OUTPUT_BASE_FOLDER)
    print(f"已创建总输出文件夹: '{OUTPUT_BASE_FOLDER}'")

print(f"找到 {len(env_folders)} 个环境文件夹，准备开始批量绘图...")

# ----------------------------------------------------
# 2. 外层循环：遍历每个环境文件夹
# ----------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

for env_path in env_folders:
    env_name = os.path.basename(env_path)
    print(f"\n========================================================")
    print(f"--- 正在处理环境: {env_name} ---")
    print(f"========================================================")

    csv_files = glob.glob(os.path.join(env_path, '*.csv'))
    if not csv_files:
        print(f"警告：环境 '{env_name}' 中未找到任何 CSV 文件，跳过此环境。")
        continue

    # 重置绘图状态和变量
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.grid(False) 
    all_handles = []
    all_labels = []
    max_steps_m_global = 0.0
    
    MARKER_INDEX = 0 
    LOCAL_COLOR_COUNT = 0 

    # ----------------------------------------------------
    # 3. 内层逻辑：遍历文件并绘制数据
    # ----------------------------------------------------
    for file_path in csv_files:
        # --- 3.1 & 3.2 读取和处理数据 ---
        try:
            df = pd.read_csv(file_path, sep=CSV_SEP, encoding='utf-8', quotechar=CSV_QUOTECHAR, header=0, skip_blank_lines=True)
            df.columns = df.columns.str.strip() 
            STEP_COL = 'Step'
            success_cols = [col for col in df.columns if 'evaluation/success' in col and not col.endswith('__MIN') and not col.endswith('__MAX')]
            if STEP_COL not in df.columns or not success_cols: continue
            MEAN_COL = success_cols[0] 
            steps = pd.to_numeric(df[STEP_COL], errors='coerce').values
            rewards_avg = pd.to_numeric(df[MEAN_COL], errors='coerce').values
            valid_indices = ~np.isnan(steps) & ~np.isnan(rewards_avg)
            steps = steps[valid_indices]
            rewards_avg = rewards_avg[valid_indices]
        except Exception as e:
            print(f"警告：读取文件 {file_path} 时发生错误: {e}，跳过此文件。")
            continue
            
        if steps.size == 0: continue
        
        if steps[0] > 0:
            steps = np.insert(steps, 0, 0)
            rewards_avg = np.insert(rewards_avg, 0, 0)
            
        # --- 3.3 模型名称替换：将TRIGFLOWQL改为ETRIGFLOW ---
        file_name = os.path.basename(file_path)
        model_name = os.path.splitext(file_name)[0]
        # 将TRIGFLOWQL替换为ETRIGFLOW
        if 'TRIGFLOWQL' in model_name.upper():
            model_name = model_name.upper().replace('TRIGFLOWQL', 'ETRIGFLOW')
        else:
            model_name = model_name.upper()  # 保持原来的大写转换

        model_name_upper = model_name  # 现在model_name已经是处理过的大写形式
            
        steps_in_M = steps / 1000000
        max_steps_m_global = max(max_steps_m_global, steps_in_M.max())
        rewards_avg_pct = rewards_avg * 100
        
        
        # 🌟 颜色和标记分配逻辑 🌟
        # 确定颜色 (修改：带D的用蓝色，其余统一用灰色)
        if 'D' in model_name_upper:
            current_color = BLUE_COLOR
        else:
            current_color = GRAY_COLOR
            
        # 确定标记：根据 MARKER_MAPPING 家族规则
        current_marker = None
        
        # 优化标记选择逻辑，使用更具体的检查
        if 'MFQL' in model_name_upper:
            current_marker = MARKER_MAPPING['MFQL']
        elif 'REBRAC' in model_name_upper:  # 新增：REBRAC 家族检查
            current_marker = MARKER_MAPPING['REBRAC']
        elif 'TRIGFLOW' in model_name_upper or 'FLOW' in model_name_upper:
            current_marker = MARKER_MAPPING['FLOW']
        elif 'FQL' in model_name_upper:
            current_marker = MARKER_MAPPING['FQL']
        
        # 如果模型不属于任何已知族，使用备用标记
        if current_marker is None:
            current_marker = BACKUP_MARKERS[MARKER_INDEX % len(BACKUP_MARKERS)]
            MARKER_INDEX += 1
            
        # 🌟 透明度计算
        BASE_ALPHA = 0.85
        current_alpha = BASE_ALPHA - (LOCAL_COLOR_COUNT % 3) * 0.15 
        current_alpha = max(current_alpha, 0.5)
        LOCAL_COLOR_COUNT += 1 
        
        # --- 绘制折线和标记 ---
        line, = ax.plot(
            steps_in_M, 
            rewards_avg_pct, 
            color=current_color, 
            linestyle='-', 
            linewidth=2, 
            alpha=current_alpha,
            zorder=2
        )
        
        # 绘制标记
        ax.plot(
            steps_in_M[1:], 
            rewards_avg_pct[1:], 
            marker=current_marker, 
            markersize=7,
            markerfacecolor='white', 
            markeredgecolor=current_color,
            markeredgewidth=1.5, 
            linestyle='none', 
            alpha=current_alpha,
            zorder=3
        )
        
        # 确保图例显示标记
        line.set_marker(current_marker)
        line.set_markersize(7)
        line.set_markerfacecolor('white')
        line.set_markeredgecolor(current_color)
        line.set_markeredgewidth(1.5)
        
        all_handles.append(line)
        all_labels.append(model_name_upper)

        print(f"   成功绘制 {model_name_upper} (标记: {current_marker}, Alpha: {current_alpha:.2f}) 曲线。")
    
    
    # ----------------------------------------------------
    # 4. 统一坐标轴和图例配置
    # ----------------------------------------------------
    if not all_handles:
        print(f"警告：环境 '{env_name}' 中没有成功绘制任何一条曲线，跳过图片生成。")
        plt.close(fig) 
        continue

    # X轴刻度设置
    max_data_m = np.ceil(max_steps_m_global * 5) / 5
    x_ticks_m_base = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    if max_data_m > 1.0:
        extended_ticks = np.arange(1.2, max_data_m + 0.001, 0.2)
        x_ticks_m = np.concatenate((x_ticks_m_base, extended_ticks))
        x_ticks_m = np.unique(x_ticks_m[x_ticks_m <= max_data_m])
    else:
        x_ticks_m = np.array([t for t in x_ticks_m_base if t <= max_data_m])
    
    x_labels = [f'{t:.1f}' for t in x_ticks_m]

    ax.set_xticks(x_ticks_m)
    ax.set_xticklabels(x_labels)

    y_ticks = np.arange(0, 101, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(t)) for t in y_ticks])

    # ----------------------------------------------------
    # 🌟 坐标轴间距设置 (对称padding) 🌟
    # ----------------------------------------------------
    
    # X轴间距设置
    x_range = x_ticks_m.max()
    x_pad = x_range * PADDING_RATIO
    ax.set_xlim(left=-x_pad, right=x_range + x_pad)
    
    # Y轴间距设置
    y_range = 100 
    y_pad = y_range * PADDING_RATIO 
    ax.set_ylim(bottom=-y_pad, top=y_range + y_pad)

    # ----------------------------------------------------
    # 🌟 轴标签和标题设置 🌟
    # ----------------------------------------------------
    
    # 横纵label大小*2
    ax.set_xlabel("Steps ($\\times 10^6$)", fontsize=AXIS_FONTSIZE) 
    ax.set_ylabel("Performance", fontsize=AXIS_FONTSIZE)
    
    # 标题只要folder名字小写
    ax.set_title(f"{env_name.lower()}", fontsize=TITLE_FONTSIZE)

    # --- 网格线和虚线配置 ---
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)     
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, zorder=0)   
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)     
    ax.axvline(x_range, color='gray', linestyle='--', linewidth=0.8, zorder=0) 

    # --- 图例 ---
    ax.legend(
        all_handles,
        all_labels,
        loc='upper left',
        fontsize=AXIS_FONTSIZE * 0.5,  # 图例字体也相应放大
        frameon=False,
        title="Models",
        title_fontsize=AXIS_FONTSIZE * 0.6
    )

    # 调整刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=AXIS_FONTSIZE * 0.8)

    # 布局调整和保存
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_BASE_FOLDER, f'{env_name}_performance.pdf')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    print(f"成功生成环境 '{env_name}' 的矢量 PDF 文件：{output_path}")

print("\n========================================================")
print("所有环境文件处理完毕。")