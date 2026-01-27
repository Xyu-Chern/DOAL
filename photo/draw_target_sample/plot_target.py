import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re

# --- 配置参数 ---
BASE_FOLDER = '.' 
OUTPUT_BASE_FOLDER = 'target_nsample' 

# ----------------------------------------------------
# 🌟 字体大小放大 2 倍 (保持不变) 🌟
# ----------------------------------------------------
TITLE_FONTSIZE = 12 * 2  # 24
AXIS_FONTSIZE = 10 * 2   # 20
PADDING_RATIO = 0.05     # 保持不变，用于坐标轴间距的比例

# CSV 读取配置
CSV_SEP = ','
CSV_QUOTECHAR = '"' 

# ------------------------------------------------------------------
# 🌟 颜色和标记配置 🌟
# ------------------------------------------------------------------
# 蓝色颜色使用 598BE7 
COLOR_MAPPING = {
    'DMFQL.CSV': '#598BE7',    # 蓝色 (代表 target_sample=4)
    'DMFQL2.CSV': '#808080',   # 灰色 (代表 target_sample=1)
}
COMMON_MARKER = 's' 

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
plt.rcParams['mathtext.fontset'] = 'cm' # 启用 LaTeX 数学字体

for env_path in env_folders:
    env_name = os.path.basename(env_path)
    print(f"\n========================================================")
    print(f"--- 正在处理环境: {env_name} ---")
    print(f"========================================================")

    csv_files_all = glob.glob(os.path.join(env_path, '*.csv'))
    
    # ⚠️ 过滤文件：仅保留 DMFQL 和 DMFQL2 文件
    allowed_files_upper = ['DMFQL.CSV', 'DMFQL2.CSV'] 
    csv_files = [f for f in csv_files_all if os.path.basename(f).upper() in allowed_files_upper]

    if not csv_files:
        print(f"警告：环境 '{env_name}' 中未找到 DMFQL.csv 或 DMFQL2.csv 文件，跳过此环境。")
        continue

    # 重置绘图状态和变量
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    
    # ----------------------------------------------------
    # ❌ 移除：禁用虚线网格线，只保留边界虚线
    # ----------------------------------------------------
    # ax.grid(True, linestyle='--', alpha=0.7, zorder=1) 
    
    all_handles = []
    all_labels = []
    max_steps_m_global = 0.0
    LOCAL_COLOR_COUNT = 0 
    all_rewards_pct = [] # 收集所有环境的 rewards_avg_pct

    # ----------------------------------------------------
    # 3. 内层逻辑：遍历文件并绘制数据
    # ----------------------------------------------------
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        file_name_upper = file_name.upper() 
        
        # --- 3.1 & 3.2 读取和处理数据 (保持不变) ---
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
            print(f"警告：读取文件 {file_name} 时发生错误: {e}，跳过此文件。")
            continue
            
        if steps.size == 0: continue
        
        # 确保数据从 (0, 0) 开始，用于学习曲线
        if steps[0] > 0:
            steps = np.insert(steps, 0, 0)
            rewards_avg = np.insert(rewards_avg, 0, 0)
            
        steps_in_M = steps / 1000000
        max_steps_m_global = max(max_steps_m_global, steps_in_M.max())
        rewards_avg_pct = rewards_avg * 100 # 转换为 0-100 范围，但标签不加 %
        
        all_rewards_pct.extend(rewards_avg_pct) # 收集 Y 轴数据
        
        
        # 🌟 颜色、标记和标签分配逻辑 (不变) 🌟
        current_color = COLOR_MAPPING.get(file_name_upper, '#000000') 
        current_marker = COMMON_MARKER
        
        if file_name_upper == 'DMFQL.CSV':
            current_label = r'DMFQL ($n_{\text{target}}=4$)' 
        elif file_name_upper == 'DMFQL2.CSV':
            current_label = r'DMFQL ($n_{\text{target}}=1$)' 
        else:
            current_label = file_name 
        
        # 透明度计算
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
        all_labels.append(current_label) 

        print(f"   成功绘制 {file_name} (标签: {current_label}, 颜色: {current_color}) 曲线。")
    
    
    # ----------------------------------------------------
    # 4. 统一坐标轴和图例配置
    # ----------------------------------------------------
    if not all_handles:
        print(f"警告：环境 '{env_name}' 中没有成功绘制任何一条曲线，跳过图片生成。")
        plt.close(fig) 
        continue

    # --- X轴刻度设置 (不变) ---
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

    # --- Y轴刻度设置 (不变) ---
    y_ticks = np.arange(0, 101, 20)
    ax.set_yticks(y_ticks)
    # Y 轴刻度标签不带百分号
    ax.set_yticklabels([str(int(t)) for t in y_ticks])


    # ----------------------------------------------------
    # 🌟 坐标轴间距设置 (保持一致) 🌟
    # ----------------------------------------------------
    
    # X轴间距设置
    x_range = x_ticks_m.max() # X 轴右边界
    x_pad = x_range * PADDING_RATIO
    ax.set_xlim(left=-x_pad, right=x_range + x_pad)
    
    # Y轴间距设置
    y_range = 100 
    y_pad = y_range * PADDING_RATIO 
    ax.set_ylim(bottom=-y_pad, top=y_range + y_pad) 
    
    
    # ----------------------------------------------------
    # 🌟 轴标签和标题设置 🌟
    # ----------------------------------------------------
    
    ax.set_xlabel(r"Steps ($\times 10^6$)", fontsize=AXIS_FONTSIZE) 
    ax.set_ylabel(r"Performance", fontsize=AXIS_FONTSIZE) 
    
    # 标题修改：文件夹名全部小写
    ax.set_title(f"{env_name.lower()}", fontsize=TITLE_FONTSIZE)

    
    # ----------------------------------------------------
    # 🌟 最终修改：只绘制 4 条边界虚线 🌟
    # ----------------------------------------------------
    
    # 绘制 X 轴的 0 和 Max 垂线 (垂直虚线)
    # X=0 垂线
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0) 
    # X=Max (X 轴右边界) 垂线
    ax.axvline(x_range, color='gray', linestyle='--', linewidth=0.8, zorder=0) 
    
    # 绘制 Y 轴的 0 和 100 水平线 (水平虚线)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    
    
    # --- 图例 (字体大小已在 AXIS_FONTSIZE 中定义) ---
    ax.legend(
        all_handles,
        all_labels,
        loc='upper left',
        fontsize=AXIS_FONTSIZE * 0.5, 
        frameon=False,
        title="Models",
        title_fontsize=AXIS_FONTSIZE * 0.6 
    )
    
    # 调整刻度标签大小以匹配放大后的轴标签
    ax.tick_params(axis='both', which='major', labelsize=AXIS_FONTSIZE * 0.8) 

    # 布局调整和保存
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_BASE_FOLDER, f'{env_name}_DMFQL_Target_Sample_Comparison.pdf') 
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    print(f"成功生成环境 '{env_name}' 的矢量 PDF 文件：{output_path}")

print("\n========================================================")
print("所有环境文件处理完毕。")