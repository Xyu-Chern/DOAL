import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# 1. 定义模型和参数量数据
MODELS_DATA = {
    "cube-octuple (591M)": {"params": 591, "color": "#4A90E2", "marker": 'o', "label_name": "MLP [4096]*8 (591M)"}, 
    "puzzle-4x6 (80M)": {"params": 80, "color": "#4A90E2", "marker": 'o', "label_name": "MLP [1024]*16 (80M)"},      
    "puzzle-4x5 (149M)": {"params": 149, "color": "#555555", "marker": 's', "label_name": "MLP [2048]*8 (149M)"},   
    "hummaze-giant (17M)": {"params": 17, "color": "#555555", "marker": 's', "label_name": "MLP [1024]*4 (17M)"}     
}

# 2. X/Y 轴和绘图参数
N_POINTS = 11
MAX_STEPS = 5000000 
steps_data = np.linspace(0, MAX_STEPS, N_POINTS)
steps_in_M = steps_data / 1000000

TITLE_FONTSIZE = 12
AXIS_FONTSIZE = 10
SHADING_ALPHA = 0.15 

# 3. 蓝色线条的精确数据 (11个点)
BLUE_REWARDS = np.array([6, 10, 14, 18, 22, 40, 60, 70, 80, 95, 100])
BLUE_STDS = np.array([1, 2, 3, 4, 5, 8, 12, 5, 7, 30, 0])

# 4. 模型对比分组
comparison_groups = [
    ("cube-octuple (591M)", "puzzle-4x5 (149M)"), 
    ("puzzle-4x6 (80M)", "hummaze-giant (17M)"),   
    ("cube-octuple (591M)", "hummaze-giant (17M)"), 
    ("puzzle-4x5 (149M)", "puzzle-4x6 (80M)"),    
]

# 5. 数据生成函数 (仅用于生成随机的灰色线条)
def generate_grey_rewards_and_std(params, max_steps, steps_data):
    # 保持灰色线条的低性能和高方差模式
    base_reward = 65 
    final_reward = base_reward + np.log(params) * 5 + np.random.uniform(-5, 5)
    final_reward = min(85, final_reward) # 确保灰色线条性能低于蓝色
    k = 0.0000008 + np.log(params) / 10000000
    np.random.seed(int(params * 10) + 1) # 确保随机性
    small_noise = np.random.normal(0, 0.5, size=len(steps_data)) 
    rewards = final_reward * (1 / (1 + np.exp(-k * (steps_data - max_steps / 4)))) + small_noise

    # 方差随时间增大，且波动大
    std_dev_scaling_factor = 0.15 * 50 
    std_dev_base = (100 / params) * std_dev_scaling_factor + 1.0 
    time_factor = 0.5 + (steps_data/max_steps) 
    std_dev_array = np.full_like(rewards, std_dev_base) * time_factor
    
    # 强制最后一个点的方差为最小值
    std_dev_array[-1] = 1.0 
    
    return rewards, std_dev_array

# 6. 创建 1 行 5 列的子图布局 (4张图 + 1个右侧图例空间)
fig, axes = plt.subplots(
    1, 5, 
    figsize=(19, 3.5), 
    gridspec_kw={
        'width_ratios': [4, 4, 4, 4, 3], 
        'top': 0.85, 
        'bottom': 0.15, 
        'wspace': 0.3
    }
)

unique_models = {}

# 7. 循环绘制四张对比图 (axes[0] 到 axes[3])
for i, (model_1_key, model_2_key) in enumerate(comparison_groups):
    ax = axes[i]
    
    # --- 模型 1 (使用精确数据) ---
    data_1 = MODELS_DATA[model_1_key]
    if data_1["color"] == "#4A90E2": # 如果是蓝色线条，使用精确数据
        rewards_1, std_1 = BLUE_REWARDS, BLUE_STDS
    else: # 如果是灰色线条，使用随机数据
        rewards_1, std_1 = generate_grey_rewards_and_std(data_1["params"], MAX_STEPS, steps_data)
        
    # 绘制方差阴影
    ax.fill_between(steps_in_M, rewards_1 - std_1, rewards_1 + std_1, 
                    color=to_rgba(data_1["color"], SHADING_ALPHA), zorder=1)
    
    # 绘制折线和标记 (保持覆盖效果)
    line_1, = ax.plot(steps_in_M, rewards_1, color=data_1["color"], linestyle='-', linewidth=2, zorder=2)
    marker_1, = ax.plot(steps_in_M, rewards_1, marker=data_1["marker"], markersize=7,
                        markerfacecolor='white', markeredgecolor=data_1["color"], 
                        markeredgewidth=1.5, linestyle='none', zorder=3)
    
    # --- 模型 2 (使用精确数据) ---
    data_2 = MODELS_DATA[model_2_key]
    if data_2["color"] == "#4A90E2":
        rewards_2, std_2 = BLUE_REWARDS, BLUE_STDS
    else:
        rewards_2, std_2 = generate_grey_rewards_and_std(data_2["params"], MAX_STEPS, steps_data)
        
    # 绘制方差阴影
    ax.fill_between(steps_in_M, rewards_2 - std_2, rewards_2 + std_2, 
                    color=to_rgba(data_2["color"], SHADING_ALPHA), zorder=1)
    
    # 绘制折线和标记
    line_2, = ax.plot(steps_in_M, rewards_2, color=data_2["color"], linestyle='-', linewidth=2, zorder=2)
    marker_2, = ax.plot(steps_in_M, rewards_2, marker=data_2["marker"], markersize=7,
                        markerfacecolor='white', markeredgecolor=data_2["color"], 
                        markeredgewidth=1.5, linestyle='none', zorder=3)

    # --- 收集唯一的图例句柄和标签 ---
    if model_1_key not in unique_models:
        unique_models[model_1_key] = (line_1, data_1["label_name"])
    if model_2_key not in unique_models:
        unique_models[model_2_key] = (line_2, data_2["label_name"])

    # --- 坐标轴和网格配置 ---
    ax.grid(False) 
    ax.set_xticks([0, 5.0])
    ax.set_xticklabels(['0', '5M'])
    ax.set_yticks([0, 100]) 
    ax.set_yticklabels(['0', '100']) 
    
    # 关键修改：Y轴起始点上移。将Y轴下限设置在 -10 左右，使 0 刻度上移。
    ax.set_ylim(-10, 105) 
    
    ax.set_xlabel("Steps", fontsize=AXIS_FONTSIZE)
    if i == 0:
        ax.set_ylabel("Average Return / Performance", fontsize=AXIS_FONTSIZE)
    
    ax.set_title(f"Task Comparison {i+1}", fontsize=TITLE_FONTSIZE)

    # 关键修改：添加虚线网格
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax.axvline(5.0, color='gray', linestyle='--', linewidth=0.8, zorder=0)


# 8. 统一图例移动到右侧空白区域 (axes[4])
legend_ax = axes[4]
legend_ax.axis('off') 

legend_handles = [h[0] for h in unique_models.values()]
legend_labels = [h[1] for h in unique_models.values()]

legend_ax.legend(
    legend_handles, 
    legend_labels, 
    loc='center left', 
    bbox_to_anchor=(0.0, 0.5), 
    fontsize=9,
    frameon=False,
    title="Model Architectures",
    title_fontsize=10
)

# 9. 布局调整
plt.tight_layout() 

# 10. 保存图片
plt.savefig('rl_performance_4_lineplots_final_v8_exact_data.png', dpi=300)
plt.close()

print("已成功生成最终版、符合所有要求的四张并排折线图图片：rl_performance_4_lineplots_final_v8_exact_data.png (Y轴上移, 蓝色线条使用精确数据)")