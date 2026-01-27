import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# --- 配置参数 ---
# ----------------------------------------------------
# 🌟 字体大小放大 🌟
# ----------------------------------------------------
TITLE_FONTSIZE = 12 * 2  # 24
AXIS_FONTSIZE = 10 * 2   # 20

# ------------------------------------------------------------------
# 🌟 颜色和标记配置 🌟
# ------------------------------------------------------------------
LINE_COLOR = '#598BE7'  # 蓝色回归线

# 不同算法的灰色和标记配置
ALGORITHM_STYLES = {
    'dmfql': {'color': "#282828", 'marker': 'o'},      # 深灰色圆圈
    'mfql BPTT': {'color': "#282828", 'marker': 's'},  # 中灰色方形
    'mfql': {'color': "#282828", 'marker': '^'},       # 灰色三角形
    'fql': {'color': "#282828", 'marker': 'D'},        # 浅灰色菱形
    'ifql': {'color': "#282828", 'marker': 'v'},       # 更浅灰色倒三角
    'difql': {'color': "#282828", 'marker': 'p'}       # 最浅灰色五边形
}

# 数据
data = {
    'Algorithm': ['dmfql', 'mfql BPTT', 'mfql', 'fql', 'ifql', 'difql'],
    'Total_NN_Calls': [18, 37, 16, 20, 8, 10],
    'Actual_Time_Minutes': [37, 61, 35, 37, 29, 31]
}

# 创建DataFrame
df = pd.DataFrame(data)

# ----------------------------------------------------
# 设置绘图风格
# ----------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'  # 启用 LaTeX 数学字体

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.grid(False)

# ----------------------------------------------------
# 准备数据进行线性回归
# ----------------------------------------------------
X = df['Total_NN_Calls'].values.reshape(-1, 1)
y = df['Actual_Time_Minutes'].values

# 执行线性回归
regressor = LinearRegression()
regressor.fit(X, y)

# 计算回归方程
slope = regressor.coef_[0]
intercept = regressor.intercept_

# ----------------------------------------------------
# 绘制数据点 - 不同灰色镂空图案
# ----------------------------------------------------
for i, algorithm in enumerate(df['Algorithm']):
    style = ALGORITHM_STYLES[algorithm]
    ax.scatter(
        df['Total_NN_Calls'][i],
        df['Actual_Time_Minutes'][i],
        marker=style['marker'],
        s=80,  # 标记大小
        facecolor='white',  # 镂空效果
        edgecolor=style['color'],
        linewidth=1.5,
        alpha=0.8,
        zorder=3
    )

# ----------------------------------------------------
# 绘制蓝色回归线 - 延长到虚线边界
# ----------------------------------------------------
# 找到数据点的实际x坐标范围
# 计算从边界虚线 x=5 到 x=40 的回归线
# 计算从边界虚线 x=5 到 x=40 的回归线
x_line_extended = np.array([5, 40])  # 从边界虚线到边界虚线
y_line_extended = regressor.predict(x_line_extended.reshape(-1, 1))

# 如果 y 值小于 25，找到与 y=25 的交点
if y_line_extended[0] < 25:  # 如果 x=5 时的 y 值小于 25
    # 计算与 y=25 的交点
    # y = slope * x + intercept = 25
    # x = (25 - intercept) / slope
    x_intersect = (25 - intercept) / slope
    # 只显示从交点到 x=40 的部分
    x_line_final = np.array([x_intersect, 40])
    y_line_final = regressor.predict(x_line_final.reshape(-1, 1))
else:
    # 如果 y 值都大于等于 25，显示完整的线
    x_line_final = x_line_extended
    y_line_final = y_line_extended

regression_line, = ax.plot(
    x_line_final,
    y_line_final,
    color=LINE_COLOR,
    linestyle='-',
    linewidth=2.5,
    zorder=2
)
# ----------------------------------------------------
# 添加算法标签 - dmfql在左侧，其余在右侧，全部大写
# ----------------------------------------------------
for i, algorithm in enumerate(df['Algorithm']):
    if algorithm == 'dmfql':
        # dmfql在左侧
        xytext = (-60, 0)  # 左侧偏移
    elif algorithm == "mfql BPTT":
        xytext = (-90, 0)  # 左侧偏移
    else:
        # 其他算法在右侧
        xytext = (12, 0)   # 右侧偏移
    
    ax.annotate(algorithm.upper(),  # 全部大写
               (df['Total_NN_Calls'][i], df['Actual_Time_Minutes'][i]),
               xytext=xytext,
               textcoords='offset points',
               fontsize=AXIS_FONTSIZE * 0.7,  # 稍微调小避免重叠
               alpha=0.8,
               verticalalignment='center')  # 垂直居中

# ----------------------------------------------------
# 添加图例说明和回归方程到左上角
# ----------------------------------------------------
legend_text = ""
for algorithm, style in ALGORITHM_STYLES.items():
    # 获取对应的标记符号
    marker = style['marker']
    # 将标记符号转换为可显示的字符
    marker_symbols = {
        'o': '○',  # 圆圈
        's': '□',  # 方形
        '^': '△',  # 三角形
        'D': '◇',  # 菱形
        'v': '▽',  # 倒三角
        'p': '⬠'   # 五边形
    }
    marker_char = marker_symbols.get(marker, '•')
    legend_text += f"{marker_char} {algorithm.upper()}\n"

equation_text = f'y = {slope:.2f}x + {intercept:.2f}'

# 合并文本
combined_text = legend_text + equation_text 

ax.text(0.05, 0.95, combined_text,
        transform=ax.transAxes,
        fontsize=AXIS_FONTSIZE * 0.7,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        zorder=4)

# ----------------------------------------------------
# 坐标轴设置 - 减小padding
# ----------------------------------------------------
# 设置坐标轴范围（减小padding）
x_left = 4
x_right = 41
y_bottom = 24
y_top = 66

ax.set_xlim(left=x_left, right=x_right)
ax.set_ylim(bottom=y_bottom, top=y_top)

# 设置刻度（避免出现0,20,45,70）
ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
ax.set_yticks([25, 30, 35, 40, 45, 50, 55, 60, 65])

# 轴标签
ax.set_xlabel("Number of NN Calls", fontsize=AXIS_FONTSIZE)
ax.set_ylabel("Actual Time (min)", fontsize=AXIS_FONTSIZE)

# 标题
ax.set_title("Regression Analysis", fontsize=TITLE_FONTSIZE)

# ----------------------------------------------------
# 🌟 绘制4条边界虚线 x=5, x=40, y=25, y=65 🌟
# ----------------------------------------------------
# X轴虚线
ax.axvline(5, color='gray', linestyle='--', linewidth=0.8, zorder=0)
ax.axvline(40, color='gray', linestyle='--', linewidth=0.8, zorder=0)

# Y轴虚线
ax.axhline(25, color='gray', linestyle='--', linewidth=0.8, zorder=0)
ax.axhline(65, color='gray', linestyle='--', linewidth=0.8, zorder=0)

# 调整刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=AXIS_FONTSIZE * 0.8)

# 布局调整和保存
plt.tight_layout()

# 保存图片
output_path = 'NN_Calls_vs_Time_Regression.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"回归方程: y = {slope:.2f}x + {intercept:.2f}")
print(f"成功生成回归分析图: {output_path}")