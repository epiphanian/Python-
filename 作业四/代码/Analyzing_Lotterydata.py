import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties, findSystemFonts
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------
# 路径获取函数
# --------------------------
def get_data_file_path(filename):
    """获取数据文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据文件在上级目录的数据文件夹中
    data_dir = os.path.join(os.path.dirname(current_dir), '数据')
    data_file_path = os.path.join(data_dir, filename)
    return data_file_path

def get_output_file_path(filename):
    """获取输出文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输出文件路径（在上级目录的分析图文件夹中的生成开奖数据图像子文件夹）
    output_dir = os.path.join(os.path.dirname(current_dir), '分析图', '生成开奖数据图像')
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, filename)
    return output_file_path

# --------------------------
# 优化：设置兼容的中文字体
# --------------------------
chinese_fonts = [
    "SimHei",          # 黑体（Windows 常见）
    "Microsoft YaHei", # 微软雅黑（Windows 常见）
    "Arial Unicode MS",# 跨平台兼容（macOS/Linux 常见）
    "Heiti TC",        # 黑体（macOS 常见）
    "WenQuanYi Micro Hei"  # 文泉驿微米黑（Linux 常见）
]
available_fonts = findSystemFonts()
selected_font = None
for font in chinese_fonts:
    if any(font.lower() in f.lower() for f in available_fonts):
        selected_font = font
        break
if selected_font:
    plt.rcParams["font.family"] = [selected_font]
else:
    print("警告：未找到可用的中文字体，可能导致中文显示异常！")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv(get_data_file_path('大乐透开奖数据.csv'))

# 1. 总销售额随开奖日期的变化趋势及预测
df['开奖时间'] = pd.to_datetime(df['开奖时间'])
df = df.sort_values('开奖时间')
plt.figure(figsize=(12,5))
plt.plot(df['开奖时间'], df['销售金额'], marker='o')
plt.title('大乐透总销售额随开奖日期的变化趋势')
plt.xlabel('开奖日期')
plt.ylabel('销售金额')
plt.grid()
plt.tight_layout()
output_file = get_output_file_path('销售额趋势.png')
plt.savefig(output_file)
plt.close()

# 线性回归预测下一期销售额
X = np.arange(len(df)).reshape(-1,1)
y = df['销售金额'].values
model = LinearRegression()
model.fit(X, y)
next_index = np.array([[len(df)]])
predicted_sales = model.predict(next_index)[0]
print(f"预测2025年7月1日之后最近一期的销售额为：{predicted_sales:.0f}")

# 2. 前区与后区号码频率统计与可视化
front_nums = []
back_nums = []
for code in df['开奖号码']:
    nums = code.split()
    front_nums.extend(nums[:5])
    back_nums.extend(nums[5:])

front_counter = Counter(front_nums)
back_counter = Counter(back_nums)

plt.figure(figsize=(14,5))
plt.bar(list(front_counter.keys()), list(front_counter.values()))
plt.title('前区号码出现频率')
plt.xlabel('号码')
plt.ylabel('出现次数')
output_file = get_output_file_path('前区号码频率.png')
plt.savefig(output_file)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(list(back_counter.keys()), list(back_counter.values()), color='orange')
plt.title('后区号码出现频率')
plt.xlabel('号码')
plt.ylabel('出现次数')
output_file = get_output_file_path('后区号码频率.png')
plt.savefig(output_file)
plt.close()

# 3. 推荐一组大乐透号码
# 推荐出现频率最高的前5+后2
front_recommend = [num for num, _ in front_counter.most_common(5)]
back_recommend = [num for num, _ in back_counter.most_common(2)]
print("推荐投注号码：前区", front_recommend, "后区", back_recommend)

# 1. 分组柱状图：显示不同开奖日的平均销售额
plt.figure(figsize=(8,5))
mean_sales = df.groupby('周几')['销售金额'].mean().reindex(['周一','周三','周六'])
mean_sales.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52'])
plt.title('不同开奖日的平均销售额')
plt.xlabel('周几')
plt.ylabel('平均销售额')
plt.xticks(rotation=0)
plt.tight_layout()
output_file = get_output_file_path('平均销售额.png')
plt.savefig(output_file)
plt.close()

# 2. 小提琴图：显示不同开奖日的销售额分布
plt.figure(figsize=(8,5))
sns.violinplot(x='周几', y='销售金额', data=df, order=['周一','周三','周六'])
plt.title('不同开奖日的销售额分布（小提琴图）')
plt.xlabel('周几')
plt.ylabel('销售金额')
plt.tight_layout()
output_file = get_output_file_path('小提琴图.png')
plt.savefig(output_file)
plt.close()

# 3. 分组散点图（可选）
plt.figure(figsize=(8,5))
sns.stripplot(x='周几', y='销售金额', data=df, order=['周一','周三','周六'], jitter=True)
plt.title('不同开奖日的销售额分布（散点图）')
plt.xlabel('周几')
plt.ylabel('销售金额')
plt.tight_layout()
output_file = get_output_file_path('散点图.png')
plt.savefig(output_file)
plt.close()

