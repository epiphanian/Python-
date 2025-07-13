import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties, findSystemFonts

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
    # 构建输出文件路径（在上级目录的分析图文件夹中的生成专家数据图像子文件夹）
    output_dir = os.path.join(os.path.dirname(current_dir), '分析图', '生成专家数据图像')
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
file = get_data_file_path('大乐透专家数据.csv')
df = pd.read_csv(file)

# 数据清洗：将缺失值和异常值处理为NaN，再去除
for col in ['彩龄', '发文量', '最近10次大乐透平均得分']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['彩龄', '发文量', '最近10次大乐透平均得分'])

# 1. 基本属性分布
plt.figure(figsize=(8,5))
sns.histplot(x=df['彩龄'], bins=20, kde=True)
plt.title('专家彩龄分布')
plt.xlabel('彩龄（年）')
plt.ylabel('专家人数')
plt.tight_layout()
output_file = get_output_file_path('专家彩龄分布.png')
plt.savefig(output_file)
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(x=df['发文量'], bins=20, kde=True)
plt.title('专家发文量分布')
plt.xlabel('发文量')
plt.ylabel('专家人数')
plt.tight_layout()
output_file = get_output_file_path('专家发文量分布.png')
plt.savefig(output_file)
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(x=df['最近10次大乐透平均得分'], bins=20, kde=True)
plt.title('最近10次大乐透平均得分分布')
plt.xlabel('平均得分')
plt.ylabel('专家人数')
plt.tight_layout()
output_file = get_output_file_path('平均得分分布.png')
plt.savefig(output_file)
plt.close()


# 3. 相互关系（散点图）
plt.figure(figsize=(8,5))
sns.scatterplot(x='彩龄', y='最近10次大乐透平均得分', data=df)
plt.title('彩龄与平均得分关系')
plt.xlabel('彩龄（年）')
plt.ylabel('平均得分')
plt.tight_layout()
output_file = get_output_file_path('彩龄与平均得分关系.png')
plt.savefig(output_file)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(x='发文量', y='最近10次大乐透平均得分', data=df)
plt.title('发文量与平均得分关系')
plt.xlabel('发文量')
plt.ylabel('平均得分')
plt.tight_layout()
output_file = get_output_file_path('发文量与平均得分关系.png')
plt.savefig(output_file)
plt.close()

# 4. 相关性热力图
plt.figure(figsize=(6,5))
corr = df[['彩龄','发文量','最近10次大乐透平均得分']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('专家属性与平均得分相关性')
plt.tight_layout()
output_file = get_output_file_path('专家属性相关性热力图.png')
plt.savefig(output_file)
plt.close()

# 彩龄与发文量的关系
plt.figure(figsize=(8,5))
sns.scatterplot(x='彩龄', y='发文量', data=df)
plt.title('彩龄与发文量的关系')
plt.xlabel('彩龄（年）')
plt.ylabel('发文量')
plt.tight_layout()
output_file = get_output_file_path('彩龄与发文量关系.png')
plt.savefig(output_file)
plt.close()

# 彩龄对平均得分影响（回归图）
plt.figure(figsize=(8,5))
sns.regplot(x='彩龄', y='最近10次大乐透平均得分', data=df, scatter_kws={'alpha':0.6})
plt.title('彩龄对平均得分的影响')
plt.xlabel('彩龄（年）')
plt.ylabel('平均得分')
plt.tight_layout()
output_file = get_output_file_path('彩龄对平均得分影响.png')
plt.savefig(output_file)
plt.close()

# 发文量对平均得分影响（回归图）
plt.figure(figsize=(8,5))
sns.regplot(x='发文量', y='最近10次大乐透平均得分', data=df, scatter_kws={'alpha':0.6})
plt.title('发文量对平均得分的影响')
plt.xlabel('发文量')
plt.ylabel('平均得分')
plt.tight_layout()
output_file = get_output_file_path('发文量对平均得分影响.png')
plt.savefig(output_file)
plt.close()

# 彩龄和发文量对平均得分的三维关系
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['彩龄'], df['发文量'], df['最近10次大乐透平均得分'], alpha=0.6)
ax.set_xlabel('彩龄（年）')
ax.set_ylabel('发文量')
ax.set_zlabel('平均得分')
ax.set_title('彩龄和发文量对平均得分的三维关系')
plt.tight_layout()
output_file = get_output_file_path('彩龄发文量对平均得分三维关系.png')
plt.savefig(output_file)
plt.close()

print('分析与可视化已完成，图片保存在 assignment_4/分析图/')
