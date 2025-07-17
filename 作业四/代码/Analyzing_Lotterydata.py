import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties, findSystemFonts
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet

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

# Prophet模型预测下一期销售额
print("开始使用Prophet模型进行销售额预测...")

try:
    # 准备Prophet所需的数据格式
    prophet_df = df[['开奖时间', '销售金额']].copy()
    prophet_df.columns = ['ds', 'y']  # Prophet要求列名为'ds'和'y'

    # 创建Prophet模型
    model = Prophet(
        yearly_seasonality=True,    # 年度季节性
        weekly_seasonality=True,    # 周度季节性
        daily_seasonality=False,    # 日度季节性（数据按天，不需要）
        seasonality_mode='multiplicative',  # 乘法季节性
        changepoint_prior_scale=0.05,  # 趋势变化点的灵活性
        seasonality_prior_scale=10.0   # 季节性的强度
    )

    # 拟合模型
    print("正在拟合Prophet模型...")
    model.fit(prophet_df)

    # 创建未来时间点进行预测
    last_date = prophet_df['ds'].max()
    if isinstance(last_date, pd.Timestamp):
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                    periods=1, freq='D')  # 只预测下一期
    else:
        # 如果last_date不是Timestamp，转换为datetime
        last_date = pd.to_datetime(last_date)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                    periods=1, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # 进行预测
    forecast = model.predict(future_df)

    # 获取下一期的预测值
    next_prediction = forecast.iloc[0]
    predicted_sales = next_prediction['yhat']
    prediction_lower = next_prediction['yhat_lower']
    prediction_upper = next_prediction['yhat_upper']

    print(f"Prophet模型预测结果:")
    print(f"下一期销售额预测值: {predicted_sales:.0f}")
    print(f"预测区间: [{prediction_lower:.0f}, {prediction_upper:.0f}]")

except Exception as e:
    print(f"Prophet模型拟合失败: {e}")
    print("回退到线性回归模型")
    # 回退到线性回归
    X = np.arange(len(df)).reshape(-1,1)
    y = df['销售金额'].values
    model = LinearRegression()
    model.fit(X, y)
    next_index = np.array([[len(df)]])
    predicted_sales = model.predict(next_index)[0]
    print(f"线性回归预测下一期销售额为：{predicted_sales:.0f}")

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

# --------------------------
# 新功能：统计并可视化周一、周三、周六开奖号码出现频率
# --------------------------
weekday_map = {'周一': 'monday', '周三': 'wednesday', '周六': 'saturday'}
for weekday, en_name in weekday_map.items():
    weekday_df = df[df['周几'] == weekday]
    weekday_front_nums = []
    weekday_back_nums = []
    for code in weekday_df['开奖号码']:
        nums = code.split()
        weekday_front_nums.extend(nums[:5])
        weekday_back_nums.extend(nums[5:])
    front_counter = Counter(weekday_front_nums)
    back_counter = Counter(weekday_back_nums)
    # 前区可视化
    plt.figure(figsize=(12,5))
    plt.bar(list(front_counter.keys()), list(front_counter.values()), color='#4C72B0')
    plt.title(f'{weekday} 前区号码出现频率')
    plt.xlabel('号码')
    plt.ylabel('出现次数')
    plt.tight_layout()
    output_file = get_output_file_path(f'{en_name}_front_freq.png')
    plt.savefig(output_file)
    plt.close()
    # 后区可视化
    plt.figure(figsize=(8,5))
    plt.bar(list(back_counter.keys()), list(back_counter.values()), color='#C44E52')
    plt.title(f'{weekday} 后区号码出现频率')
    plt.xlabel('号码')
    plt.ylabel('出现次数')
    plt.tight_layout()
    output_file = get_output_file_path(f'{en_name}_back_freq.png')
    plt.savefig(output_file)
    plt.close()



