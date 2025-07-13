import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import os
from matplotlib.font_manager import FontProperties, findSystemFonts

# --------------------------
# 路径获取函数
# --------------------------
def get_data_file_path(filename):
    """获取数据文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件路径（数据文件在上级目录的'数据'文件夹中）
    data_dir = os.path.join(os.path.dirname(current_dir), '数据')
    data_file_path = os.path.join(data_dir, filename)
    return data_file_path

def get_output_file_path(filename):
    """获取输出文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输出文件路径（在上级目录的分析图文件夹中）
    output_dir = os.path.join(os.path.dirname(current_dir), '分析图')
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


class DalianWeatherAnalysis:
    def __init__(self, csv_path, csv_2025_path=None):
        """初始化类，读取并预处理数据（新增2025年数据路径参数）"""
        self.df = pd.read_csv(csv_path)
        self.df_2025 = pd.read_csv(csv_2025_path) if csv_2025_path else None
        self.preprocess_data()
        # 预处理2025年数据
        if self.df_2025 is not None:
            self.preprocess_2025_data()

    def preprocess_data(self):
        """预处理2022-2024年数据"""
        # 转换日期格式并提取年月
        self.df['日期'] = pd.to_datetime(self.df['日期'], errors='coerce')
        self.df['年份'] = self.df['日期'].dt.year
        self.df['月份'] = self.df['日期'].dt.month

        # 处理温度列数据类型及缺失值
        self.df['最高温度'] = pd.to_numeric(self.df['最高温度'], errors='coerce')
        self.df['最低温度'] = pd.to_numeric(self.df['最低温度'], errors='coerce')
        self.df['最高温度'] = self.df.groupby(['年份', '月份'])['最高温度'].transform(
            lambda x: x.fillna(x.mean())
        )
        self.df['最低温度'] = self.df.groupby(['年份', '月份'])['最低温度'].transform(
            lambda x: x.fillna(x.mean())
        )

        # 提取风力等级和天气类别
        self.df['白天风力等级'] = self.df['白天风力'].apply(self.extract_wind_strength)
        self.df['夜间风力等级'] = self.df['夜间风力'].apply(self.extract_wind_strength)
        self.df['白天天气类别'] = self.df['白天天气'].apply(self.categorize_weather)
        self.df['夜间天气类别'] = self.df['夜间天气'].apply(self.categorize_weather)

    def preprocess_2025_data(self):
        """单独预处理2025年数据（提取月份和最高温度）"""
        if self.df_2025 is not None:
            self.df_2025['日期'] = pd.to_datetime(self.df_2025['日期'], errors='coerce')
            self.df_2025['月份'] = self.df_2025['日期'].dt.month
            self.df_2025['最高温度'] = pd.to_numeric(self.df_2025['最高温度'], errors='coerce')

    @staticmethod
    def extract_wind_strength(wind_str):
        """提取风力等级"""
        if pd.isna(wind_str):
            return '未知'
        match = re.search(r'(\d+-\d+级|\d+级|1-3级)', str(wind_str))
        return match.group(1) if match else '未知'

    @staticmethod
    def categorize_weather(weather):
        """归类天气状况"""
        if pd.isna(weather):
            return '其他'
        weather_mapping = {
            '晴': '晴', '多云': '多云', '阴': '阴',
            '小雨': '雨', '中雨': '雨', '大雨': '雨', '暴雨': '雨',
            '阵雨': '雨', '雷阵雨': '雨', '小到中雨': '雨', '中到大雨': '雨', '大到暴雨': '雨',
            '小雪': '雪', '中雪': '雪', '大雪': '雪', '阵雪': '雪', '小到中雪': '雪', '中到大雪': '雪',
            '雨夹雪': '雨夹雪'
        }
        return weather_mapping.get(str(weather), '其他')

    def task2_monthly_temp_trend_combined(self):
        """绘制2022-2024年月平均气温变化图"""
        monthly_temp = self.df.groupby(['年份', '月份'])[['最高温度', '最低温度']].mean().reset_index()
        avg_monthly_temp = monthly_temp.groupby('月份')[['最高温度', '最低温度']].mean().reset_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_monthly_temp, x='月份', y='最高温度',
                     marker='o', linewidth=2, markersize=8, label='平均最高温度')
        sns.lineplot(data=avg_monthly_temp, x='月份', y='最低温度',
                     marker='s', linewidth=2, markersize=8, label='平均最低温度')
        plt.title('大连市2022-2024年月平均气温变化趋势', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('温度 (℃)', fontsize=12)
        plt.xticks(range(1, 13), fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('月平均气温变化对比图.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        return avg_monthly_temp

    def task3_wind_distribution(self):
        """绘制白天和夜间风力分布柱状图（每月取三年平均值）"""
        # 白天风力：先按年份+月份+风力等级统计每年每月的天数，再求三年平均值
        wind_day = self.df.groupby(['年份', '月份', '白天风力等级']).size().reset_index()
        wind_day.columns = ['年份', '月份', '白天风力等级', '天数']
        wind_day_counts = wind_day.groupby(['月份', '白天风力等级'])['天数'].mean().unstack(fill_value=0)

        # 调整平均天数，使每月总和等于该月天数
        days_in_month = self.df.groupby(['年份', '月份'])['日期'].nunique().groupby('月份').mean()
        for month in range(1, 13):
            total_days = days_in_month[month]
            current_total = wind_day_counts.loc[month].sum()
            if current_total > 0:
                wind_day_counts.loc[month] = wind_day_counts.loc[month] * (total_days / current_total)

        # 白天风力图（标题更新为“平均天数”）
        plt.figure(figsize=(15, 8))
        wind_day_counts.plot(kind='bar', stacked=True, width=0.8)
        plt.title('大连市2022-2024年每月白天各风力等级平均天数分布', fontsize=15)  # 标题体现“平均”
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('平均天数', fontsize=12)  # 纵轴改为“平均天数”
        plt.xticks(range(12), range(1, 13), rotation=0)
        plt.legend(title='风力等级', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('白天风力情况平均分布图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 夜间风力：同样先按年份+月份+风力等级统计，再求平均值
        wind_night = self.df.groupby(['年份', '月份', '夜间风力等级']).size().reset_index()
        wind_night.columns = ['年份', '月份', '夜间风力等级', '天数']
        wind_night_counts = wind_night.groupby(['月份', '夜间风力等级'])['天数'].mean().unstack(fill_value=0)

        # 调整平均天数，使每月总和等于该月天数
        for month in range(1, 13):
            total_days = days_in_month[month]
            current_total = wind_night_counts.loc[month].sum()
            if current_total > 0:
                wind_night_counts.loc[month] = wind_night_counts.loc[month] * (total_days / current_total)

        # 夜间风力图（标题和纵轴更新）
        plt.figure(figsize=(15, 8))
        wind_night_counts.plot(kind='bar', stacked=True, width=0.8)
        plt.title('大连市2022-2024年每月夜间各风力等级平均天数分布', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('平均天数', fontsize=12)
        plt.xticks(range(12), range(1, 13), rotation=0)
        plt.legend(title='风力等级', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('夜间风力情况平均分布图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def task4_weather_distribution(self):
        """绘制白天和夜间天气状况分布柱状图（每月取三年平均值）"""
        # 白天天气：先按年份+月份+天气类别统计每年每月的天数，再求三年平均值
        weather_day = self.df.groupby(['年份', '月份', '白天天气类别']).size().reset_index()
        weather_day.columns = ['年份', '月份', '白天天气类别', '天数']
        weather_day_counts = weather_day.groupby(['月份', '白天天气类别'])['天数'].mean().unstack(fill_value=0)

        # 白天天气图（标题和纵轴更新）
        plt.figure(figsize=(15, 8))
        weather_day_counts.plot(kind='bar', stacked=True, width=0.8)
        plt.title('大连市2022-2024年每月白天各天气类别平均天数分布', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('平均天数', fontsize=12)
        plt.xticks(range(12), range(1, 13), rotation=0)
        plt.legend(title='天气类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('白天天气状况平均分布图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 夜间天气：同样先按年份+月份+天气类别统计，再求平均值
        weather_night = self.df.groupby(['年份', '月份', '夜间天气类别']).size().reset_index()
        weather_night.columns = ['年份', '月份', '夜间天气类别', '天数']
        weather_night_counts = weather_night.groupby(['月份', '夜间天气类别'])['天数'].mean().unstack(fill_value=0)

        # 夜间天气图（标题和纵轴更新）
        plt.figure(figsize=(15, 8))
        weather_night_counts.plot(kind='bar', stacked=True, width=0.8)
        plt.title('大连市2022-2024年每月夜间各天气类别平均天数分布', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('平均天数', fontsize=12)
        plt.xticks(range(12), range(1, 13), rotation=0)
        plt.legend(title='天气类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('夜间天气状况平均分布图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def task5_temp_prediction(self, avg_monthly_temp):
        """任务5：使用2025年真实数据进行预测对比"""
        # 1. 准备历史训练数据（2022-2024年每月平均最高温度）
        train_data = avg_monthly_temp[['月份', '最高温度']].rename(
            columns={'最高温度': '历史平均最高温度'}
        )

        # 2. 定义预测函数（基于历史平均值）
        def predict_temp(month):
            return train_data[train_data['月份'] == month]['历史平均最高温度'].values[0]

        # 3. 处理2025年真实数据（计算每月平均最高温度）
        if self.df_2025 is None:
            raise ValueError("请提供2025年数据的CSV路径")
        actual_2025 = self.df_2025.groupby('月份')['最高温度'].mean().reset_index()
        actual_2025 = actual_2025.rename(columns={'最高温度': '实际平均最高温度'})
        # 筛选1-6月数据
        actual_2025 = actual_2025[actual_2025['月份'].between(1, 6)]

        # 4. 生成预测数据
        predicted_2025 = pd.DataFrame({
            '月份': range(1, 7),
            '预测平均最高温度': [predict_temp(m) for m in range(1, 7)]
        })

        # 5. 合并对比数据
        comparison = pd.merge(actual_2025, predicted_2025, on='月份')

        # 6. 绘制对比折线图
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=comparison, x='月份', y='实际平均最高温度',
                     marker='o', linewidth=2, markersize=8, label='实际值')
        sns.lineplot(data=comparison, x='月份', y='预测平均最高温度',
                     marker='s', linewidth=2, markersize=8, label='预测值')
        plt.title('2025年1-6月平均最高温度预测与实际对比', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('平均最高温度 (℃)', fontsize=12)
        plt.xticks(range(1, 7), fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('温度预测对比图.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        return comparison


if __name__ == "__main__":
    # 初始化分析对象（使用路径获取函数）
    weather_analysis = DalianWeatherAnalysis(
        csv_path=get_data_file_path('dalian_weather_2022_2024.csv'),
        csv_2025_path=get_data_file_path('dalian_weather_2025_01_06.csv')  # 新增2025年数据路径
    )

    # 执行任务2：绘制气温趋势图
    print("正在执行任务2：绘制月平均气温变化对比图...")
    avg_temp = weather_analysis.task2_monthly_temp_trend_combined()

    # 执行任务3：绘制风力分布图
    print("正在执行任务3：绘制白天和夜间风力情况分布图...")
    weather_analysis.task3_wind_distribution()

    # 执行任务4：绘制天气状况分布图
    print("正在执行任务4：绘制白天和夜间天气状况分布图...")
    weather_analysis.task4_weather_distribution()

    # 执行任务5：基于真实数据的温度预测对比
    print("正在执行任务5：温度预测及对比...")
    prediction_result = weather_analysis.task5_temp_prediction(avg_temp)

    print("所有任务执行完成，结果已保存为图片文件。")
    print("2025年1-6月预测与实际对比数据：")
    print(prediction_result)



