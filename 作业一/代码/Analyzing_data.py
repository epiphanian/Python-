import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties, findSystemFonts
import re
import os

# --------------------------
# 路径获取函数
# --------------------------
def get_data_file_path(filename):
    """获取数据文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件路径（在上级目录的数据文件夹中）
    data_dir = os.path.join(os.path.dirname(current_dir), '数据')
    data_file_path = os.path.join(data_dir, filename)
    return data_file_path

def get_output_file_path(filename):
    """获取输出文件的完整路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输出文件路径（在上级目录的分析图文件夹中）
    output_dir = os.path.join(os.path.dirname(current_dir), '分析图')
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

# 1. 数据加载与预处理（修改后）
def load_data(file_path):
    """加载CSV数据并进行预处理，将省份中的nan替换为'未知'"""
    df = pd.read_csv(file_path)
    
    # 处理财富值（提取数值）
    df['财富'] = df['财富'].apply(lambda x: float(str(x).replace('亿', '')) if '亿' in str(x) else float(x))
    
    # 处理行业字段（拆分多个行业）
    df['行业列表'] = df['行业'].apply(lambda x: re.split(r'、|,|，', str(x)) if pd.notna(x) else [])
    
    # 处理出生地省份（关键修改：将nan替换为'未知'）
    df['省份'] = df['出生地'].apply(
        lambda x: 
            str(x).split('-')[1] if '-' in str(x) else  # 正常提取省份
            '未知' if pd.isna(x) or str(x).lower() == 'nan' else  # 处理nan
            str(x)  # 其他情况保留原值
    )
    
    return df

# 2. 行业分类映射
industry_mapping = {
    '农业与农副食品': ['智慧农业', '畜牧养殖', '种业', '农药化肥', '饲料', '生猪养殖', '农业', '农副食品加工业'],
    '工业与制造业': ['电子设备', '专用设备制造业', '船舶制造', '便携储能电池', '航空航天', '五金配件', 
                   '机械制造', '高压油缸', '自动化设备', '电线电缆', '工业母机', '电器', '机电设备',
                   '工业自动化', '电力设备', '工具制造', '光纤光缆', '充电桩', '电视面板', '机械',
                   '机电', '制造业', '光伏设备', '航天航空', '专业无线通信设备', '高密度印制电路板', '船舶'],
    '信息技术与数字技术': ['自然语言处理', '工业互联网', '人工智能芯片', '电子元件', 'SaaS解决方案', 
                       '数据中心', '信息服务', '智能硬件', '光学光电', '数据服务', '电子', '人工智能',
                       '光电子元器件', 'CMOS图像传感器', '信息', '工业软件', '电信', '数据分析', '信息技术'],
    '新能源与节能环保': ['新能源汽车', '光伏产品', '节能与环保', '新能源', '充电服务', '风电', '太阳能',
                     '光伏', '光伏电池', '光伏逆变器'],
    '消费零售与生活服务': ['体育用品', '家居', '智能家居', '休闲沙发', '化妆品', '烟草', '服装鞋帽',
                     '日用百货', '建材家居', '电子商务', '个人护理用品', '家具', '日用化学产品'],
    '医疗健康': ['医疗设备', '互联网医院', '医美', '医疗保健', '制药', '医药', '医疗卫生', '医疗器械', '医疗美容'],
    '房地产与建筑': ['房地产', '房产', '建筑', '物流地产', '建筑材料', '建材', '建筑装饰', '地产'],
    '金融与投资': ['金融', '贸易', '保险', '投资管理', '投资'],
    '文化传媒与教育': ['文旅', '教育', '文旅传媒', '电影制作', '传媒', '智慧文旅', '广告', '电视广播',
                   '传媒娱乐', '新闻传媒'],
    '物流与供应链': ['供应链', '物流', '物流仓储', '仓储物流'],
    '矿产与能源': ['矿业', '能源', '电力', '矿业开采', '矿产', '矿产资源'],
    '材料与化工': ['有色金属加工', '化学纤维', '电子材料', '化肥', '香料', '香精香料', '工程塑料',
               '工业气体', '化工', '有色金属', '新材料', '光伏材料', '工业涂料', '化纤', '化学原料',
               '造纸', '电池', '石油', '工业金属', '高分子材料', '皮革', 'LED'],
    '交通运输与航运': ['航运'],
    '服务业': ['酒店', '企业服务', '物业服务', '服务业'],
    '家电与消费电子': ['智能手机', '家电制造', '家电'],
    '汽车与零部件': ['新能源汽车零部件', '汽车制造', '汽车零部件'],
    '金属与冶炼': ['不锈钢', '矿冶', '矿产品', '钢铁', '电解铝', '氧化铝', '铝业'],
    '纺织与皮革': ['鞋业', '纺织', '服装'],
    '食品与餐饮产业': ['酒业', '制糖业', '食品', '食品制造', '食品与饮料', '食品和饮料', '乳制品',
                 '乳业', '保健品', '食用调味品', '面粉', '食用油', '餐饮', '保健酒', '餐饮连锁',
                 '饮料', '食品饮料', '餐饮设备'],
    '互联网行业': ['自媒体', '新消费', '上网平台', '物流配送', '互联网游戏', '智慧物流', '互联网技术',
               '能源互联网', '互联网服务']
}

# 创建反向映射（行业关键词 -> 大类）
keyword_to_category = {}
for category, keywords in industry_mapping.items():
    for keyword in keywords:
        keyword_to_category[keyword] = category

# 3. 行业分类统计
def analyze_industries(df):
    """分析各行业大类的富豪数量和财富分布"""
    industry_stats = {cat: {'count': 0, 'total_wealth': 0.0} for cat in industry_mapping.keys()}
    
    for _, row in df.iterrows():
        industries = row['行业列表']
        wealth = row['财富']
        matched_categories = set()
        
        for industry in industries:
            for keyword, category in keyword_to_category.items():
                if keyword in industry:
                    matched_categories.add(category)
                    break
        
        for cat in matched_categories:
            industry_stats[cat]['count'] += 1
            industry_stats[cat]['total_wealth'] += wealth
    
    stats_df = pd.DataFrame.from_dict(industry_stats, orient='index').reset_index()
    stats_df.columns = ['行业大类', '富豪数量', '总财富(亿)']
    stats_df['平均财富(亿)'] = stats_df['总财富(亿)'] / stats_df['富豪数量'].replace(0, 1)
    
    return stats_df.sort_values('总财富(亿)', ascending=False)

# 4. 生成热力图数据
def generate_heatmap_data(df):
    """生成行业-省份分布热力图数据"""
    top_provinces = df['省份'].value_counts().nlargest(10).index  # 取前10省份
    heatmap_data = pd.DataFrame(0, index=list(industry_mapping.keys()), columns=top_provinces)
    
    for _, row in df.iterrows():
        province = row['省份']
        if province not in top_provinces:
            continue
        industries = row['行业列表']
        matched_categories = set()
        for industry in industries:
            for keyword, category in keyword_to_category.items():
                if keyword in industry:
                    matched_categories.add(category)
                    break
        for cat in matched_categories:
            heatmap_data.loc[cat, province] += 1
    
    return heatmap_data

# 5. 单独的热力图可视化函数
def visualize_heatmap(heatmap_data):
    """在独立画布中展示热力图"""
    plt.figure(figsize=(16, 10))  # 更大的画布尺寸，适配行业和省份标签
    plt.title('行业-省份富豪分布热力图（前10省份）', fontsize=16)
    
    # 绘制热力图
    sns.heatmap(
        heatmap_data, 
        annot=True,          # 显示数值标注
        fmt='d',             # 数值格式为整数
        cmap='YlGnBu',       # 颜色方案
        cbar_kws={'label': '富豪数量'},  # 颜色条标签
        linewidths=.5        # 网格线宽度，增强可读性
    )
    
    plt.xlabel('省份', fontsize=12)
    plt.ylabel('行业大类', fontsize=12)
    plt.xticks(rotation=45)  # 省份标签旋转45度，避免重叠
    plt.yticks(rotation=0)   # 行业标签不旋转，保持垂直读取
    plt.tight_layout()       # 自动调整布局
    
    # 保存为单独文件
    output_file = get_output_file_path('行业-省份富豪分布热力图.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# 6. 其他可视化函数（保持原有图表）
def visualize_results(stats_df):
    """可视化行业分析结果（原有图表）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2024年胡润百富榜行业分析', fontsize=20)
    
    # 1. 富豪数量柱状图
    sns.barplot(x='富豪数量', y='行业大类', data=stats_df.sort_values('富豪数量', ascending=False), 
                ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('各行业大类富豪数量分布', fontsize=14)
    axes[0, 0].set_xlabel('富豪数量')
    axes[0, 0].set_ylabel('行业大类')
    
    # 2. 总财富柱状图
    sns.barplot(x='总财富(亿)', y='行业大类', data=stats_df, 
                ax=axes[0, 1], palette='magma')
    axes[0, 1].set_title('各行业大类总财富分布', fontsize=14)
    axes[0, 1].set_xlabel('总财富(亿)')
    axes[0, 1].set_ylabel('行业大类')
    
    # 3. 平均财富柱状图
    sns.barplot(x='平均财富(亿)', y='行业大类', data=stats_df.sort_values('平均财富(亿)', ascending=False), 
                ax=axes[1, 0], palette='plasma')
    axes[1, 0].set_title('各行业大类平均财富分布', fontsize=14)
    axes[1, 0].set_xlabel('平均财富(亿)')
    axes[1, 0].set_ylabel('行业大类')
    
    # 4. 财富占比饼图
    top_n = 10
    top_categories = stats_df.nlargest(top_n, '总财富(亿)')
    others = stats_df.nsmallest(len(stats_df) - top_n, '总财富(亿)')
    others_sum = others['总财富(亿)'].sum()
    pie_data = pd.concat([
        top_categories[['行业大类', '总财富(亿)']],
        pd.DataFrame([['其他', others_sum]], columns=['行业大类', '总财富(亿)'])
    ])
    
    axes[1, 1].pie(pie_data['总财富(亿)'], labels=pie_data['行业大类'], autopct='%1.1f%%', 
                   startangle=90, colors=sns.color_palette('tab20', n_colors=top_n+1))
    axes[1, 1].set_title(f'前{top_n}行业大类财富占比', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = get_output_file_path('胡润百富行业分析.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# 7. 其他维度分析
def other_dimension_analysis(df):
    """分析年龄、性别、出生地等维度"""
    gender_counts = df['性别'].fillna('未知').value_counts()
    
    age_df = df[df['年龄'] != '未知']
    age_df['年龄'] = age_df['年龄'].astype(int)
    
    province_counts = df['省份'].value_counts().nlargest(10)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 性别分布
    gender_colors = {'先生': 'skyblue', '女士': 'lightpink', '未知': 'lightgray'}
    colors = [gender_colors[gender] for gender in gender_counts.index]
    axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('富豪性别分布（含未知）')
    
    # 年龄分布
    sns.histplot(age_df['年龄'], bins=15, kde=True, ax=axes[1], color='green')
    axes[1].set_title('富豪年龄分布')
    axes[1].set_xlabel('年龄')
    
    # 省份分布
    sns.barplot(x=province_counts.values, y=province_counts.index, ax=axes[2], palette='rocket')
    axes[2].set_title('富豪主要出生地省份分布（前10）')
    axes[2].set_xlabel('人数')
    
    plt.tight_layout()
    output_file = get_output_file_path('富豪其他维度分析.png')
    plt.savefig(output_file, dpi=300)
    plt.show()

# 主函数
def main():
    # 加载数据（使用路径获取函数）
    data_file = get_data_file_path('assignment_1.csv')
    df = load_data(data_file)
    
    # 行业分析
    stats_df = analyze_industries(df)
    print("行业大类统计结果：")
    print(stats_df)
    
    # 生成热力图数据
    heatmap_data = generate_heatmap_data(df)
    
    # 原有图表可视化
    visualize_results(stats_df)
    
    # 单独展示热力图
    visualize_heatmap(heatmap_data)
    
    # 其他维度分析
    other_dimension_analysis(df)

if __name__ == "__main__":
    main()