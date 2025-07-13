import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
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

# 设置nltk语料库路径（使用系统默认路径）
try:
    # 尝试使用nltk的默认路径
    nltk.data.path.append(os.path.expanduser('~/nltk_data'))
    print("nltk搜索路径：", nltk.data.path)
except Exception as e:
    print(f"nltk路径设置警告：{e}")


class ConferenceDataAnalyzer:
    def __init__(self):
        # 会议与数据文件映射（使用路径获取函数）
        self.conference_files = {
            "AAAI": get_data_file_path("aaai_papers.csv"),
            "CVPR": get_data_file_path("cvpr_papers.csv"),
            "ICCV": get_data_file_path("iccv_papers.csv"),
            "ICML": get_data_file_path("icml_papers.csv"),
            "IJCAI": get_data_file_path("ijcai_papers.csv")
        }
        self.conference_cycles = {
            "AAAI": 1, "CVPR": 1, "ICCV": 2, "ICML": 1, "IJCAI": 1
        }
        self.data = {}
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update({
                'method', 'methods', 'system', 'systems', 'framework', 'frameworks',
                'approach', 'approaches', 'based', 'using', 'study', 'studies',
                'paper', 'papers', 'result', 'results', 'analysis', 'analyses'
            })
            print("停用词表加载成功（已扩展）")
        except LookupError:
            print(r"警告：停用词表未找到，请检查路径")
            self.stop_words = set()
        self.ai_core_terms = {'learning', 'network', 'algorithm', 'reinforcement',
                              'neural', 'deep', 'ai', 'vision', 'image'}

    def load_data(self):
        """加载数据"""
        for conf, file in self.conference_files.items():
            try:
                if not os.path.exists(file):
                    print(f"警告：{file}不存在，跳过")
                    continue
                df = pd.read_csv(file, encoding='utf-8')
                if 'authors' in df.columns and df['authors'].dtype == 'object':
                    df['authors'] = df['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                self.data[conf] = df
                print(f"已加载{conf}数据，共{len(df)}篇论文")
            except Exception as e:
                print(f"加载{conf}出错：{str(e)}")

    def analyze_paper_counts(self):
        """论文数量分析"""
        if not self.data:
            print("无数据可分析，请先加载数据")
            return
        for conf, df in self.data.items():
            if 'year' not in df.columns:
                print(f"{conf}缺少'year'列，无法分析")
                continue
            yearly_counts = df['year'].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            ax = yearly_counts.plot(kind='bar', color='skyblue' if conf != "ICCV" else '#3498db')
            cycle = self.conference_cycles[conf]
            plt.title(f'{conf} Paper Counts ({yearly_counts.index.min()}-{yearly_counts.index.max()}) - 每{cycle}年一次')
            plt.xlabel('Year')
            plt.ylabel('Number of Papers')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            if cycle > 1:
                ax.text(0.05, 0.95, f'每{cycle}年一次', transform=ax.transAxes, bbox=dict(facecolor='yellow', alpha=0.5))
            plt.tight_layout()
            save_path = get_output_file_path(f'{conf.lower()}_paper_counts.png')
            plt.savefig(save_path)
            plt.close()
            print(f"{conf}趋势图已保存为{save_path}")
        plt.figure(figsize=(12, 8))
        for conf, df in self.data.items():
            if 'year' in df.columns:
                yearly_counts = df['year'].value_counts().sort_index()
                marker = 'o-' if self.conference_cycles[conf] == 1 else 's--'
                plt.plot(yearly_counts.index, yearly_counts.values, marker, label=conf)
        plt.title('Paper Count Comparison Across Conferences (All Papers)')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = get_output_file_path('conference_paper_comparison.png')
        plt.savefig(output_file)
        plt.close()
        print("对比图已保存为conference_paper_comparison.png")

    def _get_wordnet_pos(self, treebank_tag):
        """将treebank词性标签转换为WordNet标签"""
        if treebank_tag.startswith('J'):
            return 'a'  # 形容词
        elif treebank_tag.startswith('V'):
            return 'v'  # 动词
        elif treebank_tag.startswith('N'):
            return 'n'  # 名词
        elif treebank_tag.startswith('R'):
            return 'r'  # 副词
        else:
            return 'n'  # 默认按名词处理

    def preprocess_text(self, text):
        """优化文本预处理：结合词性标注的精准词形还原"""
        if not text or not isinstance(text, str):
            return []
        # 分词
        tokens = word_tokenize(text.lower())
        # 过滤非字母字符和停用词
        filtered = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        # 词性标注
        tagged_tokens = pos_tag(filtered)
        # 结合词性进行词形还原
        lemmatized = [
            self.lemmatizer.lemmatize(token, pos=self._get_wordnet_pos(tag))
            for token, tag in tagged_tokens
        ]
        return lemmatized

    def extract_phrases(self, text):
        """优化短语提取：只保留双字和三字短语，避免单字与短语重复"""
        tokens = self.preprocess_text(text)
        # 提取双字词和三字词
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
        # 只返回短语（过滤单字）
        return bigrams + trigrams

    def analyze_keywords(self):
        """优化关键词分析：只生成2020 - 2024年每年综合五个会议的词云图"""
        if not self.data:
            print("无数据可分析，请先加载数据")
            return
        print("关键词分析（优化后：生成2020 - 2024年每年综合词云图）")
        for year in range(2020, 2025):
            all_titles_year = []
            for conf, df in self.data.items():
                if 'year' not in df.columns or 'title' not in df.columns:
                    print(f"{conf}缺少'year'或'title'列，无法分析")
                    continue
                year_df = df[df['year'] == year]
                if len(year_df) == 0:
                    continue
                titles = year_df['title'].dropna().astype(str).tolist()
                all_titles_year.extend(titles)

            if not all_titles_year:
                print(f"{year}年无数据，无法生成词云图")
                continue

            all_titles_text = ' '.join(all_titles_year)
            phrases = self.extract_phrases(all_titles_text)
            relevant_phrases = [
                phrase for phrase in phrases
                if any(term in phrase.split() for term in self.ai_core_terms)
            ]
            unique_phrases = []
            seen = set()
            for phrase in relevant_phrases:
                sorted_terms = tuple(sorted(phrase.split()))
                if sorted_terms not in seen:
                    seen.add(sorted_terms)
                    unique_phrases.append(phrase)
            phrase_counts = Counter(unique_phrases)
            wordcloud = WordCloud(
                width=1800,
                height=1200,
                background_color='white',
                max_words=40,
                collocations=False,
                font_path=None,  # 使用系统默认字体，不再硬编码字体路径
                min_font_size=10,
                max_font_size=200
            ).generate_from_frequencies(phrase_counts)
            plt.figure(figsize=(18, 12))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'2020 - 2024年综合五个会议 {year} 年专业短语词云图（优化后）',fontsize = 24)
            save_path = get_output_file_path(f'2020_2024_conferences_{year}_phrases_wordcloud_optimized.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"2020 - 2024年综合五个会议 {year} 年优化后短语词云图已保存为{save_path}")

    def predict_next_year_papers(self):
        """预测逻辑"""
        if not self.data:
            print("无数据可分析，请先加载数据")
            return
        for conf, df in self.data.items():
            if 'year' not in df.columns:
                print(f"{conf}缺少'year'列，无法预测")
                continue
            yearly_counts = df['year'].value_counts().sort_index().reset_index()
            yearly_counts.columns = ['year', 'count']
            if len(yearly_counts) < 2:
                print(f"{conf}数据量不足，无法预测")
                continue
            model = LinearRegression()
            model.fit(yearly_counts[['year']], yearly_counts['count'])
            last_year = yearly_counts['year'].max()
            cycle = self.conference_cycles[conf]
            next_year = last_year + cycle
            predicted_count = max(0, int(model.predict([[next_year]])[0]))
            plt.figure(figsize=(10, 6))
            plt.plot(yearly_counts['year'], yearly_counts['count'], 'o-', label='实际数量（全部论文）')
            plt.plot([next_year], [predicted_count], 'ro', label='预测数量（全部论文）')
            plt.title(f'{conf} Prediction for {next_year} (每{cycle}年一届)')
            plt.xlabel('Year')
            plt.ylabel('Number of Papers')
            plt.xticks(list(yearly_counts['year']) + [next_year], rotation=45)
            plt.legend()
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_path = get_output_file_path(f'{conf.lower()}_paper_prediction.png')
            plt.savefig(save_path)
            plt.close()
            print(f"预测{conf} {next_year}年约{predicted_count}篇论文，图已保存为{save_path}")


if __name__ == "__main__":
    analyzer = ConferenceDataAnalyzer()
    print("开始加载会议数据...")
    analyzer.load_data()
    print("\n开始分析论文数量趋势...")
    analyzer.analyze_paper_counts()
    print("\n开始提取专业短语并绘制优化后词云图...")
    analyzer.analyze_keywords()
    print("\n开始预测下一届论文数量...")
    analyzer.predict_next_year_papers()
    print("\n所有分析任务完成！")