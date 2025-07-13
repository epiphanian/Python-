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

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 明确指定nltk语料库路径
nltk.data.path = [
    r'C:\Users\Polaris\AppData\Roaming\nltk_data',
    r'C:\Users\Polaris\AppData\Roaming\nltk_data\tokenizers\punkt',
    r'C:\Users\Polaris\AppData\Roaming\nltk_data\corpora\stopwords',
    r'C:\Users\Polaris\AppData\Roaming\nltk_data\taggers\averaged_perceptron_tagger',
    r'C:\Users\Polaris\AppData\Roaming\nltk_data\corpora\wordnet'
]
print("nltk搜索路径：", nltk.data.path)


class ConferenceDataAnalyzer:
    def __init__(self):
        # 会议与数据文件映射（原有）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "数据")
        self.conference_files = {
            "AAAI": os.path.join(data_dir, "aaai_papers.csv"),
            "CVPR": os.path.join(data_dir, "cvpr_papers.csv"),
            "ICCV": os.path.join(data_dir, "iccv_papers.csv"),
            "ICML": os.path.join(data_dir, "icml_papers.csv"),
            "IJCAI": os.path.join(data_dir, "ijcai_papers.csv")
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
        """加载数据（原有功能）"""
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
        """论文数量分析（原有功能）"""
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
            save_path = f'{conf.lower()}_paper_counts.png'
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
        plt.savefig('conference_paper_comparison.png')
        plt.close()
        print("对比图已保存为conference_paper_comparison.png")

    def _get_wordnet_pos(self, treebank_tag):
        """词性标签转换（原有功能）"""
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
        """文本预处理（原有功能）"""
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
        """短语提取（原有功能）"""
        tokens = self.preprocess_text(text)
        # 提取双字词和三字词
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
        # 只返回短语（过滤单字）
        return bigrams + trigrams

    def analyze_keywords(self):
        """分会议关键词分析（原有功能）"""
        if not self.data:
            print("无数据可分析，请先加载数据")
            return
        print("关键词分析（优化后：减少重复短语）")
        for conf, df in self.data.items():
            if 'year' not in df.columns or 'title' not in df.columns:
                print(f"{conf}缺少'year'或'title'列，无法分析")
                continue
            years = sorted(df['year'].unique())
            for year in years:
                year_df = df[df['year'] == year]
                if len(year_df) == 0:
                    continue
                # 合并标题文本
                titles = ' '.join(year_df['title'].dropna().astype(str))
                # 提取短语（仅双字和三字）
                phrases = self.extract_phrases(titles)
                # 过滤不包含核心术语的短语
                relevant_phrases = [
                    phrase for phrase in phrases
                    if any(term in phrase.split() for term in self.ai_core_terms)
                ]
                # 二次过滤：移除语义重复的短语
                unique_phrases = []
                seen = set()
                for phrase in relevant_phrases:
                    sorted_terms = tuple(sorted(phrase.split()))
                    if sorted_terms not in seen:
                        seen.add(sorted_terms)
                        unique_phrases.append(phrase)
                # 统计短语频率
                phrase_counts = Counter(unique_phrases)
                # 生成词云
                wordcloud = WordCloud(
                    width=1800, height=1200,
                    background_color='white',
                    max_words=40,
                    collocations=False,
                    font_path='simhei.ttf',
                    min_font_size=10,
                    max_font_size=200
                ).generate_from_frequencies(phrase_counts)
                # 保存词云图
                plt.figure(figsize=(18, 12))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'{conf} {year}年专业短语词云图（优化后）')
                save_path = f'{conf.lower()}_{year}_phrases_wordcloud_optimized.png'
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"{conf} {year}年优化后短语词云图已保存为{save_path}")

    def analyze_keywords_combined(self):
        """新增：综合五个会议的关键词分析（2020-2024年）"""
        if not self.data:
            print("无数据可分析，请先加载数据")
            return
        print("开始生成2020-2024年综合五个会议的词云图")
        for year in range(2020, 2025):
            all_titles = []
            for conf, df in self.data.items():
                if 'year' not in df.columns or 'title' not in df.columns:
                    print(f"{conf}缺少'year'或'title'列，跳过该会议数据")
                    continue
                year_df = df[df['year'] == year]
                if len(year_df) == 0:
                    continue
                all_titles.extend(year_df['title'].dropna().astype(str))
            if not all_titles:
                print(f"{year}年无可用数据，跳过")
                continue
            # 合并所有会议的标题文本
            titles_text = ' '.join(all_titles)
            # 复用原有短语提取和过滤逻辑
            phrases = self.extract_phrases(titles_text)
            relevant_phrases = [
                phrase for phrase in phrases
                if any(term in phrase.split() for term in self.ai_core_terms)
            ]
            # 去重处理
            unique_phrases = []
            seen = set()
            for phrase in relevant_phrases:
                sorted_terms = tuple(sorted(phrase.split()))
                if sorted_terms not in seen:
                    seen.add(sorted_terms)
                    unique_phrases.append(phrase)
            # 生成并保存词云
            phrase_counts = Counter(unique_phrases)
            wordcloud = WordCloud(
                width=1800, height=1200,
                background_color='white',
                max_words=40,
                collocations=False,
                font_path='simhei.ttf',
                min_font_size=10,
                max_font_size=200
            ).generate_from_frequencies(phrase_counts)
            plt.figure(figsize=(18, 12))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'综合五个会议 {year}年专业短语词云图（优化后）')
            save_path = f'combined_{year}_phrases_wordcloud_optimized.png'
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"综合五个会议 {year}年词云图已保存为{save_path}")

    def predict_next_year_papers(self):
        """论文数量预测（原有功能）"""
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
            save_path = f'{conf.lower()}_paper_prediction.png'
            plt.savefig(save_path)
            plt.close()
            print(f"预测{conf} {next_year}年约{predicted_count}篇论文，图已保存为{save_path}")


if __name__ == "__main__":
    analyzer = ConferenceDataAnalyzer()
    print("开始加载会议数据...")
    analyzer.load_data()
    
    # 可根据需要注释/取消注释以下功能
    print("\n开始分析论文数量趋势...")
    analyzer.analyze_paper_counts()
    
    print("\n开始分会议提取关键词并绘制词云图...")
    analyzer.analyze_keywords()
    
    print("\n开始生成2020-2024年综合会议词云图...")
    analyzer.analyze_keywords_combined()
    
    print("\n开始预测下一届论文数量...")
    analyzer.predict_next_year_papers()
    
    print("\n所有分析任务完成！")