import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime

class IJCaiDataAnalyzer:
    def __init__(self):
        self.base_url = "https://dblp.org/db/conf/ijcai/"
        self.years = range(2020, datetime.now().year + 1)
        self.data = []
        self.keywords = []
    
    def fetch_papers(self):
        """从DBLP获取IJCAI会议论文数据"""
        print("开始从DBLP获取IJCAI论文数据...")
        
        for year in self.years:
            url = f"{self.base_url}ijcai{year}.html"
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找论文条目
                papers = soup.find_all('li', class_='entry inproceedings')
                
                for paper in papers:
                    title = paper.find('span', class_='title').text
                    authors = [a.text for a in paper.find_all('span', itemprop='author')]
                    year = year
                    conference = f"IJCAI {year}"
                    link = paper.find('nav', class_='publ').find('a')['href'] if paper.find('nav', class_='publ') else ""
                    
                    self.data.append({
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'conference': conference,
                        'link': link
                    })
                
                print(f"已获取IJCAI {year}年的{len(papers)}篇论文")
                
            except Exception as e:
                print(f"获取IJCAI {year}年数据时出错: {e}")
    
    def save_to_csv(self, filename='ijcai_papers.csv'):
        """将数据保存到CSV文件"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")
    
    def load_from_csv(self, filename='ijcai_papers.csv'):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(filename)
            # 处理authors列的字符串格式
            df['authors'] = df['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            self.data = df.to_dict('records')
            print(f"已从 {filename} 加载 {len(self.data)} 条论文数据")
        except FileNotFoundError:
            print(f"文件 {filename} 不存在，请先获取数据")
    
    def analyze_paper_trends(self):
        """分析论文数量趋势"""
        if not self.data:
            print("没有数据可供分析")
            return
        
        df = pd.DataFrame(self.data)
        trend = df.groupby('year').size()
        
        plt.figure(figsize=(10, 6))
        trend.plot(kind='bar', color='skyblue')
        plt.title('IJCAI Conference Paper Count (2020-Present)')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('ijcai_paper_trend.png')
        print("已生成论文数量趋势图: ijcai_paper_trend.png")
        plt.show()
    
    def extract_keywords(self):
        """从论文标题中提取关键词"""
        if not self.data:
            print("没有数据可供分析")
            return
        
        # 简单的关键词提取 - 实际应用中可能需要更复杂的NLP处理
        stop_words = {'a', 'an', 'the', 'and', 'or', 'for', 'in', 'on', 'with', 'via', 
                      'using', 'based', 'learning', 'approach', 'model', 'method'}
        
        for paper in self.data:
            title = paper['title'].lower()
            # 移除标点符号
            title = re.sub(r'[^\w\s]', '', title)
            words = title.split()
            # 过滤停用词和短词
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            self.keywords.extend(keywords)
    
    def generate_wordcloud(self):
        """生成关键词词云图"""
        if not self.keywords:
            self.extract_keywords()
        
        word_freq = Counter(self.keywords)
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=100).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('IJCAI Conference Paper Keywords Word Cloud (2020-Present)')
        plt.tight_layout()
        plt.savefig('ijcai_keywords_wordcloud.png')
        print("已生成关键词词云图: ijcai_keywords_wordcloud.png")
        plt.show()
    
    def predict_next_year_papers(self):
        """预测下一届会议的论文数量"""
        if not self.data:
            print("没有数据可供分析")
            return
        
        df = pd.DataFrame(self.data)
        trend = df.groupby('year').size()
        
        # 简单的线性预测 - 实际应用中可能需要更复杂的预测模型
        years = trend.index.values.reshape(-1, 1)
        counts = trend.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(years, counts)
        
        next_year = max(years)[0] + 1
        predicted = int(model.predict([[next_year]])[0])
        
        print(f"预测IJCAI {next_year}年将发表约{predicted}篇论文")
        
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.plot(years, counts, 'o-', label='Actual')
        plt.plot([next_year], [predicted], 'ro', label='Predicted')
        plt.title(f'IJCAI Paper Count Prediction for {next_year}')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('ijcai_paper_prediction.png')
        print("已生成论文数量预测图: ijcai_paper_prediction.png")
        plt.show()

if __name__ == "__main__":
    analyzer = IJCaiDataAnalyzer()
    
    # 获取数据 (第一次运行时使用)
    analyzer.fetch_papers()
    analyzer.save_to_csv()
    
    # # 从CSV加载数据 (后续运行使用)
    # analyzer.load_from_csv()
    
    # # 执行分析任务
    # analyzer.analyze_paper_trends()
    # analyzer.generate_wordcloud()
    # analyzer.predict_next_year_papers()