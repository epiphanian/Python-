import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import time
from sklearn.linear_model import LinearRegression
import numpy as np

class ICCVConferenceAnalyzer:
    def __init__(self):
        self.base_url = "https://dblp.org/db/conf/iccv/"
        self.years = [2021, 2023]  # ICCV是奇数年会议: 2021, 2023
        self.papers_data = []
        
    def fetch_conference_data(self):
        """爬取ICCV会议论文数据"""
        for year in self.years:
            url = f"{self.base_url}iccv{year}.html"
            print(f"正在爬取ICCV {year}的数据: {url}")
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                papers = soup.find_all('li', class_='entry inproceedings')
                
                for paper in papers:
                    title = paper.find('span', class_='title').text
                    authors = [a.text for a in paper.find_all('span', itemprop='author')]
                    year = year
                    conference = f"ICCV {year}"
                    link = paper.find('nav', class_='publ').find('a')['href'] if paper.find('nav', class_='publ') else None
                    
                    self.papers_data.append({
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'conference': conference,
                        'link': link
                    })
                
                print(f"ICCV {year} 爬取完成，共获取 {len(papers)} 篇论文")
                time.sleep(3)  # 礼貌性延迟，避免对服务器造成压力
                
            except Exception as e:
                print(f"爬取ICCV {year}时出错: {str(e)}")
                continue
    
    def save_to_csv(self, filename="iccv_papers.csv"):
        """将数据保存到CSV文件"""
        df = pd.DataFrame(self.papers_data)
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")
    
    def analyze_paper_counts(self):
        """分析并绘制每年论文数量趋势图"""
        if not self.papers_data:
            print("没有数据可供分析，请先爬取数据")
            return
        
        df = pd.DataFrame(self.papers_data)
        yearly_counts = df['year'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        yearly_counts.plot(kind='bar', color='#3498db')
        plt.title('ICCV Conference Paper Counts (2021-2023)')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('iccv_paper_counts.png')
        plt.show()
        print("论文数量趋势图已保存为 iccv_paper_counts.png")
    
    def analyze_keywords(self):
        """分析论文标题中的关键词并生成词云"""
        if not self.papers_data:
            print("没有数据可供分析，请先爬取数据")
            return
        
        # 提取所有标题文本
        titles = ' '.join([paper['title'] for paper in self.papers_data])
        
        # 预处理：移除标点符号、转换为小写、分词
        words = re.findall(r'\b[a-zA-Z]{4,}\b', titles.lower())  # 只保留长度>=4的单词
        stop_words = set(['using', 'based', 'with', 'learning', 'approach', 
                         'model', 'method', 'deep', 'network', 'vision', 'image'])
        filtered_words = [word for word in words if word not in stop_words]
        
        # 统计高频词
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(50)
        print("Top 50高频关键词:")
        for word, count in top_words:
            print(f"{word}: {count}")
        
        # 生成词云
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='plasma',
                             max_words=100).generate_from_frequencies(word_counts)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('ICCV Conference Keywords Word Cloud (2021-2023)')
        plt.tight_layout()
        plt.savefig('iccv_keywords_wordcloud.png')
        plt.show()
        print("关键词词云图已保存为 iccv_keywords_wordcloud.png")
    
    def predict_next_year_papers(self):
        """预测下一年的论文数量(简单线性回归)"""
        if not self.papers_data:
            print("没有数据可供分析，请先爬取数据")
            return
        
        df = pd.DataFrame(self.papers_data)
        yearly_counts = df['year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['year', 'count']
        
        # 简单线性回归预测
        model = LinearRegression()
        model.fit(yearly_counts[['year']].astype(int), yearly_counts['count'])

        # 确保year为整数类型，避免类型错误
        max_year = int(yearly_counts['year'].max())
        next_year = max_year + 2  # ICCV是每两年一次，下届是2025
        predicted_count = int(model.predict([[next_year]])[0])

        print(f"基于线性回归预测，ICCV {next_year}可能会有 {predicted_count} 篇论文")
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_counts['year'], yearly_counts['count'], 'o-', label='Actual Counts')
        plt.plot([next_year], [predicted_count], 'ro', label='Predicted')
        plt.title('ICCV Conference Paper Counts Prediction')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.xticks(list(yearly_counts['year']) + [next_year])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('iccv_paper_prediction.png')
        plt.show()
        print("论文数量预测图已保存为 iccv_paper_prediction.png")

if __name__ == "__main__":
    analyzer = ICCVConferenceAnalyzer()
    analyzer.fetch_conference_data()
    analyzer.save_to_csv()
    # analyzer.analyze_paper_counts()
    # analyzer.analyze_keywords()
    # analyzer.predict_next_year_papers()