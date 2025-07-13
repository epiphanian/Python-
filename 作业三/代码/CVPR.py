import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://dblp.org/db/conf/cvpr/"
years = range(2020, 2025)
all_papers = []

print("开始爬取 CVPR 会议数据...")

for year in years:
    url = f"{base_url}cvpr{year}.html"
    print(f"正在爬取 {year} 年数据: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"爬取失败: {e}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')
    paper_entries = soup.find_all('li', class_='entry inproceedings')

    print(f"{year} 年共找到 {len(paper_entries)} 篇论文")

    for entry in paper_entries:
        title_tag = entry.find('span', class_='title')
        title = title_tag.text.strip() if title_tag else ""

        # 修改作者提取逻辑
        author_list = []
        author_tags = entry.find_all('span', itemprop='author')
        for author_tag in author_tags:
            author_name = author_tag.find('span', itemprop='name')
            if author_name:
                author_list.append(author_name.text.strip())

        # 将作者列表转换为所需的字符串格式
        authors = str(author_list)

        paper_info = {
            "title": title,
            "authors": authors,
            "year": year,
            "conference": "CVPR",
            "link": f"https://dblp.org{entry.find('a')['href']}"
        }
        all_papers.append(paper_info)

print(f"数据爬取完成，共获取 {len(all_papers)} 篇论文信息")

df = pd.DataFrame(all_papers)
df.to_csv('cvpr_papers.csv', index=False, encoding='utf-8')
print("数据已保存到 cvpr_papers.csv")