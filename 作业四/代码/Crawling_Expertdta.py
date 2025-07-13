'''
请以中国体彩网(https://www.zhcw.com)为数据源，完成以下任务：
(1)爬取截至2025年7月1日之前100期的大乐透开奖数据和中奖情况(链接:https://www.zhcw.com/kjxx/dlt/),
    分析大乐透总销售额随开奖日期的变化趋势并预测2025年7月1日之后最近一期的销售额。
(2)根据爬取的大乐透数据对前区号码与后区号码进行频率统计与可视化，分析其历史分布规律。
    根据你发现的规律或者采用其他预测方法推荐一组大乐透号码,作为2025年7月1日之后最近一期的投注号码。
(3)根据爬取的大乐透数据，分别统计周一、周三、周六的大乐透开奖号码和总销售额。对比不同开奖日之间的号码分布与销售额特征，分析是否存在显著差异或相似性。
(4)爬取任意一个彩种中20位以上专家的公开数据,对专家的基本属性(彩龄、发文量等)和表现(中奖情况)进行统计分析，并通过可视化展示其分布规律、相互关系或对中奖率的影响。

'''
import requests
from lxml import etree
import csv

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

csv_headers = ['专家名字', '彩龄', '发文量', '大乐透一等奖次数', '大乐透二等奖次数', '大乐透三等奖次数','最近10次大乐透平均得分']

with open("大乐透专家数据.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    writer.writeheader()
    
    for page in range(1, 21):
        url = f'https://i.cmzj.net/expert/queryExpert?limit=10&page={page}&sort=0'
        response = requests.get(url, headers=headers)
        experts = response.json()['data']
        
        for expert in experts[:10]:  # 取前10条
            expert_id = expert['expertId']
            detail_url = f'https://i.cmzj.net/expert/queryExpertById?expertId={expert_id}'
            
            try:
                response = requests.get(detail_url, headers=headers)
                file = response.json()
                
                print("专家页面:", detail_url)
                print("HTML 长度:", len(response.text))

                
                score_urls = f'https://i.cmzj.net/expert/queryExpertHistoryScore?expertId={expert_id}&lottery=4&limit=10&page=1'
                score = 0
                count = 0
                response = requests.get(score_urls, headers=headers)
                score_data = response.json()['data']
                print("score_data:", score_data)  # 调试用

                for issue in score_data:
                    for rec in issue:
                        if 'sumScore' in rec:
                            score += rec['sumScore']
                            count += 1
                avg_score = score / count if count > 0 else 0
                    
                
                
                # 提取数据
                data = {
                    '专家名字': file['data']['name'],
                    '彩龄': file['data']['age'],
                    '发文量': file['data']['articles'],
                    '最近10次大乐透平均得分': avg_score,
                    '大乐透一等奖次数': file['data']['dltOne'],
                    '大乐透二等奖次数': file['data']['dltTwo'],
                    '大乐透三等奖次数': file['data']['dltThree']
                }
                
                print("提取的数据:", data)
                writer.writerow(data)
                
            except Exception as e:
                print(f'获取专家 {expert_id} 失败: {e}')
                continue









