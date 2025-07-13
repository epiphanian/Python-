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
import json
import re

cookies = {
    'PHPSESSID': 'lepni7550tirmqlftc96n0kiq3',
    'Hm_lvt_12e4883fd1649d006e3ae22a39f97330': '1751564834',
    'HMACCOUNT': '8D169305DF745D06',
    'Hm_lvt_692bd5f9c07d3ebd0063062fb0d7622f': '1751564834',
    '_gid': 'GA1.2.1736830114.1751564835',
    '_ga_9FDP3NWFMS': 'GS2.1.s1751564833$o1$g1$t1751565229$j60$l0$h0',
    '_ga': 'GA1.2.1433837394.1751564834',
    '_gat_UA-66069030-3': '1',
    'Hm_lpvt_692bd5f9c07d3ebd0063062fb0d7622f': '1751565230',
    'Hm_lpvt_12e4883fd1649d006e3ae22a39f97330': '1751565230',
}

headers = {
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Referer': 'https://www.zhcw.com/',
    'Sec-Fetch-Dest': 'script',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    # 'Cookie': 'PHPSESSID=lepni7550tirmqlftc96n0kiq3; Hm_lvt_12e4883fd1649d006e3ae22a39f97330=1751564834; HMACCOUNT=8D169305DF745D06; Hm_lvt_692bd5f9c07d3ebd0063062fb0d7622f=1751564834; _gid=GA1.2.1736830114.1751564835; _ga_9FDP3NWFMS=GS2.1.s1751564833$o1$g1$t1751565229$j60$l0$h0; _ga=GA1.2.1433837394.1751564834; _gat_UA-66069030-3=1; Hm_lpvt_692bd5f9c07d3ebd0063062fb0d7622f=1751565230; Hm_lpvt_12e4883fd1649d006e3ae22a39f97330=1751565230',
}

params = {
    'callback': 'jQuery112209620060368232614_1751565229883',
    'transactionType': '10001002',
    'lotteryId': '281',
    'issue': '25070',
    'tt': '0.3211447477930204',
    '_': '1751565229892',
}
#开奖号码前5位为前期号码，后2位为后期号码
csv_headers = ['期号', '开奖时间',  '周几','开奖号码', '奖池金额', '销售金额']

# 第一段：25073 → 25001（共73期）
high_issues = range(25073, 25000, -1)  # [25073, 25072, ..., 25001]

# 第二段：24152 → 241??（补足剩余的27期）
low_issues = range(24152, 24125, -1)  # [24152, 24151, ..., 24126]

# 合并两部分
all_issues = list(high_issues) + list(low_issues)


#动态网页爬取， 采用json数据筛选数据
with open('大乐透开奖数据.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    writer.writeheader()  # 只需要写一次表头
    
    url = ['https://jc.zhcw.com/port/client_json.php?callback=jQuery112209620060368232614_1751565229883&transactionType=10001002&lotteryId=281&issue={}&tt=0.3211447477930204&_=1751565229892'.format(i) for i in all_issues]
    for index, url_ in enumerate(url, 1):
        try:
            params['issue'] = str(re.search(r'issue=(\d{5})', url_).group(1))  # 获取issue参数
            response = requests.get(url_, cookies=cookies, headers=headers, params=params)
            response.raise_for_status()  # 检查请求是否成功
            
            json_data = re.search(r'^\w+\((.*)\);?$', response.text).group(1)    #处理JSONP数据
            file = json.loads(json_data)
            
            openTime = file['openTime']
            frontWinningNum = file['frontWinningNum']
            backWinningNum = file['backWinningNum']
            WinningNum = frontWinningNum + ' ' +backWinningNum
            issue = file['issue']
            prizePoolMoney = file['prizePoolMoney']
            saleMoney = file['saleMoney']
            openweek = file['week']

            writer.writerow({
                '期号': issue,
                '开奖时间': openTime,
                '周几': openweek,
                '开奖号码': WinningNum,
                '奖池金额': prizePoolMoney,
                '销售金额': saleMoney,
            })

            print('成功获取{}期(第{}期)数据'.format(issue, index))
        
        except Exception as e:
            print('获取第{}期数据失败,原因:{}'.format(index,e))
            continue

    

        











