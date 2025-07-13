##对不同行业的富豪数量以及财富值进行统计
#分析各行业的发展态势。
#例如富豪在年龄、性别、出生地等方面的分布情况

#-*- coding:utf-8 -*-
import requests
from lxml import etree
import time
import csv

index=0
with open("assignment_1.csv","w",encoding="utf-8",newline="") as f:
    for i in range(55):
        url = 'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&search=&offset={}&limit=20'.format(i*20)
        file = requests.get(url).json() #这里返回json格式的页面，所以用json来解析，而不是用text
        time.sleep(2)
        for j in range(20):
            if(i==54 and j==14): #这里是为了防止最后一页的富豪信息不全，所以跳出循环
                break
            dict = file['rows'][j] #分别取出文件中‘rows’ 下对应的第[j]位富豪的信息
            personal_info = dict['hs_Character'][0]
        
            name = dict['hs_Rank_Rich_ChaName_Cn']
            age = personal_info['hs_Character_Age']
            sex = personal_info['hs_Character_Gender']
            birthplace = personal_info['hs_Character_BirthPlace_Cn']
            education = personal_info['hs_Character_Education_Cn']
            Schools = personal_info['hs_Character_School_En']
            wealth = dict['hs_Rank_Rich_Wealth']
            industry = dict['hs_Rank_Rich_Industry_Cn']
            ranking = dict['hs_Rank_Rich_Ranking']
            company = dict['hs_Rank_Rich_ComName_Cn']
            permanent_residence = personal_info['hs_Character_Permanent_Cn']
            wealth_change = dict['hs_Rank_Rich_Wealth_Change']
            ranking_change = dict['hs_Rank_Rich_Ranking_Change']
    
            header = ["排名","财富","姓名","性别","出生地","学历","学校","行业","公司","常驻地","年龄","财富变化","排名变化"]

            #写入内容必须是字典形式
            content = {header[0]:ranking,header[1]:wealth,header[2]:name,header[3]:sex,header[4]:birthplace,header[5]:education,header[6]:Schools,header[7]:industry,header[8]:company,header[9]:permanent_residence,header[10]:age,header[11]:wealth_change,header[12]:ranking_change}

            #newline 参数是为了插入内容没有空行
        
            dw = csv.DictWriter(f,fieldnames=header)
            if index == 0:
                dw.writeheader()
                index += 1
            dw.writerow(content)

        
           




