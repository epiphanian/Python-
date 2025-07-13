import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time

def get_weather_data(city='dalian', start_date=datetime(2025, 1, 1), end_date=datetime(2025, 6, 30)):
    base_url = f"https://www.tiangihoubao.com/weather/top/{city}.html"  # 实际需要找到历史数据接口
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
    }

    all_data = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        try:
            # 实际URL需要根据网站结构调整
            url = f"http://www.tianqihoubao.com/lishi/{city}/{date_str}.html"
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # 使用BeautifulSoup解析页面
            soup = BeautifulSoup(response.text, 'html.parser')

            weather_table = soup.find('table', class_='weather-table')
            if not weather_table:
                print(f"未找到天气表格: {url}")
                current_date += timedelta(days=1)
                continue

            # 直接获取tbody中的行（跳过表头）
            tbody = weather_table.find('tbody')
            if not tbody:
                print(f"未找到表格内容: {url}")
                current_date += timedelta(days=1)
                continue

            rows = tbody.find_all('tr')

            # 验证行数是否足够
            if len(rows) < 3:
                print(f"表格数据不完整: {url}")
                current_date += timedelta(days=1)
                continue

            # 提取天气状况
            weather_row = rows[0].find_all('td')
            if len(weather_row) < 3:
                print(f"天气状况行格式异常: {url}")
                current_date += timedelta(days=1)
                continue

            # 提取天气文本（跳过img标签）
            day_weather = weather_row[1].get_text(strip=True)
            night_weather = weather_row[2].get_text(strip=True)

            # 提取温度
            temp_row = rows[1].find_all('td')
            if len(temp_row) < 3:
                print(f"温度行格式异常: {url}")
                current_date += timedelta(days=1)
                continue

            high_temp = temp_row[1].get_text(strip=True).replace('℃', '')
            low_temp = temp_row[2].get_text(strip=True).replace('℃', '')

            # 提取风力
            wind_row = rows[2].find_all('td')
            if len(wind_row) < 3:
                print(f"风力行格式异常: {url}")
                current_date += timedelta(days=1)
                continue

            day_wind = wind_row[1].get_text(strip=True)
            night_wind = wind_row[2].get_text(strip=True)

            # 构建天气数据字典
            weather_data = {
                '日期': current_date.strftime('%Y-%m-%d'),
                '白天天气': day_weather,
                '夜间天气': night_weather,
                '最高温度': high_temp,
                '最低温度': low_temp,
                '白天风力': day_wind,
                '夜间风力': night_wind,
            }

            all_data.append(weather_data)
            print(f"已获取 {current_date.strftime('%Y-%m-%d')} 数据")

        except Exception as e:
            print(f"获取 {current_date.strftime('%Y-%m-%d')} 数据失败: {e}")
        #finally:
            #ime.sleep(1)  # 防止请求过于频繁

        current_date += timedelta(days=1)

    df = pd.DataFrame(all_data)
    df.to_csv(f'{city}_weather_2025_01_06.csv', index=False)
    return df

# 执行爬取
weather_df = get_weather_data()