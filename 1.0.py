import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP

# 【功能概述】：爬取新聞標題並進行情緒分析 

def start_project():
    print("--- 正在啟動：全自動新聞情緒分析儀 ---")
    
    # 1. 資料爬蟲 (以 Yahoo 新聞為例) 
    url = "https://tw.news.yahoo.com/technology"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 抓取新聞標題標籤
        titles = [t.get_text() for t in soup.find_all('h3')][:10] # 取前10則
        
        if not titles:
            print("未能抓取到標題，請檢查網頁結構。")
            return

        # 2. 資料分析與情緒分析 
        data = []
        for text in titles:
            s = SnowNLP(text)
            # score 越接近 1 越正面，越接近 0 越負面
            data.append({'標題': text, '情緒分數': round(s.sentiments, 2)})
        
        df = pd.DataFrame(data)
        print("\n--- 分析結果 ---")
        print(df)

        # 3. 數據視覺化 (折線圖) [cite: 18, 43]
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['情緒分數'], marker='o', color='b', linestyle='-')
        plt.title("News Sentiment Trend (Technology)")
        plt.xlabel("News Index")
        plt.ylabel("Sentiment Score")
        plt.ylim(0, 1)
        plt.grid(True)
        print("\n正在生成趨勢圖表...")
        plt.show()

    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    start_project()