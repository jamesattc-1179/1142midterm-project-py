import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP

class NewsScraper:
    """模組一：資料爬蟲 (對應規範 16)"""
    def __init__(self, target_url):
        self.url = target_url
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def get_titles(self):
        # 實作爬取新聞標題的邏輯
        try:
            response = requests.get(self.url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            # 範例：抓取特定標籤 (此處需根據目標網站調整)
            return [t.get_text() for t in soup.find_all('h3')]
        except Exception as e:
            print(f"爬取錯誤: {e}")
            return []

class EmotionAnalyzer:
    """模組二：資料分析與情緒計算 (對應規範 11, 18)"""
    def calculate_sentiment(self, text_list):
        results = []
        for text in text_list:
            s = SnowNLP(text)
            # sentiment 分數介於 0~1，越接近 1 越正面
            results.append({'content': text, 'score': s.sentiments})
        return pd.DataFrame(results)

class Visualizer:
    """模組三：視覺化呈現 (對應規範 18, 43)"""
    def plot_trend(self, df):
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['score'], marker='o', linestyle='-')
        plt.title("News Sentiment Trend")
        plt.xlabel("News Index")
        plt.ylabel("Sentiment Score")
        plt.grid(True)
        plt.show()