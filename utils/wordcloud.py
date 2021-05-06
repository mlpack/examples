from wordcloud import WordCloud, STOPWORDS

def cwordcloud(words, filename='output.png', height=2000, width=4000):
  words = words.replace(';', ' ')
  word_cloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=height, width=width).generate(words)
  word_cloud.to_file(filename)
  return 1
