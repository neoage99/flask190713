from konlpy.corpus import kolaw
import pandas as pd
from konlpy.tag import *
from konlpy.corpus import kobill
from nltk import Text
import matplotlib.pyplot as plt
from wordcloud import WordCloud

print(kolaw.fileids())

c = kolaw.open('constitution.txt').read()
print(c[:40])

print(kobill.fileids())

d = kobill.open('1809890.txt').read()
print(d[:40])


hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
#mecab = Mecab()  # 일본어
okt = Okt()

hannanum.nouns(c[:40])
kkma.nouns(c[:40])
# komoran은 빈줄이 있으면 에러가 남
komoran.nouns("\n".join([s for s in c[:40].split("\n") if s]))
#mecab.nouns(c[:40])
okt.nouns(c[:40])
hannanum.morphs(c[:40])
kkma.morphs(c[:40])
# komoran은 빈줄이 있으면 에러가 남
komoran.morphs("\n".join([s for s in c[:40].split("\n") if s]))
#mecab.morphs(c[:40])
okt.morphs(c[:40])

hannanum.pos(c[:40])
kkma.pos(c[:40])
# komoran은 빈줄이 있으면 에러가 남
komoran.pos("\n".join([s for s in c[:40].split("\n") if s]))
#mecab.pos(c[:40])
okt.pos(c[:40])

okt.tagset
tagsets = pd.DataFrame()
N = 67
tagsets["Hannanum-기호"] = list(hannanum.tagset.keys()) + list("*" * (N - len(hannanum.tagset)))
tagsets["Hannanum-품사"] = list(hannanum.tagset.values()) + list("*" * (N - len(hannanum.tagset)))
tagsets["Kkma-기호"] = list(kkma.tagset.keys()) + list("*" * (N - len(kkma.tagset)))
tagsets["Kkma-품사"] = list(kkma.tagset.values()) + list("*" * (N - len(kkma.tagset)))
tagsets["Komoran-기호"] = list(komoran.tagset.keys()) + list("*" * (N - len(komoran.tagset)))
tagsets["Komoran-품사"] = list(komoran.tagset.values()) + list("*" * (N - len(komoran.tagset)))
#tagsets["Mecab-기호"] = list(mecab.tagset.keys()) + list("*" * (N - len(mecab.tagset)))
#tagsets["Mecab-품사"] = list(mecab.tagset.values()) + list("*" * (N - len(mecab.tagset)))
tagsets["OKT-기호"] = list(okt.tagset.keys()) + list("*" * (N - len(okt.tagset)))
tagsets["OKT-품사"] = list(okt.tagset.values()) + list("*" * (N - len(okt.tagset)))
tagsets


kolaw = Text(okt.nouns(c), name="kolaw")
kolaw.plot(30)
plt.show()


# 자신의 컴퓨터 환경에 맞는 한글 폰트 경로를 설정
font_path = 'C:\Windows\Fonts/Malgun.ttf'

wc = WordCloud(width = 1000, height = 600, background_color="white", font_path=font_path)
plt.imshow(wc.generate_from_frequencies(kolaw.vocab()))
plt.axis("off")
plt.show()


