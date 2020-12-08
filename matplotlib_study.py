# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:26:58 2020

@author: msi
"""

# Matplotlib
# =============================================================================

# Matplotlib 특징

# 파이썬의 대표적인 과학 계산용 그래프 라이브러리
# 선 그래프, 히스토그램, 산점도 등의 고품질 그래프 제공
# 저수준 api를 사용한 다양한 시각화 기능 제공
# 다양한 운영체제와 그래픽 백엔드에서 동작

# matplotlib, numpy, pandas호출
print('matplotlib, numpy, pandas호출')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib 현재 버전 출력
print('matplotlib 현재 버전 출력')
print(mpl.__version__)
print()

plt.style.use(['seaborn-notebook'])

# 라인 플롯

# plt.figure : 축과 그래프 , 텍스트 레이블을 표시하는 모든 객체를 포함하는 컨테이너
# plt.axes : 눈금과 레이블이 있는 테두리 박스로 시각화를 형성하는 플롯 요소 포함

# 하얀 도화지(플롯) 출력
print('하얀 도화지(플롯) 출력')
fig = plt.figure()
ax = plt.axes()
plt.show()
# plt.show() : 파이참에서 시각화를 하기위한 명령어
print()

# 지그재그 직선 출력
print('지그재그 직선 출력')
fig = plt.figure()
plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1] * 5)
plt.show()
print()

# sin곡선 출력
print('sin 곡선 출력')
x = np.arange(0, 10, 0.01)
fig = plt.figure()
plt.plot(x, np.sin(x))
plt.show() 
print()

# sin, cos 곡선 출력
print('sin, cos 곡선 출력')
plt.plot(x, np.sin(x));
plt.plot(x, np.cos(x));
plt.show()
print()

# 랜덤한 50개의 숫자 누적 합
print('랜덤한 50개의 숫자 누적 합')
plt.plot(np.random.randn(50).cumsum());
plt.show()
print()

# 라인 스타일

# '-' : 'solid'
print("'-' : 'solid'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'solid')
plt.show()
print()

# '--' : 'dashed'
print("'--' : 'dashed'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dashed')
plt.show()
print()

# '-.' : 'dashdot'
print("'-.' : 'dashdot'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dashdot')
plt.show()
print()

# ':' : 'dotted'
print("':' : 'dotted'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dotted')
plt.show()
print()

# 모두 합치기
print('모두 합치기')
plt.plot(np.random.randn(50).cumsum(), linestyle = '-')
plt.plot(np.random.randn(50).cumsum(), linestyle = '--')
plt.plot(np.random.randn(50).cumsum(), linestyle = '-.')
plt.plot(np.random.randn(50).cumsum(), linestyle = ':')
plt.show()
print()

# 색상 스타일

# 색상 지정 출력
print('색상 지정 출력1')
plt.plot(np.random.randn(50).cumsum(), color = 'g')
plt.plot(np.random.randn(50).cumsum(), color = '#1243FF')
plt.plot(np.random.randn(50).cumsum(), color = (0.2, 0.4, 0.6))
plt.plot(np.random.randn(50).cumsum(), color = 'darkblue')
plt.show()
print()

print('색상 지정 출력2') 
plt.plot(np.random.randn(50).cumsum(), color = 'skyblue')
plt.plot(np.random.randn(50).cumsum(), color = 'dodgerblue')
plt.plot(np.random.randn(50).cumsum(), color = 'royalblue')
plt.plot(np.random.randn(50).cumsum(), color = 'navy')
plt.show()
print()

# 라인스타일과 색상 함께 지정 출력
print('라인스타일과 색상 함께 지정 출력')
plt.plot(np.random.randn(50).cumsum(), 'b-')
plt.plot(np.random.randn(50).cumsum(), 'g--')
plt.plot(np.random.randn(50).cumsum(), 'c-.')
plt.plot(np.random.randn(50).cumsum(), 'm:')
plt.show()
print()

# 플롯 축

# 랜덤한 50개 출력
print('랜덤한 50개 출력')
plt.plot(np.random.randn(50))
plt.show()
print()

# x,y축 범위 지정
print('x,y축 범위 지정1')
plt.plot(np.random.randn(50))
plt.xlim(-1, 50)
plt.ylim(-5, 5)
plt.show()
print()

print('x,y축 범위 지정2')
plt.plot(np.random.randn(50))
plt.axis([-1, 50, -5, 5])
plt.show()
print()

# x,y축 범위에 딱 맞게 출력
print('x,y축 범위에 딱 맞게 출력')
plt.plot(np.random.randn(50))
plt.axis('tight')
plt.show()
print()
# x,y축 범위 널널하게 출력
print('x,y축 범위 널널하게 출력')
plt.plot(np.random.randn(50))
plt.axis('equal')
plt.show()
print()

# 플롯 레이블

# 시각화 된 표에 이름 정해서 출력
print('시각화 된 표에 이름 정해서 출력')
plt.plot(np.random.randn(50))
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.show()
print()

# 각 그래프에 이름 정해주고 범례 출력
print('각 그래프에 이름 정해주고 범례 출력')
plt.plot(np.random.randn(50), label = 'A')
plt.plot(np.random.randn(50), label = 'B')
plt.plot(np.random.randn(50), label = 'C')
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.legend()
plt.show()
print()

# 폰트 관리자

# 지정해서 쓸수있는 폰트
print('지정해서 쓸수있는 폰트')
print(set(f.name for f in mpl.font_manager.fontManager.ttflist))
print()

font1 = {'family' : 'Dejavu Sans', 'size' : 24, 'color': 'black'}
font2 = {'family' : 'Jokerman', 'size' : 18, 'weight' : 'bold', 'color': 'darkred'}
font3 = {'family' : 'STIXGeneral', 'size' : 16, 'weight' : 'light', 'color': 'black'}

# 각 그래프 이름에 폰트와 크기 컬러 굵기 지정
print('각 그래프 이름에 폰트와 크기 컬러 굵기 지정')
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.title('title', fontdict = font1)
plt.xlabel('x', fontdict = font2)
plt.ylabel('random.randn', fontdict = font3)
plt.show()
print()

# 플롯 범례

# 범례 기본값(best)
print('범례 기본값(best)')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.axis('equal')
ax.legend()
fig.show()
plt.show()
print()

# 범례 lower right
print('범례 lower right')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(loc = 'lower right')
fig.show()
plt.show()
print()

# 범례 upper center
print('범례 upper center')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(loc = 'upper center', frameon = False, ncol = 2)
fig.show()
plt.show()
print()

# 범례에 여려가지 효과
print('범례에 여려가지 효과')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
fig.show()
plt.show()
print()

# 범례 3개만 표현1
print('범례 3개만 표현1')
plt.figure(figsize = (8, 4))
x = np.linspace(0, 10, 1000)
y = np.cos(x[:, np.newaxis] * np.arange(0, 2, 0.2))
lines = plt.plot(x, y)
plt.legend(lines[:3], ['c1', 'c2', 'c3'])
plt.show()
print()

# 범례 3개만 표현2
print('범례 3개만 표현2')
plt.figure(figsize = (8, 4))
x = np.linspace(0, 10, 1000)
y = np.cos(x[:, np.newaxis] * np.arange(0, 2, 0.2))
plt.plot(x, y[:, 0], label = 'c1')
plt.plot(x, y[:, 1], label = 'c2')
plt.plot(x, y[:, 2], label = 'c3')
plt.plot(x, y[:, 3:])
plt.legend(framealpha = 1, frameon = True)
plt.show()
print()

# 컬러바 출력
print('컬러바 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar()
plt.show()
print()

# 컬러바 색상 지정해서 출력(파란색)
print('컬러바 색상 지정해서 출력(파란색)')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = 'Blues')
plt.colorbar()
plt.show()
print()

# 컬러바 색상 지정해서 출력(파란색 + 빨간색)
print('컬러바 색상 지정해서 출력(파란색 + 빨간색)')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = 'RdBu')
plt.colorbar()
plt.show()
print()

# 컬러바 범위와 컬러바 생김새 변화해서 출력
print('컬러바 범위와 컬러바 생김새 변화해서 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
plt.imshow(I, cmap = 'RdBu')
plt.colorbar(extend = 'both')
plt.clim(-1, 1)
plt.show()
print()

# 컬러바 범위를 5개로 지정해서 출력
print('컬러바 범위를 5개로 지정해서 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = plt.cm.get_cmap('Blues', 5))
plt.colorbar()
plt.clim(-1, 1)
plt.show()
print()

# 다중 플롯

# 플롯 안에 또 다른 플롯 출력
print('플롯 안에 또 다른 플롯 출력')
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.5, 0.2, 0.3])
plt.show()
print()

# 반복문으로 여러개 플롯 출력
print('반복문으로 여러개 플롯 출력')
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha = 'center')
plt.show()
print()

# 플롯끼리 겹쳐서 출력된것 간격줘서 출력
print('플롯끼리 겹쳐서 출력된것 간격줘서 출력')
fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha = 'center')
plt.show()
print()

# 반복문 쓰지않고 여러개 한번에 출력
print('반복문 쓰지않고 여러개 한번에 출력')
fig, ax = plt.subplots(3, 3, sharex = 'col', sharey = 'row')
plt.show()
print()

# 위 플롯에 위치값 넣어주기
print('위 플롯에 위치값 넣어주기')
fig, ax = plt.subplots(3, 3, sharex = 'col', sharey = 'row')
for i in range(3):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), ha = 'center')
plt.show()
print()

# 위치값을 주어 플롯 크기 다르게 하기
print('위치값을 주어 플롯 크기 다르게 하기')
grid = plt.GridSpec(2, 3, wspace = 0.4, hspace = 0.4)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
plt.show()
print()

# 반복문을 통한 서브플롯
print('반복문을 통한 서브플롯')
plt.figure(figsize = (5, 6))
x = range(1, 21)
columns = [np.random.randn(20) * i for i in range(1, 7)]
i = 0
for c in columns:
    i += 1
    plt.subplot(3, 2, i)
    plt.plot(x, c, marker = 'o', linewidth = 1, label = c)
    plt.xlim(-1, 21)
    plt.ylim(c.min() - 1, c.max() + 1)
plt.show()
print()

# 텍스트와 주석

# 텍스트로 플롯에 위치치정해서 출력1
print('텍스트로 플롯에 위치치정해서 출력1')
fig, ax = plt.subplots()
ax.axis([0, 10, 0, 10])
ax.text(3, 6, '. transData(3, 6)', transform = ax.transData) # 데이터에 실 위치
ax.text(0.2, 0.4, '. transAxes(0.2, 0.4)', transform = ax.transAxes) # 축에의한 위치
ax.text(0.2, 0.2, '. transFigure(0.2, 0.2)', transform = fig.transFigure) # 실체 플롯의 위치
plt.show()
print()

# 텍스트로 플롯에 위치치정해서 출력2
print('텍스트로 플롯에 위치치정해서 출력2')
fig, ax = plt.subplots()
ax.axis([0, 10, 0, 10])
ax.text(3, 6, '. transData(3, 6)', transform = ax.transData) # 데이터에 실 위치
ax.text(0.2, 0.4, '. transAxes(0.2, 0.4)', transform = ax.transAxes) # 축에의한 위치
ax.text(0.2, 0.2, '. transFigure(0.2, 0.2)', transform = fig.transFigure) # 실체 플롯의 위치
ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
plt.show()
print()

# 플롯에 화살표로 가르키는 그림 출력1
print('플롯에 화살표로 가르키는 그림 출력1')
x = np.arange(1, 40)
y = x * 1.1
plt.scatter(x, y, marker = '.')
plt.axis('equal')
plt.annotate('interesting point', xy = (4, 5), xytext = (20, 10),
             arrowprops = dict(shrink = 0.05))
plt.show()
print()

# 플롯에 화살표로 가르키는 그림 출력2
print('플롯에 화살표로 가르키는 그림 출력2')
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 3, 100)
x3 = np.random.normal(10, 3, 100)
plt.plot(x1, label = 'p1')
plt.plot(x2, label = 'p2')
plt.plot(x3, label = 'p3')
plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 0, ncol = 3,
           mode= 'extend', borderaxespad = 0.)
plt.annotate('important value', (50, 20), xytext = (5, 40),
             arrowprops = dict(arrowstyle = '->'))
plt.annotate('incorrect value', (40, 30), xytext = (50, 40),
             arrowprops = dict(arrowstyle = '->'))
plt.show()
print()

# 눈금 맡춤

# x,y 축을 로그스케일로 출력
print('x,y 축을 로그스케일로 출력')
plt.axes(xscale = 'log', yscale = 'log')
plt.show()
print()

# NullLocator, NullFormatter : 눈금없음, 눈금레이블 없음
print('NullLocator, NullFormatter : 눈금없음, 눈금레이블 없음')
ax = plt.axes()
ax.plot(np.random.randn(100).cumsum())
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.show()
print()

# MaxNLocator : 눈금 최대 숫자에 맞게 적절한 위치 찾음
print('MaxNLocator : 눈금 최대 숫자에 맞게 적절한 위치 찾음')
fig, ax = plt.subplots(3, 3, sharex = True, sharey = True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(4))
    axi.yaxis.set_major_locator(plt.MaxNLocator(4))
fig.show()
plt.show()
print()

# 좌표계열 플롯 출력
print('좌표계열 플롯 출력')
x = np.linspace(-np.pi, np.pi, 1000, endpoint = True)
y = np.sin(x)
plt.plot(x, y)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.show()
print()

# linear스케일과 log스케일 차이
print('linear스케일과 log스케일 차이')
x = np.linspace(1, 10)
y = [10 ** el for el in x]
z = [2 * el for el in x]
fig = plt.figure(figsize = (10, 8))
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, '-y')
ax1.set_yscale('log')
ax1.set_title(r'Logarithmic plot of $ {10}^{x} $')
ax1.set_ylabel(r'$ {y} = {10}^{x} $')
plt.grid(b = True, which = 'both', axis = 'both')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y, '--r')
ax2.set_yscale('linear')
ax2.set_title(r'Linear plot of $ {10}^{x} $')
ax2.set_ylabel(r'$ {y} = {10}^{x} $')
plt.grid(b = True, which = 'both', axis = 'both')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, y, '-.g')
ax3.set_yscale('log')
ax3.set_title(r'Logarithmic plot of $ {2}^{x} $')
ax3.set_ylabel(r'$ {y} = {2}^{x} $')
plt.grid(b = True, which = 'both', axis = 'both')

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, y, ':b')
ax4.set_yscale('linear')
ax4.set_title(r'Linear plot of $ {2}^{x} $')
ax4.set_ylabel(r'$ {y} = {2}^{x} $')
plt.grid(b = True, which = 'both', axis = 'both')
plt.show()
print()

# 스타일

# 플롯 스타일 출력
print('플롯 스타일 출력')
fig = plt.figure(figsize = (10, 10))
x = range(1, 11)
columns = [np.random.randn(10) * i for i in range(1, 26)]

for n, v in enumerate(plt.style.available[1:]):
    plt.style.use(v)
    plt.subplot(5, 5, n + 1)
    plt.title(v)
    
    for c in columns:
        plt.plot(x, c, marker = '', color = 'royalblue', linewidth = 1, alpha = 0.4)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
plt.show()
print()

# 플롯 스타일 적용
plt.style.use(['seaborn-notebook'])

# 플롯 종류

# cohere : x와 y의 일관성 시각화 그리기
# contour : 플롯 등고선
# errorbar : 오류 막대 그래프
# hexbin : 육각형 binning 플롯 생성
# hist: 히스토그램 플롯
# imshow : 축에 이미지 표시
# pcolor : 2차원 배열의 유사 플롯 생성

# bar : 막대 플롯 생성
print('bar : 막대 플롯 생성')
height = [np.random.randn() * i for i in range(1, 6)]
names = ['A', 'B', 'C', 'D', 'E']
y_pos = np.arange(len(names))
plt.bar(y_pos, height)
plt.xticks(y_pos, names, fontweight = 'bold')
plt.xlabel('group')
plt.show()
print()

# 막대 플롯 세로로 출력
print('막대 플롯 세로로 출력')
height = [np.random.randn() * i for i in range(1, 6)]
names = ['A', 'B', 'C', 'D', 'E']
y_pos = np.arange(len(names))
plt.barh(y_pos, height)
plt.yticks(y_pos, names, fontweight = 'bold')
plt.ylabel('group')
plt.show()
print()

# 막대그래프 쌓이는 형태
print('막대그래프 쌓이는 형태')
bars1 = [12, 28, 1, 8, 22]
bars2 = [28, 7, 16, 4, 10]
bars3 = [25, 3, 23, 25, 17]
bars = np.add(bars1, bars2).tolist()

r = [0, 1, 2, 3, 4]
names = ['A', 'B', 'C', 'D', 'E']

plt.bar(r, bars1, color = 'royalblue', edgecolor = 'white')
plt.bar(r, bars2, bottom = bars1, color = 'skyblue', edgecolor = 'white')
plt.bar(r, bars3, bottom = bars2, color = 'lightblue', edgecolor = 'white')

plt.xlabel('group', fontweight = 'bold')
plt.xticks(r, names, fontweight = 'bold')
plt.show()
print()


# 막대그래프 펼쳐지는 형태
print('막대그래프 펼쳐지는 형태')
bar_width = 0.25

bars1 = [14, 17, 9, 8, 7]
bars2 = [14, 7, 12, 4, 10]
bars3 = [21, 4, 24, 13, 17]

r = [0, 1, 2, 3, 4]
names = ['A', 'B', 'C', 'D', 'E']

r1 = np.arange(len(bars1))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, bars1, color = 'royalblue', width = bar_width,edgecolor = 'white', label = r1)
plt.bar(r2, bars2, color = 'skyblue', width = bar_width, edgecolor = 'white', label = r2)
plt.bar(r3, bars3, color = 'lightblue', width = bar_width, edgecolor = 'white', label = r3)

plt.xlabel('group', fontweight = 'bold')
plt.xticks([r + bar_width for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
plt.legend()
plt.show()
print()

# barbs : barbs의 2차원 필드 그리기
print('barbs : barbs의 2차원 필드 그리기')
x = [0, 5, 10, 15,30, 40, 50, 60, 100]
v = [0, -5, -10, -15, -30, -40, -50, -60, -100]
n = len(v)
y = np.ones(n)
u = np.zeros(n)

plt.barbs(x, y, u, v, length = 9)
plt.xticks(x)
plt.ylim(0.98, 1.05)
plt.show()
print()

# 스템 플롯

# 줄기모양 플롯출력
print('줄기모양 플롯출력')
x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))

plt.stem(x, y, linefmt = 'gray', bottom = 1, use_line_collection = True)
plt.show()
print()

# 박스 플롯

# boxplot : 상자 및 수염 플롯 생성
print('boxplot : 상자 및 수염 플롯 생성')
r1 = np.random.normal(loc = 0, scale = 0.5, size = 100)
r2 = np.random.normal(loc = 0.5, scale = 1, size = 100)
r3 = np.random.normal(loc = 1, scale = 1.5, size = 100)
r4 = np.random.normal(loc = 1.5, scale = 2, size = 100)
r5 = np.random.normal(loc = 2, scale = 2.5, size = 100)

f, ax = plt.subplots(1, 1)
ax.boxplot((r1, r2, r3, r4, r5))
ax.set_xticklabels(['r1', 'r2', 'r3', 'r4', 'r5'])
plt.show()
# 박스가 안나옴...
print()

# 산점도

# 산점도 출력
print('산점도 출력')
plt.plot(np.random.randn(50), 'o')
plt.show()
print()

# 산점도 종류별로 출력
print('산점도 종류별로 출력')
plt.figure(figsize = (8, 4))
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'D', 'd', '|', '_']
for m in markers:
    plt.plot(np.random.randn(5), np.random.randn(5), m, label = "'{0}'".format(m))
plt.legend(loc = 'center right', ncol = 2)
plt.xlim(0, 1.5)
plt.show()
print()

# plot가 아닌 scatter로 산점도 출력
print('plot가 아닌 scatter로 산점도 출력')
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.scatter(x, y, marker = 'o')
plt.show()
print()

# 반복문을 돌려서 산점도 출력
print('반복문을 돌려서 산점도 출력')
for i in range(9):
    x = np.arange(1000)
    y = np.random.randn(1000).cumsum()
    plt.scatter(x, y, alpha = 0.2, cmap = 'viridis')
plt.show()
print()

# 버블차트로 출력
print('버블차트로 출력')
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c = colors, s = sizes, alpha = 0.3, cmap = 'viridis')
plt.colorbar()
plt.show()
print('cmap : color map : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html')
print()

# 버블차트 색상 변경
print('버블차트 색상 변경')
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c = colors, s = sizes, alpha = 0.3, cmap = 'magma')
plt.colorbar()
plt.show()
print()

# x와 y의 일관성 차트

# 2개의 일관성 출력
print('2개의 일관성 출력')
dt = 0.01
t = np.arange(0, 30, dt)
n1 = np.random.randn(len(t))
n2 = np.random.randn(len(t))

s1 = 1.5 * np.sin(2 * np.pi * 10 * t) + n1
s2 = np.cos(np.pi * t) + n2
plt.cohere(s1, s2 ** 2, 128, 1./dt)
plt.xlabel('time')
plt.ylabel('coherence')
plt.show()
print()


# =============================================================================


