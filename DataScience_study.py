# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:51:18 2020

@author: msi
"""
# 데이터 사이언스는 데이터와 연관된 모든 것을 의미

# 데이터 사이언티스트가 하는일
# 가치를 더할 수 있는 일을 찾고 데이터를 이용해서 문제를 해결하는 것

# 데이터사이언스 단계
# 1. 문제 정의
# 2. 데이터 모으기
# 3. 데이터 다듬기
# 4. 데이터 분석하기
# 5. 데이터 시각화 및 커뮤니케이션

# 데이터 모으기
# - 웹크롤링
# - 자료 모으기
# - 파일 읽고 쓰기

# 데이터 분석하기
# - 데이터 파악
# - 데이터 변형
# - 통계 분석
# - 인사이트 발견
# - 의미 도출


# 실습과제

# 팔린드롬 문제 설명
# "토마토"나 "기러기"처럼 거꾸로 읽어도 똑같은 단어를 팔린드롬(palindrome)이라고 부릅니다. 
# 문자열 word가 팔린드롬인지 확인하는 함수 is_palindrome를 쓰세요. 
# is_palindrome은 word가 팔린드롬이면 True를, 팔린드롬이 아니면 False를 리턴합니다.

i = 'abcba'
print(i[::-1])

def is_palindrome(word):
    if word == word[::-1]:
        return True
    else:
        return False

# 테스트
print(is_palindrome("racecar"))
print(is_palindrome("stars"))
print(is_palindrome("토마토"))
print(is_palindrome("kayak"))
print(is_palindrome("hello"))

import numpy as np

# 1차원 배열
print('1차원 배열')
array1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
print(array1)
print(type(array1))
print(array1.shape)
print()

# 인덱싱, 슬라이싱 색인
print('인덱싱, 슬라이싱 색인')
print(array1[0])
print(array1[2])
print(array1[-1])
print(array1[-2])
print(array1[[1, 3, 4]])
print(array1[2:7])
print(array1[:7])
print(array1[2:])
print(array1[2:11:2])
print()

# 2차원 배열
print('2차원 배열')
array2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(array2)
print(type(array2))
print(array2.shape)
# 세로길이가 3, 가로길이가 4

# 1부터 100까지 에서 3의배수만 출력
print('1부터 100까지 에서 3의배수만 출력')
arr = np.arange(1, 101)
print(arr[arr % 3 == 0])

# numpy 기본 연산
print('numpy 기본 연산')
array1 = np.arange(10)
array2 = np.arange(10, 20)
print(array1)
print(array1 * 2)
print(array1 / 2)
print(array1 + 1)
print(array1 ** 2)
print()

print(array1)
print(array2)
print(array1 + array2)
print(array1 * array2)
print(array1 / array2)
# 같은 자리끼리 연산한다.
print()

# numpy 불린 연산
print('numpy 불린 연산')
array1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
print(array1)
print(array1 > 4)
print(array1 % 2 == 0)
print()

booleans = np.array([True, True, False, True, True, False, True, True, True, False, True])
print(np.where(booleans))
# True인 인덱스를 출력해준다
Filter = np.where(array1 > 4)
print(Filter)
print(array1[Filter])
print()

import pandas as pd

# DataFrame 사용해 보기
print('DataFrame 사용해 보기')
two_dimensional_list = [['dongwook', 50, 86], ['sineui', 89, 31], ['ikjoong', 68, 91], ['yoonsoo', 88, 75]]
arr = np.array(two_dimensional_list)
df = pd.DataFrame(two_dimensional_list,
                  columns = ['name', 'english_score', 'math_score'],
                  index = ['a', 'b', 'c', 'd'])
print(arr)
print(df)
print(type(df))
print(df.columns)
print(df.index)
print(df.dtypes)
print()

# pandas로 데이터 읽어들이기
print('pandas로 데이터 읽어들이기')
iphone_df = pd.read_csv('iphone.csv')
print(iphone_df)
print()

# 첫번째 컬럼 index로 만들기
print('첫번째 컬럼 index로 만들기')
iphone_df2 = pd.read_csv('iphone.csv', index_col = 0)
print(iphone_df2)
print()

# 가장 인기있는 아기 이름 csv파일 불러오기
print('가장 인기있는 아기 이름 csv파일 불러오기')
df = pd.read_csv('popular_baby_names.csv')
print(df)
print()

# 메가밀리언 로또 당첨번호 csv파일 날짜컬럼을 인덱스로 만들기
print('메가밀리언 로또 당첨번호 csv파일 날짜컬럼을 인덱스로 만들기')
df2 = pd.read_csv('mega_millions.csv', index_col = 0)
print(df2)
print()

# DataFrmae 인덱싱1
print('DataFrmae 인덱싱1')
iphone_df2 = pd.read_csv('iphone.csv', index_col = 0)
print(iphone_df2.loc['iPhone 8', '메모리']) # iPhone 8의 메모리 출력
print(iphone_df2.loc['iPhone X', :]) # iPhone X 전체 출력
print(iphone_df2.loc[:, '출시일']) # 모든 iPhone의 출시일
print()

# 2016년 KBS 방송사 시청률 출력
print('2016년 KBS 방송사 시청률 출력')
broadcast_df = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df)
print(broadcast_df.loc[2016, 'KBS'])
# 년도 자료형이 정수
print()

# JTBC 시청률 출력
print('JTBC 시청률 출력')
broadcast_df2 = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df2)
print(broadcast_df2.loc[:,'JTBC'])
print()

# SBS와 JTBC의 시청률만 출력
print('SBS와 JTBC의 시청률만 출력')
broadcast_df3 = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df3)
print(broadcast_df3.loc[:,['SBS', 'JTBC']])
print()

# 삼송.csv와 현디.csv 카테고리중 요일별 문화 생활비를 출력
print('삼송.csv와 현디.csv 카테고리중 요일별 문화 생활비를 출력')
samsong_df = pd.read_csv('samsong.csv')
hyundee_df = pd.read_csv('hyundee.csv')
print(samsong_df)
print(hyundee_df)
sh_df = pd.DataFrame({'day': hyundee_df['요일'],
                      'samsong' : samsong_df['문화생활비'],
                      'hyundee' : hyundee_df['문화생활비']})
print(sh_df)

# DataFrame 인덱싱2
print('DataFrame 인덱싱2')
iphone_df3 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df3)
print(iphone_df3.loc[['iPhone 8', 'iPhone X'], :]) # 아이폰8과 아이폰X 데이터 출력
print(iphone_df3[['Face ID', '출시일', '메모리']]) # Face ID, 출시일, 메모리 컬럼 출력
print(iphone_df3.loc['iPhone 8':'iPhone XS', :]) # 아이폰8부터 아이폰XS까지 출력
print(iphone_df3.loc[:, '메모리':'Face ID']) # 메모리부터 Face ID컬럼 까지 출력
print()

# 방송사는 KBS에서 SBS까지 연도는 2012년부터 2017년까지의 시청률 출력
print('방송사는 KBS에서 SBS까지 연도는 2012년부터 2017년까지의 시청률 출력')
broadcast_df4 = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df4)
print(broadcast_df4.loc[2012:2017, 'KBS':'SBS'])
print()

# DataFrame 조건으로 인덱싱
print('DataFrame 조건으로 인덱싱')
iphone_df4 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df4)
print(iphone_df4.loc[[True, False, True, True, False, True, False]]) # True인 인덱스들만 출력
# iphone_df4.loc[[True, False, False, True]]
print(iphone_df4.loc[[True, False, False, True, False, False, False]])
# 강의영상에는 4개만 적으면 나머지들은 False라고 하였으나 현재버전에서는 전부 적어줘야 한다
print(iphone_df4['디스플레이'] > 5)
print(iphone_df4.loc[iphone_df4['디스플레이'] > 5]) # 디스플레이가 5보다 큰것들만 출력
print(iphone_df4['Face ID'] == 'Yes')
print(iphone_df4[iphone_df4['Face ID'] == 'Yes']) # Face ID가 Yes인 것들만 출력
print((iphone_df4['디스플레이'] > 5) & (iphone_df4['Face ID'] == 'Yes'))
condition = (iphone_df4['디스플레이'] > 5) & (iphone_df4['Face ID'] == 'Yes')
print(iphone_df4.loc[condition]) # 디스플레이가 5보다 크면서 Face ID가 Yes인 것들 만 출력
print()

# KBS에서 시청률이 30이 넘는 데이터만 출력
print('KBS에서 시청률이 30이 넘는 데이터만 출력')
broadcast_df5 = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df5)
print(broadcast_df5.loc[broadcast_df5['KBS'] > 30, 'KBS'])
# print(broadcast_df5.loc[broadcast_df5['KBS'] > 30]['KBS'])
print()

# SBS가 TV CHOSUN보다 더 시청률이 낮았던 시기 출력
print('SBS가 TV CHOSUN보다 더 시청률이 낮았던 시기 출력')
broadcast_df6 = pd.read_csv('broadcast.csv', index_col = 0)
print(broadcast_df6['SBS'])
print(broadcast_df6['TV CHOSUN'])
print(broadcast_df6['SBS'] < broadcast_df6['TV CHOSUN'])
print(broadcast_df6.loc[broadcast_df6['SBS'] < broadcast_df6['TV CHOSUN'], ['SBS', 'TV CHOSUN']])
print()

# DataFrame 위치로 인덱싱하기
print('DataFrame 위치로 인덱싱하기')
iphone_df5 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df5)
print(iphone_df5.iloc[2, 4])
print(iphone_df5.iloc[[1, 3], [1, 4]])
print(iphone_df5.iloc[3:, 1:4])
print()

