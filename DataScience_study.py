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

print(array1)
print(array2)
print(array1 * 2)
print(array1 ** 2)
print(array1 + array2)
print(array1 * array2)
print()

# numpy 불린 연산
print('numpy 불린 연산')
array1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
print(array1)
print(array1 > 4)
print(array1 % 2 == 0)
print()

booleans = np.array([True, True, False, True, True, False, True, True, True, False, True])
print(booleans)
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
print()
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
print()
print(iphone_df2.loc['iPhone X', :]) # iPhone X 전체 출력
print()
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
print(iphone_df4.loc[[True, False, True, True, False, True, False],:]) # True인 인덱스들만 출력
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


# DataFrame에 값 쓰기1
print('DataFrame에 값 쓰기1')
iphone_df6 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df6)
iphone_df6.loc['iPhone 8', '메모리'] = '2.5GB' # 메모리 2GB -> 2.5GB로 변경
print(iphone_df6)
iphone_df6.loc['iPhone 8', '출시 버전'] = 'iOS 10.3' # 출시버전 iOS 11.0 -> iOS 10.3으로 변경
print(iphone_df6)
iphone_df6.loc['iPhone 8',:] = ['2016-09-22', '4.7', '2GB', 'iOS 11.0', 'No']
# iPhone 8 행 데이터 전부 변경
print(iphone_df6)
iphone_df6['디스플레이'] = ['4.7 in', '5.5 in', '4.7 in', '5.5 in', '5.8 in', '5.8 in', '6.5 in']
# 디스플레이 컬럼 데이터 전부 변경
print(iphone_df6['디스플레이'])
iphone_df6['Face ID'] = 'Yes'
# 하나의 값으로 전부 변경할 때
print(iphone_df6)
print()


# DataFrame에 값 쓰기2
print('DataFrame에 값 쓰기2')
iphone_df7 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df7)
iphone_df7[['디스플레이', 'Face ID']] = 'x'
# 디스플레이, Face ID 컬럼 전체에 'x'로 수정
print(iphone_df7)
iphone_df7.loc[['iPhone 7', 'iPhone X']] = 'o'
# iPhone 7, iPhone X 행 전체에 'o'로 수정
print(iphone_df7)
iphone_df7.loc['iPhone 7':'iPhone X'] = 'o'
# iPhone 7 ~ iPhone x 행 까지 전체 'o'로 수정
print(iphone_df7)
iphone_df7 = pd.read_csv('iPhone.csv', index_col = 0)
iphone_df7[iphone_df7['디스플레이'] > 5] = 'p'
# 디스플레이가 5보다 큰 모든데이터 'p'로 수정
print(iphone_df7)
iphone_df7.iloc[[1, 3], [1, 4]] = 'v'
# iPhone 7 Plus, iPhone 8 Plus행의 디스플레이, Face ID 컬럼 데이터 'v'로 수정
print(iphone_df7)
print()

# DataFrame에 값 추가/삭제
print('DataFrame에 값 추가/삭제')
iphone_df8 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df8)
iphone_df8.loc['iPhone XR'] = ['2018-10-26', 6.1, '3GB', 'iOS 12.0.1', 'Yes']
# iPhone XR행과 데이터 추가
print(iphone_df8)
iphone_df8['제조사'] = 'Apple'
# 제조사 컬럼과 데이터 추가
print(iphone_df8)
iphone_df8.drop('iPhone XR', axis = 'index', inplace = False)
# iPhone XR행 제거
print(iphone_df8)
# 하지만 확인해보면 제거안됨 위에서 inplace = False(기본값) 때문에 
iphone_df8.drop('iPhone XR', axis = 'index', inplace = True)
print(iphone_df8)
# 제거된것 확인
iphone_df8.drop('제조사', axis = 'columns', inplace = True)
# 제조사 컬럼 제거 
print(iphone_df8)
iphone_df8.drop(['iPhone 7', 'iPhone 8', 'iPhone X'], axis = 'index', inplace = True)
# iPhone 7, iPhone 8, iPhone X 데이터 삭제
print(iphone_df8)
print()


# 키와 몸무게가 담겨있는 DataFrame이 있다. 아래 3가지만 수정
print('키와 몸무게가 담겨있는 DataFrame이 있다. 아래 3가지만 수정')
body_imperial_df = pd.read_csv('body_imperial1.csv', index_col = 0)

# 1. ID 1의 무게를 200으로 변경
print('1. ID 1의 무게를 200으로 변경')
body_imperial_df.loc[1,'Weight (Pound)'] = 200
print(body_imperial_df)
print()

# 2. ID 21의 row를 삭제
print('2. ID 21의 row를 삭제')
body_imperial_df.drop(21, axis = 'index', inplace = True)
print(body_imperial_df)
print()

# 3. ID 20의 row를 추가, 키: 70, 무게: 200
print('3. ID 20의 row를 추가, 키: 70, 무게: 200')
body_imperial_df.loc[20,:] = [70, 200]
print(body_imperial_df)
print()


# 키와 몸무게가 담겨있는 DataFrame이 있다. 아래 2가지만 수정
print('키와 몸무게가 담겨있는 DataFrame이 있다. 아래 2가지만 수정')
body_imperial_df2 = pd.read_csv('body_imperial2.csv', index_col = 0)

# 1. '비만도' column을 추가하고 모든 ID에 정상으로 설정
print("1. '비만도' column을 추가하고 모든 ID에 정상으로 설정")
body_imperial_df2['비만도'] = '정상'
print(body_imperial_df2)
print()

# 2. 'Gender' column의 값을 0~10까지는 'Male' 11~20까지는 'Female'로 변경
print("2. 'Gender' column의 값을 0~10까지는 'Male' 11~20까지는 'Female'로 변경")
body_imperial_df2.loc[:10, 'Gender'] = 'Male'
body_imperial_df2.loc[11:, 'Gender'] = 'Female'
print(body_imperial_df2)
print()


# index/column 설정하기
print('index/column 설정하기')
liverpool_df = pd.read_csv('liverpool.csv', index_col = 0)
print(liverpool_df)
liverpool_df.rename(columns = {'position' : 'Position'}, inplace = True)
# position컬럼 Position으로 앞글자만 대문자로 변경
print(liverpool_df)
liverpool_df.rename(columns = {'position' : 'Position',
                               'born' : 'Born',
                               'number' : 'Number',
                               'nationality' : 'Nationality'}, inplace = True)
# 컬럼 앞글자 전부 대문자로 변경
print(liverpool_df)
liverpool_df.index.name = 'Player Name'
# 인덱스의 이름 설정
print(liverpool_df)
print(liverpool_df.set_index('Number'))
# 등번호로 index를 바꿨지만 기존에있던 이름 사라짐 따로 빼줘야 한다.
liverpool_df['Player Name'] = liverpool_df.index
print(liverpool_df)
print(liverpool_df.set_index('Number', inplace = True))
# 인덱스를 Number로 변경
print(liverpool_df)
print()


# 토익 각 파트 최소 250점, 총점수 600점이 되어야 서류전형을 합격할수 있다.
# 합격 여부 컬럼 생성후 합격은 True, 불합격은 False를 넣어라
print('''토익 각 파트 최소 250점, 총점수 600점이 되어야 서류전형을 합격할수 있다.
합격 여부 컬럼 생성후 합격은 True, 불합격은 False를 넣어라''')
toeic_df = pd.read_csv('toeic.csv')
# toeic_df['합격 여부'] = False
toeic_df
# toeic_df.loc[(toeic_df['LC'] >= 250) & (toeic_df['RC'] >= 250) & (toeic_df['LC'] + toeic_df['RC'] >= 600), '합격 여부'] = True
toeic_df['합격 여부'] = (toeic_df['LC'] >= 250) & (toeic_df['RC'] >= 250) & (toeic_df['LC'] + toeic_df['RC'] >= 600)
print(toeic_df)


# 퍼즐
print('퍼즐')
puzzle_df = pd.read_csv('https://github.com/codeit-courses/data-science/raw/master/Puzzle_before.csv')
puzzle_df['A'] *=  2
puzzle_df[puzzle_df.loc[:, 'B':'E'] < 80] = 0
puzzle_df[puzzle_df.loc[:, 'B':'E'] >= 80] = 1
puzzle_df.loc[2, 'F'] = 99
print(puzzle_df)
print()


# 큰 DataFrame 살펴보기
print('큰 DataFrame 살펴보기')
laptops_df = pd.read_csv('laptops.csv')
print(laptops_df)
print(laptops_df.head(3))
print(laptops_df.head(7))
print(laptops_df.tail(6))
print(laptops_df.shape)
# 167개 로우와 15개 컬럼
print(laptops_df.columns)
print(laptops_df.info())
print(laptops_df.describe())
# descride() : 데이터 프레임의 통계정보
print(laptops_df.sort_values(by = 'price')) # 옵션을 아무거도 안쓰면 오름차순
print(laptops_df.sort_values(by = 'price', ascending = False)) # 내림차순
print()


# 큰 Series 살펴보기
print('큰 Series 살펴보기')
laptops_df2 = pd.read_csv('laptops.csv')
print(laptops_df2['brand']) # 브랜드 출력
print(laptops_df2['brand'].unique()) # 중복된 브랜드 제거
print(laptops_df2['brand'].value_counts()) # 각 브랜드별로 몇개씩 들어있는지 출력
print(laptops_df2['brand'].describe()) # 브랜드 컬럼 통계정보
print()


# 여행지 선정하기1
print('여행지 선정하기1')
world_cities_df = pd.read_csv('world_cities.csv', index_col = 0)

# 1. 주어진 데이터에는 총 몇 개의 도시와 몇 개의 나라가 있는지
print('1. 주어진 데이터에는 총 몇 개의 도시와 몇 개의 나라가 있는지')
print(world_cities_df)
print(world_cities_df['City / Urban area'].describe())
# world_cities_df['City / Urban area'].value_counts().shape
print(world_cities_df['Country'].describe())
# world_cities_df['Country'].value_counts().shape
print()

# 2. 주어진 데이터에서, 인구 밀도(명/sqKm) 가 10000 이 넘는 도시는 총 몇 개
print('2. 주어진 데이터에서, 인구 밀도(명/sqKm) 가 10000 이 넘는 도시는 총 몇 개')
print(world_cities_df)
wc_count_df = world_cities_df[world_cities_df['Population'] / world_cities_df['Land area (in sqKm)'] > 10000]
print(wc_count_df.shape)
# wc_count_df.infro()
print()

# 3. 인구 밀도가 가장 높은 도시를 찾아봅시다.
print('3. 인구 밀도가 가장 높은 도시를 찾아봅시다.')
world_cities_df['인구밀도'] = world_cities_df['Population'] / world_cities_df['Land area (in sqKm)']
print(world_cities_df.sort_values(by = '인구밀도', ascending = False))
print()


# 여행지 선정하기2
print('여행지 선정하기2')
world_cities_df2 = pd.read_csv('world_cities2.csv', index_col = 0)

# 1. 나라 이름이 기억나지 않고, 데이터에 4개의 도시가 나왔다는 것만 기억. 이 나라는 무엇인가?
print('1. 나라 이름이 기억나지 않고, 데이터에 4개의 도시가 나왔다는 것만 기억. 이 나라는 무엇인가?')
print(world_cities_df2['Country'].value_counts()[world_cities_df2['Country'].value_counts() == 4])
# country = world_cities_df2['Country'].value_counts()
# country[country == 4]
print()


# 수강신청 준비하기 (다른방법으로 다시해보기)
print('수강신청 준비하기')
# 수강신청에는 다음 3개의 조건이 있다.
# 1. “information technology” 과목은 심화과목이라 1학년은 수강할 수 없습니다.
# 2. “commerce” 과목은 기초과목이고 많은 학생들이 듣는 수업이라 4학년은 수강할 수 없습니다.
# 3. 수강생이 5명이 되지 않으면 강의는 폐강되어 수강할 수 없습니다.

# “status”라는 이름의 column을 추가하고, 
# 학생이 수강 가능한 상태이면 “allowed”, 
# 수강 불가능한 상태이면 “not allowed”를 넣어주세요.
enrolment_df = pd.read_csv('enrolment_1.csv')
enrolment_df

cnt_1 = (enrolment_df['year'] == 1) & (enrolment_df['course name'] == 'information technology')
# True면 수강신청 못함

cnt_4 = (enrolment_df['year'] == 4) & (enrolment_df['course name'] == 'commerce')
# True면 수강신청 못함

std_cnt = enrolment_df['course name'].value_counts()
std_idx = std_cnt[std_cnt < 5].index
std_list = list(std_idx)
std_list

enrolment_df['course name'].values

cnt_5 = []
for i in enrolment_df['course name'].values:
    if i in std_list:
        cnt_5.append(True)
    else:
        cnt_5.append(False)

cnt_5
# True면 수강신청 못함

enrolment_df['status'] = np.where(cnt_5, 'not allowed', np.where(cnt_1 | cnt_4, 'not allowed', 'allowed'))
print(enrolment_df)
print()


# 강의실 배정하기1
print('강의실 배정하기1')

# 강의실은 규모에 따라 “Auditorium”, “Large room”, “Medium room”, “Small room” 총 4가지 종류가 있다.
# 1. 80명 이상의 학생이 수강하는 과목은 “Auditorium”에서 진행
# 2. 40명 이상, 80명 미만의 학생이 수강하는 과목은 “Large room”에서 진행
# 3. 15명 이상, 40명 미만의 학생이 수강하는 과목은 “Medium room”에서 진행
# 4. 5명 이상, 15명 미만의 학생이 수강하는 과목은 “Small room”에서 진행
# 5. 폐강 등의 이유로 status가 “not allowed”인 수강생은 room assignment 또한 “not assigned”가 되어야 한다.
enrolment_df2 = pd.read_csv('enrolment_2.csv')
enrolment_df2
enrolment_df2['room assignment'] = 'not assigned'

course_cnt = enrolment_df2['course name'].value_counts()

list_80 = list(course_cnt[course_cnt >= 80].index)
list_40 = list(course_cnt[(course_cnt >= 40) & (course_cnt < 80)].index)
list_15 = list(course_cnt[(course_cnt >= 15) & (course_cnt < 40)].index)
list_5 = list(course_cnt[(course_cnt >= 5) & (course_cnt < 15)].index)

cnt = 0
for i in enrolment_df2['course name']:
    if i in list_80:
        enrolment_df2['room assignment'][cnt] = 'Auditorium'
    elif i in list_40:
        enrolment_df2['room assignment'][cnt] = 'Large room'
    elif i in list_15:
        enrolment_df2['room assignment'][cnt] = 'Medium room'
    elif i in list_5:
        enrolment_df2['room assignment'][cnt] = 'Small room'
    else:
        enrolment_df2['room assignment'][cnt] = 'not assigned'
    cnt += 1
    
enrolment_df2.loc[enrolment_df2['status'] == 'not allowed', 'room assignment'] = 'not assigned'
print(enrolment_df2)

# 강의실 배정하기2
print('강의실 배정하기2')
enrolment_df3 = pd.read_csv('enrolment_3.csv')

# 아래 세 가지 조건을 만족하도록 코드를 작성
# 1. 같은 크기의 강의실이 필요한 과목에 대해 알파벳 순서대로 방 번호를 배정하세요.
#    예를 들어 Auditorium이 필요한 과목으로 “arts”, “commerce”, “science” 세 과목이 있다면, 
#    “arts”는 “Auditorium-1”, “commerce”는 “Auditorium-2”, “science”는 “Auditorium-3” 
#    순서로 방 배정이 되어야 한다.
# 2. “status” column이 “not allowed”인 수강생은 “room assignment” column을 그대로 “not assigned”로 남겨둡니다.
# 3. “room assignment” column의 이름을 “room number”로 바꿔주세요.

enrolment_df3.rename(columns = {'room assignment' : 'room number'}, inplace = True)
enrolment_df3['room number'].unique()

course_list = list(enrolment_df3['course name'].value_counts().index)

Auditorium = enrolment_df3.loc[enrolment_df3['room number'] == 'Auditorium', 'course name'].value_counts()
Large = enrolment_df3.loc[enrolment_df3['room number'] == 'Large room', 'course name'].value_counts()
Medium = enrolment_df3.loc[enrolment_df3['room number'] == 'Medium room', 'course name'].value_counts()
Small = enrolment_df3.loc[enrolment_df3['room number'] == 'Small room', 'course name'].value_counts()

list_1 = list(Auditorium.index.sort_values())
list_2 = list(Large.index.sort_values())
list_3 = list(Medium.index.sort_values())
list_4 = list(Small.index.sort_values())

# enrolment_df3.loc[enrolment_df3['course name'] == 'arts', 'room number']
# f'Auditorium-{num}'

num = 1
for i in list_1:
    enrolment_df3.loc[enrolment_df3['course name'] == i, 'room number'] = f'Auditorium-{num}'
    num += 1

num = 1
for i in list_2:
    enrolment_df3.loc[enrolment_df3['course name'] == i, 'room number'] = f'Large-{num}'
    num += 1

num = 1
list_3
for i in list_3:
    enrolment_df3.loc[enrolment_df3['course name'] == i, 'room number'] = f'Medium-{num}'
    num += 1

num = 1
for i in list_4:
    enrolment_df3.loc[enrolment_df3['course name'] == i, 'room number'] = f'Small-{num}'
    num += 1

enrolment_df3.loc[enrolment_df3['status'] == 'not allowed', 'room number'] = 'not assigned'
print(enrolment_df3)
print()






# 데이터 분석과 시각화
# =============================================================================
# 시각화의 두 가지 목적

# 분석에 도움이 된다.
# 리포팅에 도움이 된다.

# 선 그래프
print('선 그래프')
# import matplotlib as mlt

broadcast_df7 = pd.read_csv('broadcast.csv', index_col = 0)
broadcast_df7.plot(kind = 'line') 
# 아무것도 입력하지 않으면 제일 기본값으로 선그래프(line) 출력
broadcast_df7.plot(y = 'KBS')
broadcast_df7.plot(y = ['KBS', 'JTBC'])
broadcast_df7
print()

# 한국, 미국, 영국, 독일, 중국, 일본의 GDP그래프를 출력
print('한국, 미국, 영국, 독일, 중국, 일본의 GDP그래프를 출력')
gdp_df = pd.read_csv('gdp.csv', index_col = 0)

from_nm = pd.Series(gdp_df.columns)
from_nm[from_nm.str.startswith('U')]

gdp_df[['Korea_Rep', 'United_States', 'United_Kingdom', 'Germany', 'China', 'Japan']].plot()
gdp_df.plot(y = ['Korea_Rep', 'United_States', 'United_Kingdom', 'Germany', 'China', 'Japan'])
# 순서대로 한국, 미국. 영국. 독일, 중국, 일본이다
print()

# 막대 그래프
print('막대 그래프')
sports_df = pd.read_csv('sports.csv', index_col = 0)
sports_df
sports_df.plot() # 기본은 선그래프
sports_df.plot(kind = 'bar') # 막대 그래프 
sports_df.plot(kind = 'barh') # 가로 막대 그래프
sports_df.plot(kind = 'bar', stacked = True) # 쌓여지는 막대그래프
sports_df['Female'].plot(kind = 'bar')
print()

# 실리콘 밸리에서 일하는 남자 관리자 (Managers)에 대한 인종 분포를 막대 그래프 출력
print('실리콘 밸리에서 일하는 남자 관리자 (Managers)에 대한 인종 분포를 막대 그래프 출력')
silicon_valley_summary_df = pd.read_csv('silicon_valley_summary.csv')
managers = silicon_valley_summary_df[silicon_valley_summary_df['job_category'] == 'Managers']
managers_male = managers[managers['gender'] == 'Male']
managers_male.iloc[:4, :].plot(x = 'race_ethnicity', y = 'count', kind = 'bar')
print()

# 파이 그래프
print('파이 그래프')
# 절대적인 수치보다 비율을 표기
broadcast_df8 = pd.read_csv('broadcast.csv', index_col = 0)
broadcast_df8.loc[2017].plot(kind = 'pie')
print()

# 어도비 전체 직원들의 직군 분포를 파이 그래프로 출력
print('어도비 전체 직원들의 직군 분포를 파이 그래프로 출력')
silicon_valley_details_df = pd.read_csv('silicon_valley_details.csv')
silicon_valley_details_df

adobe = silicon_valley_details_df[(silicon_valley_details_df['company'] == 'Adobe') & (silicon_valley_details_df['race'] == 'Overall_totals')]
adobe_jobs = adobe.loc[:, ['job_category', 'count']]
adobe_jobs.set_index('job_category', inplace = True)
adobe_jobs.drop(['Totals', 'Previous_totals'], inplace = True)
adobe_jobs = adobe_jobs[adobe_jobs['count'] != 0]
adobe_jobs.plot(kind = 'pie', y = 'count')
print()

# 히스토그램
print('히스토그램')
body_df = pd.read_csv('body.csv', index_col = 0)
body_df.plot(kind = 'hist', y = 'Height')
body_df.plot(kind = 'hist', y = 'Height', bins = 15)
# 범위 세분화
body_df.plot(kind = 'hist', y = 'Height', bins = 200)
# 너무 많음
print()

# 스타벅스 음료 칼로리를 히스토그램으로 총 20개 구간으로 나눠서 출력
print('스타벅스 음료 칼로리를 히스토그램으로 총 20개 구간으로 나눠서 출력')
starbucks_drinks_df = pd.read_csv('starbucks_drinks.csv')

starbucks_drinks_df['Calories']
starbucks_drinks_df.plot(kind = 'hist', y = 'Calories', bins = 20)
print()

# 박스 플롯
print('박스 플롯')
# 위에서부터 순서대로 
# 최댓값
# 75% 지점(Q3)
# 중간값 : 50% 지점(Q2)
# 25% 지점(Q1)
# 최솟값
# 바깥에 있는 점들은 이상점이라고 한다.

exam_df = pd.read_csv('exam.csv')
exam_df['math score'].describe() # 통계수치들이 요약되서 나온다.
exam_df.plot(kind = 'box', y = 'math score')
exam_df.plot(kind = 'box', y = ['math score', 'reading score', 'writing score'])
print()

# 스타벅스 음료 칼로리를 박스 플롯으로 출력
print('스타벅스 음료 칼로리를 박스 플롯으로 출력')
starbucks_drinks_df2 = pd.read_csv('starbucks_drinks.csv')
starbucks_drinks_df2

starbucks_drinks_df2.plot(kind = 'box', y = 'Calories')
print()

# 산점도
print('산점도')
exam_df2 = pd.read_csv('exam.csv')
exam_df2
exam_df2.plot(kind = 'scatter', x = 'math score', y = 'reading score')
# 수학과 읽기 점수의 산점도
exam_df2.plot(kind = 'scatter', x = 'math score', y = 'writing score')
# 수학과 쓰기 점수의 산점도
exam_df2.plot(kind = 'scatter', x = 'reading score', y = 'writing score')
# 읽기와 쓰기 점수의 산점도
print()

# 국가 지표 분석하기
print('국가 지표 분석하기')
world_indexes_df = pd.read_csv('world_indexes.csv', index_col = 0)
# 다음 중 가장 연관성이 깊은 지표는?
# 1. 기대 수명 - 인터넷 사용자 비율
# 2. 숲 면적 비율 - 탄소 배출 증가율
# 3. 인터넷 사용자 비율 - 숲 면적 비율
# 4. 기대 수명 - 탄소 배출 증가율
# 5. 기대 수명 - 숲 면적 비율

# 1.
world_indexes_df.plot(kind = 'scatter', x = 'Life expectancy at birth- years', y = 'Internet users percentage of population 2014')
# 2.
world_indexes_df.plot(kind = 'scatter', x = 'Forest area percentage of total land area 2012', y = 'Carbon dioxide emissionsAverage annual growth')
# 3.
world_indexes_df.plot(kind = 'scatter', x = 'Internet users percentage of population 2014', y = 'Forest area percentage of total land area 2012')
# 4.
world_indexes_df.plot(kind = 'scatter', x = 'Life expectancy at birth- years', y = 'Carbon dioxide emissionsAverage annual growth')
# 5.
world_indexes_df.plot(kind = 'scatter', x = 'Life expectancy at birth- years', y = 'Forest area percentage of total land area 2012')
print()

# 확률밀도 함수(PDF)
print('확률밀도 함수(PDF)')
# Seaborn(Statistical Data Visualization)
# 통계를 기반으로 한 데이터 시각화
import seaborn as sns

# 확률 밀도 함수는 데이터셋의 분포를 나타낸다.
# 특정 구간의 확률은 그래프 아래 그 구간의 면적과 동일하다.
# 그래프 아래의 모든 면적을 더하면 1(100%)이다.
print()

# KDE Plot
print('KDE Plot')
body_df2 = pd.read_csv('body.csv', index_col = 0)
body_df2
body_df2['Height'].value_counts().sort_index().plot()
sns.kdeplot(body_df2['Height'])
sns.kdeplot(body_df2['Height'], bw = 0.5)
sns.kdeplot(body_df2['Height'], bw = 2)
# bw를 적당하게 값을 줘야한다.
print()

# 승차인원에 대한 KDE Plot를 그려라
print('승차인원에 대한 KDE Plot를 그려라')
subway_df = pd.read_csv('subway.csv')
sns.kdeplot(subway_df['in'])
print()

# KDE활용 예시
print('KDE활용 예시')
body_df3 = pd.read_csv('body.csv', index_col = 0)
body_df3
body_df3.plot(kind = 'hist', y = 'Height')
body_df3.plot(kind = 'hist', y = 'Height', bins = 15)
# 기본 pandas에서 제공하는 히스토그램
sns.distplot(body_df3['Height'], bins = 15)
# seaborn에서 제공하는 히스토그램
body_df3.plot(kind = 'box', y = 'Height')
# 기본 pandas에서 제공하는 박스플롯
sns.violinplot(y = body_df3['Height'])
# 분포 전체를 보여준다는 장점이 있다.
body_df3.plot(kind = 'scatter', x = 'Height', y = 'Weight')
# 기본 pandas에서 제공하는 산점도
sns.kdeplot(body_df3['Height'], body_df3['Weight'])
# seaborn의 등고선 그래프
print()

# 교수의 급여에 대한  Violin Plot을 출력
print('교수의 급여에 대한  Violin Plot을 출력')
salaries_df = pd.read_csv('salaries.csv')
salaries_df
sns.violinplot(x = salaries_df['salary'])
print()

# LM Plot
print('LM Plot')
body_df4 = pd.read_csv('body.csv', index_col = 0)
body_df4
sns.lmplot(data = body_df4, x = 'Height', y = 'Weight')
# 키와 몸무게가 연관서이 많이 없기때문에 선과 점이 많이 떨어져있다.
print()

# 카테고리별 시각화
print('카테고리별 시각화')
laptops_df2 = pd.read_csv('laptops.csv')
laptops_df2
laptops_df2['os'].unique()
sns.catplot(data = laptops_df2, x = 'os', y = 'price', kind = 'box')
sns.catplot(data = laptops_df2, x = 'os', y = 'price', kind = 'violin')
sns.catplot(data = laptops_df2, x = 'os', y = 'price', kind = 'strip')
# 카테고리별로 얼만큼 분포가 나눠져 있는지 볼수있다.
laptops_df2['processor_brand'].unique() # 프로세서
sns.catplot(data = laptops_df2, x = 'os', y = 'price', kind = 'strip', hue = 'processor_brand')
# hue옵션을 색을 다르게 해주는 옵션, 하지만 점들이 겹쳐져있어서 보기 불편
sns.catplot(data = laptops_df2, x = 'os', y = 'price', kind = 'swarm', hue = 'processor_brand')
# kind = 'swarm'으로 하면 점들이 겹치지않고 펼쳐져서 보기 편해진다.
print()


# 흡연 여부 카테고리에 따라 보험금을 살펴볼수 있는 그래프를 출력 
print('흡연 여부 카테고리에 따라 보험금을 살펴볼수 있는 그래프를 출력')
insurance_df = pd.read_csv('insurance.csv')
sns.catplot(data = insurance_df, x = 'smoker', y = 'charges', kind = 'violin')
print()


# 중간값
print('중간값')
# 데이터셋에서 딱 중간에 있는 값

# 홀수 개수일 때
# 32, 48, 56, 78, 86, 96, 100
# 가운데 있는 78이 중간값이다.

# 짝수 개수일 때
# 7, 11, 12, 15, 16, 21, 24, 29
# 가운데에 있는 15, 16의 평균인 15.5가 중간값이다.
print()


# 평균값 vs 중간값
print('평균값 vs 중간값')

print((13 + 16 + 23 + 35 + 37 + 43 + 48 + 52 + 82 + 120 + 63000) / 11)
print((13 + 16 + 23 + 35 + 37 + 43 + 48 + 52 + 82 + 120) / 10)
# 값이 많이 차이났을때 평균값은 데이터셋의 중심을 제대로 표현하지 못한다.
print()


# 상관계수
print('상관계수')

# 통계에서는 연관성을 표현하는 수치를 상관계수
# 가장 널리쓰이는것이 피어슨 상관계수이다.
# 피어슨 상관계수는 -1부터 1까지 표현 가능한데
# 0이면 연관이 없고 1에 가까워 질수록 연관성이 커진다.
# 반대로 -1에 가까워지면 x랑 y가 반대의 관계이다.
print()


# 상관 계수 시각화
print('상관 계수 시각화')
exam_df3 = pd.read_csv('exam.csv')
exam_df3

print(exam_df3.corr())
# corr()메소드를 사용하면 숫자데이터 사이의 상관 계수를 보여준다.
# 하지만 숫자가 많으면 한눈에 들어오지 않을 수 있다.
# 이럴때는 히트맵을 사용하면 좋다.
sns.heatmap(exam_df3.corr())
# 색이 밝을수록 상관계수가 높다.
sns.heatmap(exam_df3.corr(), annot = True)
# 여기에 annot = True 옵션을 써주면 숫자도 함께 표현해 준다.
print()


# EDA란?
print('EDA란?')
# Exploratory Data Analysis(EDA)
# 탐색적 데이터 분석
# 데이터셋을 다양한 관점에서 살펴보고 탐색하면서 인사이트를 찾는 것
print()


# 기본 정보 파악하기
print('기본 정보 파악하기')
young_survey_df = pd.read_csv('young_survey.csv')
# 997명에게 설문조사 column 147개
# 0 ~ 18 : 음악 취향
# 19 ~ 30 : 영화 취향
# 31 ~ 62 : 취미/관심사
# 63 ~ 72 : 공포증
# 73 ~ 75 : 건강 습관
# 76 ~ 132 : 성격, 인생관 등
# 133 ~ 139 : 소비 습관
# 140 ~ 146 : 기본 정보
basic_info = young_survey_df.iloc[:,140:] # 기본정보
basic_info.describe() # 숫자로 된 컬럼들만 나옴
basic_info['Gender'].value_counts()
basic_info['Handedness'].value_counts()
basic_info['Education'].value_counts()
sns.violinplot(data = basic_info, y = 'Age')
# 10대 후반에서 20대 초반이 많다.
sns.violinplot(data = basic_info, x = 'Gender', y = 'Age')
sns.violinplot(data = basic_info, x = 'Gender', y = 'Age', hue = 'Handedness')
sns.jointplot(data = basic_info, x = 'Height', y = 'Weight')
print()


# 요즘 인기있는 직업은?
print('요즘 인기있는 직업은?')
occupations_df = pd.read_csv('occupations.csv')
occupations_df

# 여성이 가장 많이 종사하고 있는 상위 직종 3개
women = occupations_df[occupations_df['gender'] == 'F']
women['occupation'].value_counts()
# 가장 인기있는 직종 3개 : student, other, administrator

# 남성이 가장 많이 종사하고 있는 직종 상위 3개
men = occupations_df[occupations_df['gender'] == 'M']
men['occupation'].value_counts()
# 가장 인기있는 직종 3개 : student, educator, other
print()


# 상관 관계 분석
print('상관 관계 분석')
young_survey_df2 = pd.read_csv('young_survey.csv')
music = young_survey_df2.iloc[:,:19] # 음악과 관련된 컬럼
music
# 각 데이터에 1.0 ~ 5.0까지 데이터가 들어가 있다.
# 가장좋아하면 5.0, 싫어하면 1.0이다.
sns.heatmap(music.corr())

young_survey_df2.corr()['Age'].sort_values(ascending = False)
# 나이에 따른 상관계수
print()

# 브런치 카페 음악 셀렉션
print('브런치 카페 음악 셀렉션')
# "Getting up"이라는 coumn을 보면 5라고 대답한 사람들은 
# 아침에 일어나기 어려운 사람이고 1이라고 대답한 사람은
# 아침에 쉽게 일어나는 사람이다. 이 데이터로 봤을때
# 아침에 일찍일어나는 사람들이 가장 좋아할 만한 음악 장르는??
young_survey_df3 = pd.read_csv('young_survey.csv')
get_up = young_survey_df3.corr()['Getting up']
# 알침에 일어나는 것에 따른 상관계수
get_up[1:19].sort_values()
# 음악 장르
print()


# 스타트업 아이템 탐색하기
print('스타트업 아이템 탐색하기')
young_survey_df4 = pd.read_csv('young_survey.csv')
young_survey_df4
# 영준이는 스타트업을 준비하고 있다.
# 사업 아이템을 고민하면서 나름대로 가설을 몇 개 세웠다.
# 1. 악기를 다루는 사람들은 시 쓰기를 좋아하는 경향이 있을 것이다.
# 2. 외모에 돈을 많이 투자하는 사람들은 브랜드 의류를 선호할 것이다.
# 3. 메모를 자주 하는 사람들은 새로운 환경에 쉽게 적응할 것이다.
# 4. 워커홀릭들은 건강한 음식을 먹으려는 경향이 있을 것이다.
# 설문조사 데이터를 바탕으로 가장 가능성이 낮은 가설은?

# 가성과 관련있는 컬럼
# Branded clothing: 나는 브랜드가 없는 옷보다 브랜드가 있는 옷을 선호한다.
# Healthy eating: 나는 건강하거나 품질이 좋은 음식에는 기쁘게 돈을 더 낼 수 있다.
# Musical instruments: 나는 악기 연주에 관심이 많다.
# New environment: 나는 새 환경에 잘 적응하는 편이다.
# Prioritising workload: 나는 일을 미루지 않고 즉시 해결해버리려고 한다.
# Spending on looks: 나는 내 외모에 돈을 많이 쓴다.
# Workaholism: 나는 여가 시간에 공부나 일을 자주 한다.
# Writing: 나는 시 쓰기에 관심이 많다.
# Writing notes: 나는 항상 메모를 한다.

# 1. 
re_1 = young_survey_df4.loc[:, ['Musical instruments', 'Writing']]
re_1.corr()
# 0.343816

# 2. 
re_2 = young_survey_df4.loc[:, ['Spending on looks', 'Branded clothing']]
re_2.corr()
# 0.418399

# 3. (가장낮은 -0.079397을 가지고 있다.)
re_3 = young_survey_df4.loc[:, ['Writing notes', 'New environment']]
re_3.corr()
# -0.079397

# 4. 
re_4 = young_survey_df4.loc[:, ['Workaholism', 'Healthy eating']]
re_4.corr()
# 0.238644
print()


# 클러스터 분석
print('클러스터 분석')
# 클러스터는 데이터들을 몇가지 부류로 만드는것이다
young_survey_df5 = pd.read_csv('young_survey.csv')
interests = young_survey_df5.iloc[:, 31:63]
# 관심사에 대한 컬럼 추출
inter_corr = interests.corr()
# 관심사들끼리의 상관 관계
inter_corr['History'].sort_values(ascending = False)
# 하나의 컬럼마다 하나하나 하면 힘들고 정확하지 않을수도 있다.
sns.clustermap(inter_corr)
# 시각화 하게되면 사다리처럼 엮여있는것이 있다.
# 이것이 서로 연관되어 있다는 것을 보여주는 것이다.
print()


# 영화 카페 운영하기
print('영화 카페 운영하기') 
# 수 많은 영화 DVD를 어떻게 배치해야 할지 고민입니다. 좀 연관된 장르끼리 묶어서 보관해야, 
# 각 손님들의 취향을 잘 맞출 수 있을 것 같습니다. 영화 장르에 대한 column은 
# 'Horror'부터 'Action'까지 입니다. 영화 장르에 대해서 clustermap을 그려 보세요
survey_df = pd.read_csv('survey.csv')
movie = survey_df.loc[:, 'Horror':'Action']
movie_corr = movie.corr()
sns.clustermap(movie_corr)
print()


# 타이타닉 EDA
print('타이타닉 EDA')
titanic = pd.read_csv('titanic.csv')
titanic
# 타이타닉호의 침몰은 무려 1514명 정도가 사망한 비운의 사건이다.
# 영화 ‘타이타닉’으로 인해 이름이 가장 널리 알려진 여객선이기도 합니다.
# 당시 탑승자들의 생존 여부, 성별, 나이, 지불한 요금, 좌석 등급 등의 정보가 있다.
# 다양한 방면으로 EDA(탐색적 데이터 분석)를 한 후, 다음 보기 중 맞는 것을 모두 고르세요.

# 생존 여부는 'Survived' column에 저장되어 있습니다. 0이 사망, 1이 생존을 의미합니다.
# 좌석 등급은 'Pclass' column에 저장되어 있습니다. 1은 1등실, 2는 2등실, 3은 3등실을 의미합니다.
# 지불한 요금은 'Fare' column에 저장되어 있습니다.

titanic.columns

# 1.
# 타이타닉의 승객은 30대와 40대가 가장 많다.
titanic_70 = titanic[titanic['Age'] >=  70] # 70대 : 7명
titanic_60 = titanic[(titanic['Age'] >=  60) & (titanic['Age'] <  70)] # 60대 : 19명
titanic_50 = titanic[(titanic['Age'] >=  50) & (titanic['Age'] <  60)] # 50대 : 48명
titanic_40 = titanic[(titanic['Age'] >=  40) & (titanic['Age'] <  50)] # 40대 : 89명
titanic_30 = titanic[(titanic['Age'] >=  30) & (titanic['Age'] <  40)] # 30대 : 167명
titanic_20 = titanic[(titanic['Age'] >=  20) & (titanic['Age'] <  30)] # 20대 : 220명
titanic_10 = titanic[(titanic['Age'] >=  10) & (titanic['Age'] <  20)] # 10개 : 102명
titanic_0 = titanic[titanic['Age'] <  10] # 10살 미만 : 62명
titanic.plot(kind = 'hist', y = 'Age', bins = 50)

# 2.
# 가장 높은 요금을 낸 사람은 30대이다.
titanic_70['Fare'].describe() # 70대 : 71.0
titanic_60['Fare'].describe() # 60대 : 263.0
titanic_50['Fare'].describe() # 50대 : 247.52
titanic_40['Fare'].describe() # 40대 : 227.525
titanic_30['Fare'].describe() # 30대 : 512.3292
titanic_20['Fare'].describe() # 20대 : 263.0
titanic_10['Fare'].describe() # 10대 : 263.0
titanic_0['Fare'].describe() # 10살 미만 151.55
titanic.plot(kind = 'scatter', x = 'Age', y = 'Fare')

# 3.
# 생존자가 사망자보다 더 많다.
titanic['Survived'].value_counts()
# 0(사망) : 549, 1(생존): 342

# 4.
# 1등실, 2등실, 3등실 중 가장 많은 사람이 탑승한 곳은 3등실이다.
titanic['Pclass'].value_counts()
# 1(1등실) : 216, 2(2등실) : 184, 3(3등실) : 491

# 5.
# 가장 생존율이 높은 객실 등급은 1등실이다.
titanic_cs = titanic.loc[:, ['Pclass', 'Survived']]
titanic_cs.value_counts()
sns.kdeplot(titanic['Pclass'], titanic['Survived'])

# 6
# 나이가 어릴수록 생존율이 높다.
titanic_70['Survived'].value_counts() # 1 / 7 (약 14.3%)
titanic_60['Survived'].value_counts() # 6 / 19 (약 31.5%)
titanic_50['Survived'].value_counts() # 20 / 48 (약 41.7%)
titanic_40['Survived'].value_counts() # 34 / 89 (약 38.2%)
titanic_30['Survived'].value_counts() # 73 / 167 (약 43.7%)
titanic_20['Survived'].value_counts() # 77 / 220 (약 35.0%)
titanic_10['Survived'].value_counts() # 41 / 102(약 40.2%)
titanic_0['Survived'].value_counts() # 24 / 62 (약 38.7%)
sns.stripplot(data = titanic, x = 'Survived', y = 'Age')

# 7
# 나이보다 성별이 생존율에 더 많은 영향을 미친다.
titanic.loc[:,['Sex', 'Survived']].value_counts()
# female : 233 / 314(약 74.2%)
# male : 109 / 577(약 18.89%)
sns.stripplot(data = titanic, x = 'Survived', y = 'Age', hue = 'Sex')
print()


# =============================================================================
