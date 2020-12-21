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






# DataFrame 다루기
# =============================================================================
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
body_imperial_df
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

# 1. 주어진 데이터에는 총 몇 개의 도시와 몇 개의 나라가 있는가
print('1. 주어진 데이터에는 총 몇 개의 도시와 몇 개의 나라가 있는가')
print(world_cities_df)
print(world_cities_df['City / Urban area'].describe())
# world_cities_df['City / Urban area'].value_counts().shape
print(world_cities_df['Country'].describe())
# world_cities_df['Country'].value_counts().shape
print()

# 2. 주어진 데이터에서, 인구 밀도(명/sqKm) 가 10000 이 넘는 도시는 총 몇 개인가
print('2. 주어진 데이터에서, 인구 밀도(명/sqKm) 가 10000 이 넘는 도시는 총 몇 개인')
print(world_cities_df)
wc_count_df = world_cities_df[world_cities_df['Population'] / world_cities_df['Land area (in sqKm)'] > 10000]
print(wc_count_df.shape)
# wc_count_df.info()
print()

# 3. 인구 밀도가 가장 높은 도시를 찾아봅시다.
print('3. 인구 밀도가 가장 높은 도시를 찾아봅시다.')
world_cities_df['인구밀도'] = world_cities_df['Population'] / world_cities_df['Land area (in sqKm)']
print(world_cities_df.sort_values(by = '인구밀도', ascending = False).head(1))
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
print()


# 강의실 배정하기2
print('강의실 배정하기2')
enrolment_df3 = pd.read_csv('enrolment_3.csv')

# 아래 세 가지 조건을 만족하도록 코드를 작성
# 1. 같은 크기의 강의실이 필요한 과목에 대해 알파벳 순서대로 방 번호를 배정하세요.
#     예를 들어 Auditorium이 필요한 과목으로 “arts”, “commerce”, “science” 세 과목이 있다면, 
#     “arts”는 “Auditorium-1”, “commerce”는 “Auditorium-2”, “science”는 “Auditorium-3” 
#     순서로 방 배정이 되어야 한다.
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
# =============================================================================






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
gdp_df

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


# 새로운 값 계산하기
print('새로운 값 계산하기')
broadcast_df9 = pd.read_csv('broadcast.csv', index_col = 0)
broadcast_df9
broadcast_df9.plot()
broadcast_df9['KBS'] + broadcast_df9['MBC'] + broadcast_df9['SBS'] + broadcast_df9['TV CHOSUN'] + broadcast_df9['JTBC'] + broadcast_df9['Channel A'] + broadcast_df9['JTBC'] + broadcast_df9['MBN']
# 이것을 더 깔끔하게 쓴다면
broadcast_df9.sum(axis = 'columns')
# 또는 axis = 1이라 쓰면 된다.
broadcast_df9['Total']= broadcast_df9.sum(axis = 'columns')
broadcast_df9
broadcast_df9.plot(y = 'Total')
# 시청률이 매년 떨어지는것을 볼 수 있다.
broadcast_df9['Group 1'] = broadcast_df9.loc[:, 'KBS':'SBS'].sum(axis = 'columns')
# 지상파 시청률 합
broadcast_df9['Group 2'] = broadcast_df9.loc[:, 'TV CHOSUN':'MBN'].sum(axis = 'columns')
# 종편채널 시청률 합
broadcast_df9.plot(y = ['Group 1', 'Group 2'])
# 지상파 시청률은 떨어지고있고 종편채널의 시청률은 올라가는것을 확인할 수 있다.
print()


# 문자열 필터링
print('문자열 필터링')
# albums_df = pd.read_csv('albums.csv', engine = 'python')
albums_df = pd.read_csv('albums.csv', encoding = 'latin1')
# 파일이 latin1방식으로 인코딩 되어있어서 불러왔을때 글자가 깨지지 않기 위한 옵션
albums_df
albums_df['Genre'].str.contains('Blues')
# 음악 장르중 Blues장르 음악을 추출하고 싶을 때
albums_df[albums_df['Genre'].str.contains('Blues')]['Genre']

albums_df['Genre'].str.startswith('Blues')
# 음악 장르중 Blues가 앞에 나와있는거만 추출하고 싶을 때
albums_df[albums_df['Genre'].str.startswith('Blues')]['Genre']
print()


# 박물관이 살아 있다1
print('박물관이 살아 있다1')
# 대학 박물관을 개선하기 위해 다음과 같이 박물관을 분류하기로 하였습니다.

# 박물관은 대학/일반 박물관으로 나뉜다.
# 시설명에 '대학'이 포함되어 있으면 '대학', 그렇지 않으면 '일반'으로 나누어 '분류' column에 입력한다.
# '분류' column을 만들어서 솔희를 도와주세요!
museum_1_df = pd.read_csv('museum_1.csv')
museum_1_df['분류'] = np.where(museum_1_df['시설명'].str.contains('대학'), '대학', '일반')
museum_1_df
print()


# 문자열 분리
print('문자열 분리')
parks_df = pd.read_csv('parks.csv')
parks_df['소재지도로명주소'].str.split(n = 1, expand = True)[0]
# 관할 구역을 추출 n = 1(첫번째까지만 나눔), expand = True(데이터프레임 형식으로) 
parks_df['관할구역'] = parks_df['소재지도로명주소'].str.split(n = 1, expand = True)[0]
parks_df
print()


# 박물관이 살아 있다2
print('박물관이 살아 있다2')
# 어느 지역에 박물관이 많은지 분석해보려 한다. 하지만 주어진 데이터에는 
# 주소가 없고 전화번호 앞자리가 지역을 나타낸다는 것을 알수있다.
# 박물관의 위치를 파악할 수 있게 '운영기관전화번호' column의 
# 맨 앞 3자리를 추출하고, '지역번호' column에 넣어라.
museum_2_df = pd.read_csv('museum_2.csv')
area_code = museum_2_df['운영기관전화번호'].str.split('-', n = 1, expand = True)[0]
museum_2_df['지역번호'] = area_code
print()


# 카테고리로 분류
print('카테고리로 분류')
laptops_df3 = pd.read_csv('laptops.csv')
laptops_df3['brand'].unique()
# 브랜드가 있지만 각 브랜드를 만든 국가별로 분석하고 싶다.
brand_nation = {
    'Dell' : 'U.S.',
    'Apple' : 'U.S.',
    'Acer' : 'Taiwan',
    'HP' : 'U.S.',
    'Lenovo' : 'Chine',
    'Alienware' : 'U.S.',
    'Microsoft' : 'U.S.',
    'Asus' : 'Taiwan'
}
laptops_df3['brand_nation'] = laptops_df3['brand'].map(brand_nation)
# brand의 값이 dictionary의 키값에 매칭 되어 그안에 있는 값이 들어간다.
laptops_df3


# 박물관이 살아 있다3
print('박물관이 살아 있다3')
# 지역번호가 02이면 '서울시'이고, 지역번호가 064라면 '제주도'입니다.
# '지역번호' column을 '지역명' 으로 변경하고, 아래 규칙에 따라 지역을 넣어라
# 서울시: 	02
# 경기도:	031, 032
# 강원도	:   033
# 충청도:   041, 042, 043, 044
# 부산시:	051
# 경상도:	052, 053, 054, 055
# 전라도:	061, 062, 063
# 제주도:	064
# 기타:   	1577, 070
museum_3_df = pd.read_csv('museum_3.csv', dtype = {'지역번호' : str})
museum_3_df
area_dic = {
    '02': '서울시',
    '031' : '경기도',
    '032' : '경기도',
    '033' : '강원도',
    '041' : '충청도',
    '042' : '충청도',
    '043' : '충청도',
    '044' : '충청도',
    '051' : '부산시',
    '052' : '경상도',
    '053' : '경상도',
    '054' : '경상도',
    '055' : '경상도',
    '061' : '전라도',
    '062' : '전라도',
    '063' : '전라도',
    '064' : '제주도',
    '1577' : '기타',
    '070' : '기타'
}
museum_3_df['지역번호'] = museum_3_df['지역번호'].map(area_dic)
museum_3_df.rename(columns = {'지역번호' : '지역명'}, inplace = True)
museum_3_df
print()


# groupby 
print('group')
laptops_df3
# brand_nation컬럼의 나라 카테고리별로 나누고 싶다.
nation_groups = laptops_df3.groupby('brand_nation')
type(nation_groups)
nation_groups.count()
# 나라별로 각 컬럼에 몇개씩 있는지 알수 있다.
# nation_groups.max()
# 현재 pandas버전에서는 안됨
nation_groups.mean()
# 나라별 각 컬럼의 평균값
nation_groups.first()
# 나라별 제일 첫번째에 있는 값
nation_groups.last()
# 나라별 제일 마지막에 있는 값
nation_groups.plot(kind = 'box', y = 'price')
# 나라별 가격 box플롯
nation_groups.plot(kind = 'hist', y = 'price')
# 나라별 가격 히스토그램
print()


# 직업 탐구하기1
print('직업 탐구하기1')
# 각 직업의 평균 나이가 궁금하다.
# groupby 문법을 사용해서 '평균 나이'가 어린 순으로 직업을 나열하라.
occupations_df2 = pd.read_csv('occupations.csv')
occupation_groups = occupations_df2.groupby('occupation')
occupation_groups.mean()['age'].sort_values()
print()


# 직업 탐구하기2
print('직업 탐구하기2')
# 이번에는 여자 비율이 높은 직업과, 남자 비율이 높은 직업이 무엇인지 궁금하다.
# groupby 문법을 사용해서 '여성 비율'이 높은 순으로 직업을 나열하라.
# DataFrame이 아닌 Series로, 'gender'에 대한 값만 출력되어야 한다.
occupations_df3 = pd.read_csv('occupations.csv')
occupation_groups2 = occupations_df3.groupby('occupation')
gender_f = occupations_df3[occupations_df3['gender'] == 'F']
gender_f.groupby('occupation').count()['gender']
gen_f = gender_f.groupby('occupation').count()['gender'] / occupation_groups2.count()['gender']
gen_f.sort_values(ascending = False).fillna(0)

# 다른방법
# occupation_groups2.mean()
# occupations_df3['gender'] = np.where(occupations_df3['gender'] == 'M', 0, 1)
# occupation_groups2 = occupations_df3.groupby('occupation')
# occupation_groups2.mean()['gender'].sort_values(ascending = False)
print()


# 데이터 합치기
print('데이터 합치기')
# 합치는 4가지 방법
# 1. inner join : 겹치는 부분만 합친다.(교집합)
# 2. left outer join : 왼쪽 데이터에 존재하는 데이터기준으로 합친다.
# 3. right outer join : 오른쪽 데이터에 존재하는 데이터기준으로 합친다.
# 4. full outer join : 양쪽데이터 전부 합친다.

vegetable_price_df = pd.read_csv('vegetable_price.csv')
vegetable_quantity_df = pd.read_csv('vegetable_quantity.csv')
vegetable_price_df
vegetable_quantity_df

pd.merge(vegetable_price_df, vegetable_quantity_df, on ='Product')
# 기본값이 inner join
pd.merge(vegetable_price_df, vegetable_quantity_df, on ='Product', how = 'left')
# vegetable_price_df 데이터 기준으로 합친다.
pd.merge(vegetable_price_df, vegetable_quantity_df, on ='Product', how = 'right')
# vegetable_quantity_df 데이터 기준으로 합친다.
pd.merge(vegetable_price_df, vegetable_quantity_df, on ='Product', how = 'outer')
# 양족데이터 전부 합친다.
print()


# 박물관이 살아 있다4
print('박물관이 살아 있다4')
# 파이썬 사전과 .map()을 사용해서 지역명을 알아낸 솔희는, 조금 더 편한 방법을
# 고민하던 중, '지역번호와 지역명에 대한 데이터는 누군가 이미 만들어두지 않았을까'라는 생각을 하게 되는데
# 인터넷에서 지역번호와 지역명이 있는 데이터 region_number.csv를 구했다
# 이 데이터를 먼저 살펴보고, .merge() 메소드를 활용해서 museum_3.csv에 '지역명' column을 추가.
# 단, museum_3.csv의 박물관 순서가 유지되어야 한다.
region_number_df = pd.read_csv('region_number.csv', dtype = {'지역번호' : str})
museum_3_df2 = pd.read_csv('museum_3.csv', dtype = {'지역번호' : str})
museum_3_df2 = pd.merge(museum_3_df2, region_number_df, on = '지역번호', how = 'left')
museum_3_df2
print()
# =============================================================================






# 데이터 퀄리티 높이기
# =============================================================================
# 데이터 퀄리티의 중요성

# 좋은 데이터의 기준 : 완결성
# 만약 회원가입을 할 때 필수항목을 입력하지 않으면 다음단계로 넘어가지 못한다.
# 그건 바로 데이터의 완결성이 위배되기 때문이다.NaN이 대표적인 예

# 데이터 완결성은 어떻게 알 수 있을까?
# 결측값(채워져야 하는데 비어있는 값)이 있는지 확인해야한다.
# NaN이 대표적인 예

# 좋은 데이터의 기준 : 유일성
# 동일한 데이터가 불필요하게 중복되어 있으면 안 됨

# 이메일 인증, 주민등록번호 본인 확인, 휴대폰 번호 본인 확인 => 데이터의 유일성 유지


# 좋은 데이터의 기준 : 통일성
# 데이터가 통일한 형식으로 저장돼 있어야 함

# 형식이란?
# 데이터 타입, 단위, 포맷 등 다양한것을 의미


# 좋은 데이터의 기준 : 정확성
# 데이터가 정확해야 한다.
# 주로 데이터를 모으는 과정에서 발생
# 대표적으로 이상점이 있다면 올바르게 측정되었는지 확인해볼 필요가 있다.


# 데이터 클리닝 : 완결성1
print('데이터 클리닝 : 완결성1')
attendance_df = pd.read_csv('attendance.csv', index_col = 0)
attendance_df

attendance_df.isnull()
# isnull() : NaN이 있는값에는 True반환

attendance_df.isnull().sum()
# NaN의 갯수 확인
print()


# 데이터 클리닝 : 완결성2
print('데이터 클리닝 : 완결성2')
attendance_df2 = pd.read_csv('attendance.csv', index_col = 0)
attendance_df2

attendance_df2.dropna()
# NaN이 있는 행 제거
# 이렇게하면 2012, 2010, 2013년 데이터가 사라진다(분석하기 힘들어진다.)

attendance_df2
# 잘 보면 배구 컬럼에만 NaN이 있는것을 확인할 수 있다.
attendance_df2.dropna(axis = 'columns')
# NaN이 있는 배구컬럼만 제거
# 배구컬럼은 분석할수 없다.

attendance_df2.fillna(0)
# NaN을 모두 0으로 변경

attendance_df2.fillna(attendance_df2.mean())
# NaN을 배구의 평균값으로 변경


# 스팀 게임 데이터 정리하기
print('스팀 게임 데이터 정리하기')
steam_1_df = pd.read_csv('steam_1.csv')
steam_1_df

# 스팀 플랫폼에서 가장 반응이 좋은 게임이 무엇인지 알아보려고 한다.
# 데이터를 살펴보니 결측값이 있다. 분석에 앞서 결측값을 제거해 보자.
# 결측값이 있는 모든 row를 삭제하고, DataFrame을 출력
steam_1_df.dropna(inplace = True)
steam_1_df
print()


# 데이터 클리닝 : 유일성
print('데이터 클리닝 : 유일성')
dust_df = pd.read_csv('dust.csv', index_col = 0)
dust_df
dust_df.index
dust_df.index.value_counts()
dust_df.loc['07월 31일', :]
# 중복된 인덱스 있는지 확인

dust_df.drop_duplicates(inplace = True)
dust_df
# 중복된 날짜 데이터 삭제

dust_df = dust_df.T.drop_duplicates().T
dust_df
# 중복된 지역 데이터 삭제
# T는 전치 메서트 컬럼와 인덱스를 바꾼다
# drop_duplicates는 인덱스나 컬럼이름이 달라도
# 그 안에 있는 데이터들이 전부 같으면 중복된 데이터로 인식해서 지운다.
print()


# 데이터 클리닝 : 정확성1

# 이상점이 잘못된 데이터라면?
# 고치거나, 제거해야 한다.

# 이상점을 판단하는 기준
# 박스 플롯에서 25%지점을 Q1 75%지점을 Q3이라 한다.
# 그리고 Q3와 Q1의 거리를 IQR이라 한다.
# Q1보다 1.5 x IQR보다 밑에 있거나 Q3보다 1.5 x IQR보다 위에 있으면 
# 이상점이라 한다.

# 이상점이 제대로 된 데이터라면?
# 분석에 방해가되면 제거하고, 의미있는 정보라면 그냥 둔다.


# 데이터 클리닝 : 정확성2
print('데이터 클리닝 : 정확성2')
beer_df = pd.read_csv('beer.csv', index_col = 0)
beer_df

beer_df.plot(kind = 'box', y = 'abv')
# 알콜 도수 박스플롯

beer_df['abv'].describe()
q1 = beer_df['abv'].quantile(0.25)
q3 = beer_df['abv'].quantile(0.75)
q1
q3
# 알코도수 25% 지점과 75%지점 확인

iqr = q3 - q1
beer_df[(beer_df['abv'] < q1 - 1.5 *iqr) | (beer_df['abv'] > q3 + 1.5 * iqr)]
# 이상점 값 구하기

beer_df.loc[2250, 'abv'] = 0.055
beer_df.drop([963, 1856], axis = 'index', inplace = True)
beer_df[(beer_df['abv'] < q1 - 1.5 *iqr) | (beer_df['abv'] > q3 + 1.5 * iqr)]
# 이상점 데이터 수정 및 제거

beer_df.plot(kind = 'box', y = 'abv')
# 데이터 정제 후 알콜 도수 박스플롯
print()


# 데이터 클리닝 : 정확성3
print('데이터 클리닝 : 정확성3')
exam_outlier_df = pd.read_csv('exam_outlier.csv')
exam_outlier_df
# 관계적 이상점
# 두 변수의 관계를 고려했을 때 이상한 데이터

# exam_outlier_df.plot?
exam_outlier_df.plot(kind = 'scatter', x  = 'reading score', y = 'writing score')
exam_outlier_df.corr()
# 읽기점수와 쓰기점수의 산점도 그래프와 상관계수 출력

exam_outlier_df[exam_outlier_df['writing score'] > 100]
# 쓰기점수가 100보다 큰 점수가 있다.(이상점)

exam_outlier_df.drop(51, axis = 'index', inplace = True)
exam_outlier_df.plot(kind = 'scatter', x = 'reading score', y = 'writing score')
exam_outlier_df.corr()
# 제거 후 산점도 그래프와 상관계수 출력

rw_score = (exam_outlier_df['writing score'] > 90) & (exam_outlier_df['reading score'] < 40)
exam_outlier_df[rw_score]
# 혼자 동떨어져있는 데이터 출력

exam_outlier_df.drop(373, axis = 'index', inplace = True)
exam_outlier_df.plot(kind = 'scatter', x = 'reading score', y = 'writing score')
exam_outlier_df.corr()
# 위 데이터 제거후 산점도 그래프와 상관계수 출력
print()


# 영화 평점 분석하기1
print('영화 평점 분석하기1')
movie_metadata_df = pd.read_csv('movie_metadata.csv')
movie_metadata_df
# 영화 감독이 꿈인 래진이는 영화에 대한 데이터 분석을 해보려고 한다.
# movie_metadata.csv에는 영화에 대한 제목, 감독, 배우, 평점, 예산 등의 정보가 있는데
# 과연 예산을 많이 쓰면 소비자 평점이 높아질 지 궁금하다.
# 산점도를 그려봤더니, 아주 큰 예산을 쓴 영화 몇 개 때문에 상관 관계를 파악할 수가 없다.
# 너무 예산이 큰 일부 영화를 제거하고, 다시 분석해봐야 할 것 같다.
# 예산을 기준으로 75% 지점에서 5 IQR 만큼 더한 것보다 큰 예산의 영화는 제거하고, 다시 산점도를 그려라.
movie_metadata_df.columns
movie_metadata_df.plot(kind = 'scatter', x = 'imdb_score', y = 'budget')
movie_metadata_df['budget'].describe()
q3 = movie_metadata_df['budget'].quantile(0.75)
q1 = movie_metadata_df['budget'].quantile(0.25)

iqr = q3 - q1 

mv_budget = movie_metadata_df['budget'] > q3 + 5 * iqr
movie_metadata_df.drop(movie_metadata_df[mv_budget].index, inplace = True)
movie_metadata_df.plot(kind = 'scatter',  x = 'budget', y = 'imdb_score')
print()


# 영화 평점 분석하기2
print('영화 평점 분석하기2')
movie_metadata_df2 = pd.read_csv('movie_metadata.csv')
movie_metadata_df2
# 이번에도 예산이 너무 큰 영화 몇 개를 제거해보려고 한다.
# 하지만 이번에는 IQR이 아니라 예산 상위 15개를 제거하려고 하는데
# movie_metadata.csv에서 예산이 가장 높은 15개 영화를 제거하고, 산점도를 그려라.
mv_budget = movie_metadata_df2['budget'].sort_values(ascending = False)[:15].index
movie_metadata_df2.drop(mv_budget, inplace = True)
movie_metadata_df2.plot(kind = 'scatter',  x = 'budget', y = 'imdb_score')
print()
# =============================================================================





# 데이터 만들기
# =============================================================================
# 데이터 만들기의 중요성

# 데이터를 분석하기전에
# 우선 데이터를 모아야 한다.


# 데이터 다운로드 받기

# 데이터를 구하는 가장 쉬운 방법은, 이미 누군가 만들어둔 데이터를 사용하는 것이다.
# 대표적으로, 국가 기관에서는 공익 목적으로 여러 데이터를 공개한다.  
# 그 외에도 데이터를 검색하는 사이트나, 데이터를 공유하는 사이트들이 있다.

# 국내 사이트

# 서울열린데이터광장  
# https://data.seoul.go.kr/

# 공공데이터포털  
# https://www.data.go.kr

# e-나라지표  
# http://www.index.go.kr/

# 국가통계포털  
# http://kosis.kr

# 서울특별시 빅데이터 캠퍼스  
# https://bigdata.seoul.go.kr/

# 통계청  
# http://kostat.go.kr/

# 해외 사이트

# 구글 데이터 검색  
# https://toolbox.google.com/datasetsearch

# 캐글  
# https://www.kaggle.com/datasets

# Awesome Public Datasets Github  
# https://github.com/awesomedata/awesome-public-datasets

# Data and Story Library  
# https://dasl.datadescription.com/

# 데이터허브  
# https://datahub.io/

# 하지만 데이터에 저작권이 있기도 하니, 
# 실제로 데이터를 활용할 때는 잘 확인하고 사용하셔야 한다.


# 센서 사용하기

# 센서 : 물리적인 현상을 감지하서 전기 신호로 변환 해 주는 장치
# 아두이노, 라즈베리 파이에 약간의 프로그래밍을하면
# 센서를 활용한 데이터 수집이 가능하다.


# 웹에서 모으기

# 가장 빠르게 데이터가 쌓이는 곳은 인터넷이다.

# 웹에있는 데이터를 수집할 때 많이 사용하는 것이
# 웹 스크레이핑과 웹 크롤링이다.

# 하나의 특정 웹페이지에서 원하는 정보를 받아오는 것을 웹 스크레이핑이라 한다.
# 웹페이지를 프로그램을 짜서 컴퓨터가 자동으로 여러 웹페이지를 수집하게 하는 것을
# 웹크롤링이라 한다. 이러한 것을 해주는 프로그램을 웹크롤러라고 부른다


# HTML 얼마나 많이 알아야 할까요?

# 데이터를 모으려는 우리에게 중요한 건 태그의 구조입니다.  
# 태그의 구조를 잘 이해한다면, 각 태그가 어떤 기능을 하는지 
# 정확히 몰라도 데이터를 얻어올 수 있다.

# HTML 태그의 구성
# HTML 태그는 두 가지 요소로 구성되어 있다.  
# 태그 이름과 속성(attribute)이다.

# 1. 태그 이름
# 태그 이름은 계속 봐왔던 p, li, img 이런 것들이다.  
# 태그를 상징하는 <> 기호 안에 태그 이름을 가장 먼저 넣어주게 된다.  
# <p>, <li>, <img> 처럼

# 2. 속성
# 모든 HTML 태그는 속성이라는 추가 정보를 가질 수 있다. 
# 태그 이름이 아닌 것은 모두 속성이라고 생각하면 된다.
# 속성은 일반적으로 속성 이름과 속성 값을 하나의 쌍으로 갖게 된다. (예: name="value")  
# 만약 HTML 태그가 <p>, </p> 태그처럼 둘로 나누어져 있다면, 시작 태그에 속성을 적어준다.

# <li id="favorite">우유</li>
# 위 <li> 태그에는 id라는 속성이 있고, 그 값은 "favorite"이다
# 한 태그가 여러 속성을 가질 수도 있다.

# <img alt="logo" class="logo-img" src="/images/music/logo.png"/>
# 위 img태그에는 총 3개의 속성이 있다.
# alt라는 속성은 "logo"라는 값을, class라는 속성은 "logo-img"라는 값을, 
# src라는 속성은 "/images/music/logo.png"라는 값을 각각 갖고 있다.

# HTML 태그의 구조

# 하나의 HTML 태그에 대해 이해했다면, 태그 사이의 관계에 대해서도 이해해야 한다.
# 한 페이지의 HTML 태그는 서로 연결되어 있다.  
# 이 구조가 마치 가계도나 나무(트리)와 유사해서, 부모 관계라고 부르거나 트리 구조라도고 부른다.
# <ul>
#     <li>커피</li>
#     <li>녹차</li>
#     <li>우유</li>
# </ul>
# 여기서 <ul> 태그가 <li> 태그를 포함하고 있으니 <ul> 태그가 부모, <li> 태그가 자녀인 셈이다.

# HTML, 얼마나 알아야 하나요?

# 목표가 무엇이냐에 따라 다르다.
# 웹사이트에서 데이터를 모을 수 있는 정도가 목표라면, 앞서 말한 태그의 구조를 이해할 수 있는 수준이면 된다. 
# 내부 원리를 자세히는 몰라도, 작동하는 프로그램을 만들 수는 있다. 
# 더 나아가는 것은 여러분의 선택입니다.
# 만약 기본적인 데이터 수집을 넘어 자동화 봇이나 복잡한 사이트의 
# 크롤링 등 고급 응용을 할 수 있는 전문가 수준을 원한다면, 
# 반드시 HTML과 CSS를 능숙하게 다루어야 한다. 
# HTML과 CSS는 웹 크롤링과 웹 스크레이핑의 핵심 기술이기 때문이다.

# 집의 구조를 가장 잘 이해하는 사람은 집을 짓는 사람이고  
# 웹 사이트의 구조를 가장 잘 이해하는 사람은 웹 개발자다.  
# HTML과 CSS이 능숙하면, 기본적인 데이터 수집 이상으로 무궁무진한 활용 가능성이 있다.


# 서버와 클라이언트

# 서버 -> 클라이언트
# 서버 : 서비스를 제공하는쪽
# 클라이언트 : 서비스를 제공 받는 쪽


# 파이썬으로 서버에 요청 보내기
print('파이썬으로 서버에 요청 보내기')
import requests
page = requests.get('https://www.google.com')
# 서버에 요청
type(page)
page.text
# 응답에 대한 내용(HTML)을 볼수있다.
print()


# TV시청률 크롤링1
print('TV시청률 크롤링1')
response = requests.get('https://workey.codeit.kr/ratings/index')
rating_page = response.text
print(rating_page)
print()


# 웹 사이트 주소 이해하기
print('웹 사이트 주소 이해하기')

# https://www.ikea.com/catalog/news?sorting=price&pageNumber=4
# 소통방식 : http, https
# 도메인 이름 : www.ikea.com
# 경로 : catalog/news?
# 쿼리스트링(파라미터) : sorting=price&pageNumber=4

# https://en.wikipedia.org/wiki/Computer_programming#Debugging
# 위치 지정 : #Debugging
print()


# TV 시청률 크롤힐2
print('TV 시청률 크롤힐2')
# 우리가 원하는 모든 기간의 데이터를 뽑아내기 위해, 티비랭킹닷컴 사이트를 자세히 살펴보자.

# 웹사이트의 주소 구조를 파악해보고, 제공되는 모든 데이터를 받아올 수 있도록 
# 모든 페이지의 HTML 코드(response의 text)를 가져와서 rating_pages에 저장해라.
# 2010년 1월부터 2018년 12월까지 모든 달에 대해
# 1주차~5주차 페이지를 순서대로 리스트에 넣으면 된다.(모든 달에 5주차가 있다고 가정하세요.)

# year = 2011
# month = 1
# idx = 1
# requests.get(f'https://workey.codeit.kr/ratings/index?year={month}&month={month}&weekIndex={idx}')

# rating_pages = []
# for i in range(2010, 2019):
#     for j in range(1, 13):
#         for z in range(0, 5):
#             re_get = requests.get(f'https://workey.codeit.kr/ratings/index?year={i}&month={j}&weekIndex={z}')
#             rating_pages.append(re_get.text)
# len(rating_pages)
# rating_pages[0]
print()


# 웹 페이지 스타일링 원리

# 데이터 추출의 원리
# 사실 HTML 코드에서 데이터를 골라내는 작업은 
# 웹 페이지를 꾸미는 스타일링 원리와 밀접한 연관이 있다
# 특정 태그를 선택한다는 공통점 때문이다.


# CSS 선택자

# CSS 선택자의 의미
# 기본적으로 CSS 선택자는 HTML 태그의 스타일을 지정하기 위해 사용한다.  
# 하지만 추출할 데이터의 위치를 지정할 때도 활용할 수 있다. 


# 파싱1
print('파싱1')

# 파싱(Parsing)이란?
# '파싱'이란 문자의 구조를 분석해서 원하는 정보를 얻어내는 걸 말한다.
# 복잡한 HTML 코드에서 정보를 뽑아내는 것도 파싱의 일종이다.

# Beautiful Soup
# 파이썬에서는 Beautiful Soup이라는 툴로 HTML을 파싱한다.
# 아래의 HTML 코드에서 '커피', '녹차', '우유'라는 텍스트 데이터를 추출하려고 한다.
html_code = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Website</title>
</head>
<body>
<h2>HTML 연습!</h2>

<p>이것은 첫 번째 문단입니다.</p>
<p>이것은 두 번째 문단입니다!</p>

<ul>
    <li>커피</li>
    <li>녹차</li>
    <li>우유</li>
</ul>

<img src='https://i.imgur.com/bY0l0PC.jpg' alt="coffee"/>
<img src='https://i.imgur.com/fvJLWdV.jpg' alt="green-tea"/>
<img src='https://i.imgur.com/rNOIbNt.jpg' alt="milk"/>

</body>
</html>"""

# 1. BeautifulSoup 타입 만들기
# HTML 코드를 파싱하려면, 먼저 HTML 코드를 'BeautifulSoup 타입'으로 바꿔줘야 한다.  
from bs4 import BeautifulSoup
# bs4 라이브러리의 BeautifulSoup불러오기

soup = BeautifulSoup(html_code, 'html.parser')
# BeautifulSoup타입으로 변

type(soup)
# type출력

# 2. 특정 태그 선택하기
# BeautifulSoup 타입에는 .select() 메소드를 쓸 수 있다
# '선택한다'는 의미 괄호 안에 CSS 선택자를 넣으면 특정 HTML 태그만 선택할 수 있다
# 예를 들어, 모든 <li> 태그를 가져오고 싶으면 CSS 선택자 'li'를 넘겨주면 된다.
li_tags = soup.select('li')
print(li_tags)
# 모든 li태그 출력(리스트로)

print(li_tags[0])
# 첫번째 <li>태그 출력

# 3. 태그에서 문자열 추출하기
# .select()로 가져온 태그는, 사실 그냥 텍스트가 아니다.  
type(li_tags[0])
# 타입을 확인해 보면type이 bs4.element.Tag라 나온다.
# BeautifulSoup 태그에는 여러 기능이 있는데, 그 중 하나가 텍스트 추출이다.
print(li_tags[0].text)
# 첫번째 <li> 태그의 텍스트 출력
print()


# 파싱2
print('파싱2')
html_code = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Website</title>
</head>
<body>
<h2>HTML 연습!</h2>

<p>이것은 첫 번째 문단입니다.</p>
<p>이것은 두 번째 문단입니다!</p>

<ul>
    <li>커피</li>
    <li>녹차</li>
    <li>우유</li>
</ul>

<img src='https://i.imgur.com/bY0l0PC.jpg' alt="coffee"/>
<img src='https://i.imgur.com/fvJLWdV.jpg' alt="green-tea"/>
<img src='https://i.imgur.com/rNOIbNt.jpg' alt="milk"/>

</body>
</html>"""

# <img> 태그의 src 속성에는 일반적으로 이미지 주소가 저장되어 있다.  
# 이미지 주소를 받아올 것이다.  
soup = BeautifulSoup(html_code, 'html.parser')
img_tag = soup.select('img')
print(img_tag)
# 모든 <img>태그 출력

print(img_tag[0])
# 첫 번째 요소 출력

# 태그에 .text를 붙이면 텍스트가 추출된 것처럼 
# 태그에 ["속성 이름"]을 붙여주면 해당 속성의 값을 가져올 수 있다.  
# 이미지 주소는 src라는 속성에 저장되어 있으니, ["src"]라고 붙이면 된다.

print(img_tag[0]['src'])
# 첫 번째 요소의 'src' 속성 값 가져오기

img_src = []
for img in img_tag:
    img_src.append(img['src'])
print(img_src)
# 전체 <img>태그 src 속성값 추출 및 출력
print()


# 크롬 개발자 도구로 선택자 알아내기

# HTML 코드에 웹사이트의 내용물이 모두 드러난다.
# 하지만 코드만으론 구조를 파악하기 어렵다.
# 조금 더 쉽게 HTML 태그의 구조를 알아내려면 
# 크롬 브라우저의 개발자 도구를 사용하면 된다.
# 알고 싶은 요소를 오른쪽 클릭하면 영어로 Inspect 혹은 한국말로 검사라고 나온다.

# 개발자 도구에서 특정 HTML 요소를 오른쪽 클릭하고 
# Copy 메뉴의 Copy Selector를 클릭하면, CSS 선택자가 클립보드에 복사된다.  
# 텍스트 에디터에 붙여넣기하면
# 개발자 도구를 잘 활용하면 파싱이 좀 더 쉬워진다.


# 파싱3
print('파싱3')
# 이번에는 실제 웹사이트의 response를 받아서 파싱할것이다. 
# 음악 사이트의 인기 아티스트를 추출하겠다.

response = requests.get('https://workey.codeit.kr/music/index')
# HTML 코드 받아오기

print(response.text)
# 결과출력

soup = BeautifulSoup(response.text, 'html.parser')
# BeautifulSoup 타입으로 변환

print(soup)
# 결과출력

# 우리가 원하는 인기 아티스트는 <h3>태그에 있다
# 그 아래에 우리가 10명의 아티스트들이 있다.
# 인기 아티스트를  'popular__order'클래스명을 가지는  <ul> 태그가 감싸고 있다. 
# 그리고 그 안에 <li>태그가 <span>태그와 이름을 담고 있다.

# 태그 골라내기

li_tags = soup.select('.popular__order li')
print(li_tags)
# "popular__order" 클래스를 가진 태그에 중첩된 모든 <li> 태그 선택

# popular_artists = []
# for i in li_tags:
#     popular_artists.append(i.text)
# print(popular_artists)
# <li>태그 안에 있는 텍스트만 꺼내기

# 앞뒤 공백만 제거 한다면 우리가 원하는 데이터만 고를수 있다.
# 다시 처음부터 하면
# popular_artists = []
# for i in li_tags:
#     popular_artists.append(i.text.strip())
# print(popular_artists)
# 공백 제거 후 추출
print()


# 그녀의 전화번호를 찾아서
print('그녀의 전화번호를 찾아서')
# 운명적인 그녀를 만났습니다. 하지만 오렌지 보틀에서 일한다는 것 말고는 아는 게 없다.
# 오렌지 보틀의 웹사이트에 가서, 모든 지점의 전화번호를 모아보려고 한다.
# 모든 지점의 전화번호가 포함된 리스트 출력
# response = requests.get('https://workey.codeit.kr/orangebottle/index')
# response.text
# soup = BeautifulSoup(response.text, 'html.parser')

# span_phone = soup.select('.container .phoneNum')

# phone_num = []
# for i in span_phone:
#     phone_num.append(i.text)
# print(phone_num)
# print()


# 검색어 순위 받아오기
print('검색어 순위 받아오기')
# 음악 사이트의 검색어 순위를 받아오려 한다.
# '검색어 순위'의 1위~10위 데이터를 파싱해서 리스트 출력

# response = requests.get('https://workey.codeit.kr/music/index')
# soup = BeautifulSoup(response.text, 'html.parser')
# rank = soup.select('.rank .list')

# rank[0].text.split()[2]

# search_ranks = []
# for i in rank:
#     search_ranks.append(i.text.split()[2])
# print(search_ranks)    
print()


# 필요한 페이지만 가져오기
print('필요한 페이지만 가져오기')
# 앞선 과제 TV 시청률 크롤링2를 돌이켜 보면.  
# 우리는 모든 달에 5주차가 있다고 가정하고, 반복문을 작성했다.
# 하지만 사실 5주차가 없는 달도 있다.  
# 사실, 5주차가 없는 달은 아예 페이지를 안 받아오는게 더 바람직하다.

# 이런 상황은 생각보다 많이 발생하는데
# SSG닷컴에서 'nintendo'를 검색하면 총 몇 개의 결과 페이지가 있을까요?
# 예상할 수가 없다. 심지어 날마다 달라질 수도 있다.

# SSG닷컴에서 'nintendo'를 검색하면, 주소가 
# http://www.ssg.com/search.ssg?target=all&query=nintendo로 바뀐다.

# 다음 페이지로 이동해 보면
# http://www.ssg.com/search.ssg?target=all&query=nintendo&page=2
# 뒤에 &page=2 가 추가된다. 이 부분을 활용해서 페이지를 지정하고 있다.

# nintendo 검색 결과에 999 페이지도 있을까?.  
# http://www.ssg.com/search.ssg?target=all&query=nintendo&page=999로 들어가면 
# "검색어와 일치하는 상품이 없습니다." 라 나온다.

# 여기서 힌트를 얻을 수 있다. 
# 계속해서 페이지를 가져오다가, 이 페이지를 만나면 중단하면 된다.

# 검색 결과가 없는지 아닌지는 어떻게 확인할까?  
# 두 페이지를 개발자 도구로 직접 한번 비교해 보자.  
# 상품이 있는 페이지: http://www.ssg.com/search.ssg?target=all&query=nintendo&page=9  
# 상품이 없는 페이지: http://www.ssg.com/search.ssg?target=all&query=nintendo&page=999

# 빈 페이지인지 확인하는 방법은 여러 가지이겠지만, 
# "검색어와 일치하는 상품이 없습니다."라는 문구를 담고 있는 
# csrch_tip 클래스의 유무로 한번 확인해 보자. csrc_tip 클래스가 없을 때만 페이지를 저장하는 거다.


# 참고로, 연속으로 페이지를 가져오려고 하면 사이트에서 차단을 하는 경우도 있으므로, 
# 한 페이지를 가져온 뒤 3초간 쉬었다가 다음 페이지를 가져오도록 해 보자. 
# 처음에 import time을 하고, 3초 멈추고 싶은 곳에 time.sleep(3) 이라고 적으면 된다.  
import time

# 빈 리스트 생성
# pages = []

# 첫 페이지 번호 지정
# page_num = 1
# while True:
#     # HTML코드 받아오기
#     response = requests.get(f'http://www.ssg.com/search.ssg?target=all&query=nintendo&page={page_num}')
    
#     # BeautifulSoup 타입으로 변환
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # '.csrch_tip'클래스가 없을때만 HTML코드 담기
#     if len(soup.select('.csrch_tip')) == 0:
#         pages.append(soup)
#         print(f'{page_num}번째 페이지 가져오기 완료')
#         page_num += 1
#         time.sleep(3)
#     else:
#         break

# # 가져온 페이지 개수 출력
# print(f'총 {len(pages)}페이지')
print()


# TV 시청률 크롤링3
print('TV 시청률 크롤링3')
# 티비랭킹닷컴 사이트를 다시 크롤링해보려 한다.
# 앞선 과제 TV 시청률 크롤링 pt. 2에서는 모든 달에 5주차가 있다고 가정하여 받아왔다.
# 이번에는 파싱을 활용해서 실제로 데이터가 있는 페이지만 받아오려고 하는데
# 2010년 1월부터 2018년 12월까지 모든 달에 대해, 
# 데이터가 있는 모든 페이지의 HTML 코드(response의 text)를 rating_pages에 저장해 보세요.

# 주의: BeautifulSoup 타입으로 변환한 코드가 아닌, response의 text를 리스트에 저장

# year = 2010
# month = 3
# week_idx = 3

# response = requests.get(f'https://workey.codeit.kr/ratings/index?year={year}&month={month}&weekIndex={week_idx}')
# soup = BeautifulSoup(response.text, 'html.parser')
# len(soup.select('.rank'))

# rating_pages = []
# for i in range(2010, 2019):
#     for j in range(1, 13):
#         for z in range(0, 5):
#             response = requests.get(f'https://workey.codeit.kr/ratings/index?year={i}&month={j}&weekIndex={z}')
#             soup = BeautifulSoup(response.text, 'html.parser')
#             if len(soup.select('.rank')) != 1:
#                 rating_pages.append(response.text)
            
                
# print(len(rating_pages))
# print(rating_pages[0])
# print()


# 웹 페이지를 DataFrame으로
print('웹 페이지를 DataFrame으로')

# SSG의 검색 결과를, DataFrame을 만드는 방법

# 어떻게 하면 DataFrame을 만들 수 있을까?  
# DataFrame을 만드는 방법에는 여러 가지가 있고, 그중 하나는 리스트를 담은 리스트가 있다.  
# 기억이 잘 안 나시는 분은 이전 레슨을 참고
# 우리는 웹 페이지에서 상품의 정보를 파싱한 뒤, 리스트를 담은 리스트 형태로 저장할 것이다.

# DataFrame 설계하기

# 먼저 하나의 레코드(row)에 대한 설계를 한다.  
# column이 "이름", "가격", "이미지 주소" 총 세 개니까, 다음과 같은 형태로 만든다.
# ["이름 1", "가격 1", "이미지 주소 1"]
# 그리고 이 레코드가 여러 개 모이면, DataFrame을 만들 수 있다.  
# 우리가 결국 원하는 형태는 이런 형태다.
# [["이름 1", "가격 1", "이미지 주소 1"], 
# ["이름 2", "가격 2", "이미지 주소 2"], 
# ["이름 3", "가격 3", "이미지 주소 3"]]

# 파싱하기

# 이제 방법을 알았으니, 데이터를 파싱해서 DataFrame으로 만들어 보자.
# # 빈 리스트 생성
# records = []

# # 시작 페이지 지정
# page_num = 1

# while True:
#     # HTML 코드 받아오기
#     response = requests.get("http://www.ssg.com/search.ssg?target=all&query=nintendo&page=" + str(page_num))

#     # BeautifulSoup 타입으로 변형하기
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # "prodName" 클래스가 있을 때만 상품 정보 가져오기
#     if len(soup.select('.csrch_tip')) == 0:
#         product_names = soup.select('.cunit_info > div.cunit_md.notranslate > div > a > em.tx_ko')
#         product_prices = soup.select('.cunit_info > div.cunit_price.notranslate > div.opt_price > em.ssg_price')
#         product_urls = soup.select('.cunit_prod > div.thmb > a > img')
#         print(f'{page_num}페이지')
#         page_num += 1
#         time.sleep(3)
        
#         # 상품의 정보를 하나의 레코드로 만들고, 리스트에 순서대로 추가하기
#         for i in range(len(product_names)):
#             record = []
#             record.append(product_names[i].text)
#             record.append(product_prices[i].text.strip())
#             record.append("https://www.ssg.com" + product_urls[i].get('src'))
#             records.append(record)
#     else:
#         break
    
# print(records)

# DataFrame 만들기
# 이제 DataFrame 형태로 만들어주면 된다.
# df = pd.DataFrame(data = records, columns = ['이름', '가격', '이미지 주소'])

# DataFrame 출력
# print(df)
print()


# TV 시청률 크롤링 최종 프로젝트
print('TV 시청률 크롤링 최종 프로젝트')
# 티비랭킹닷컴의 데이터를 DataFrame으로 만들어서 분석해보려 한다.
# period, rank, channel, program, rating 컬럼을 가지는 DataFrame을 만들어라.

# response = requests.get('https://workey.codeit.kr/ratings/index?year=2010&month=1&weekIndex=0')
# soup = BeautifulSoup(response.text, 'html.parser')

# # period추출
# soup.select('#weekSelectBox > option')[0].text

# # rank추출
# soup.select('tr.row > td.rank')[0]

# # channel추출
# soup.select('tr.row > td.channel')[0]

# # program추출
# soup.select('tr.row > td.program')

# # rating(시청률)추출
# soup.select('tr.row > td.percent')


# ch_lists = []
# for i in range(2010, 2019):
#     for j in range(1, 13):
#         for z in range(0, 5):
#             response = requests.get(f'https://workey.codeit.kr/ratings/index?year={i}&month={j}&weekIndex={z}')
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             if len(soup.select('tr.row')) > 1:
#                 for ch in range(len(soup.select('tr.row')) - 1):
#                     ch_list = []
#                     ch_list.append(soup.select('#weekSelectBox > option')[z].text)
#                     ch_list.append(soup.select('tr.row > td.rank')[ch].text)
#                     ch_list.append(soup.select('tr.row > td.channel')[ch].text)
#                     ch_list.append(soup.select('tr.row > td.program')[ch].text)
#                     ch_list.append(soup.select('tr.row > td.percent')[ch].text)
#                     ch_lists.append(ch_list)
        

# df = pd.DataFrame(data = ch_lists, columns = ['period', 'rank', 'channel', 'program', 'rating'])
# print(df)
# print()
# =============================================================================
