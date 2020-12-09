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

# DataFrame에 값 쓰기1
print('DataFrame에 값 쓰기2')
iphone_df6 = pd.read_csv('iPhone.csv', index_col = 0)
print(iphone_df6)
iphone_df6.loc['iPhone 8', '메모리'] = '2.5GB' # 메모리 2GB -> 2.5GB로 변경
print(iphone_df6)
iphone_df6.loc['iPhone 8', '출시 버전'] = 'iOS 10.3' # 출시버전 iOS 11.0 -> iOS 10.3으로 변경
print(iphone_df6)
iphone_df6.loc['iPhone 8'] = ['2016-09-22', '4.7', '2GB', 'iOS 11.0', 'No']
# iPhone 8 행 데이터 전부 변경
print(iphone_df6)
iphone_df6['디스플레이'] = ['4.7 in', '5.5 in', '4.7 in', '5.5 in', '5.8 in', '5.8 in', '6.5 in']
# 디스플레이 컬럼 데이터 전부 변경
print(iphone_df6)
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

enrolment_df['status'] = np.where(cnt_5, 'not allowed', np.where(cnt_1 | cnt_4, 'not allowed', 'allowed'))
enrolment_df
