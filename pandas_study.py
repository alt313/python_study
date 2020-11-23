# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:37:36 2020

@author: msi
"""

# pandas 호출 및 버전 확인
# =============================================================================

# pandas 호출
print('pandas와 numpy 호출')
import numpy as np
import pandas as pd
print()

# pandas 현재 버전 출력
print('pandas 현재 버전 출력')
print(pd.__version__)
print()
# =============================================================================




# pandas 객체
# =============================================================================

# Series 객체
print('Series 객체')
s = pd.Series([0, 0.25, 0.5, 0.75, 1.0])
print(s)
print(s.values) # 값만 출력
print(s.index)  # 인덱스 출력
print(s[1])
print(s[1:4])
print()

s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = ['a', 'b', 'c', 'd', 'e'])
print(s)
print(s['c'])
print(s[['c', 'd', 'e']])
print('b' in s)
print()

s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = [2, 4, 6, 8, 10])
print(s)
print(s[4])
print(s[2:])
print(s.unique()) # Series에서 유니크한 값만 출력해준다
print(s.value_counts()) # Series에서 각 값들의 갯수 출력
print(s.isin([0.25, 0.75])) # 값이 있으면 True, 없으면 False로 알려준다
print()

# 인구수
print('인구수')
pop_tuple = {'서울특별시' : 9720846,
             '부산광역시' : 3404423,
             '인천광역시' : 2947217,
             '대구광역시' : 2427954,
             '대전광역시' : 1471040,
             '광주광역시' : 1455048}
print(pop_tuple)
population = pd.Series(pop_tuple)
print(population)
print(population['서울특별시':'인천광역시'])
print()

# DataFrame 객체
print('DataFrame 객체')
print(pd.DataFrame([{'A' : 2, 'B' : 4, 'D' : 3},
                    {'A' : 4, 'B' : 5, 'C' : 7}]))
print(pd.DataFrame(np.random.rand(5, 5),
                   columns = ['A', 'B', 'C', 'D', 'E'],
                   index = [1, 2, 3, 4, 5]))
print()

# 남자인구수
print('남자 인구수')
male_tuple = {'서울특별시' : 4732275,
              '부산광역시' : 1668618,
              '인천광역시' : 1476813,
              '대구광역시' : 1198815,
              '대전광역시' : 734441,
              '광주광역시' : 720060}
male = pd.Series(male_tuple)
print(male)
print()

# 여자인구수
print('여자 인구수')
female_tuple = {'서울특별시' : 4988571,
                '부산광역시' : 1735805,
                '인천광역시' : 1470404,
                '대구광역시' : 1229139,
                '대전광역시' : 736599,
                '광주광역시' : 734988}
female = pd.Series(female_tuple)
print(female)
print()

# 인구수, 남자인구수, 여자인구수 하나의 DataFrame으로 만들기
print('인구수, 남자인구수, 여자인구수, 하나의 DataFrame으로 만들기')
korea_df = pd.DataFrame({'인구수': population,
                         '남자인구수': male,
                         '여자인구수': female})
print(korea_df)
print(korea_df.index)
print(korea_df.columns)
print(korea_df['여자인구수'])
print(korea_df['서울특별시':'인천광역시'])
print()

# Index 객체

# Index : 일반적인 index객체이며, numpy배열 형식으로 축의 이름 표현
# Int64Index : 정수 값을 위한 index
# MultiIndex : 단일 축에 여러 단계 색인을 표현하는 계층적 index객체
# DatetimeIndex : numpy의 datetime64타입으로 타임스탬프 저장
# PeriodIndex : 기간 데이터를 위한 index
idx = pd.Index([2, 4, 6, 8, 10])
print(idx)
print(idx[1])
print(idx[1:2:2])
print(idx[-1::])
print(idx[::2])
print(idx.size)
print(idx.shape)
print(idx.ndim)
print(idx.dtype)
print()

# Index 연산

print('Index 연산')
idx1 = pd.Index([1, 2, 4, 6, 8])
idx2 = pd.Index([2, 4, 5, 6, 7])
print(idx1)
print(idx2)
print()

# append : 색인 객체를 추가한 새로운 색인 반환
print('append : 색인 객체를 추가한 새로운 색인 반환')
print(idx1.append(idx2))
print()


# difference : 색인의 차집합 반환
print('difference : 색인의 차집합 반환')
print(idx1.difference(idx2))
print()

# intersection : 색인의 교집합 반환
print('intersection : 색인의 교집합 반환')
print(idx1.intersection(idx2))
print(idx1 & idx2)
print()

# union : 색인의 합집합 반환
print('union : 색인의 합집합 반환')
print(idx1.union(idx2))
print(idx1 | idx2)
print()

# delete : 색인이 삭제된 새로운 색인 반환
print('delete : 색인이 삭제된 새로운 색인 반환')
print(idx1.delete(0))
print()

# drop : 값이 삭제된 새로운 색인 반환
print('drop : 값이 삭제된 새로운 색인 반환')
print(idx1.drop(1))
print()

# insert : 색인이 추가된 새로운 색인 반환
# is_monotonic : 색인이 단조정을 가지면 True
# is_union : 중복되는 색인이 없다면 False
# unique : 색인에서 중복되는 요소 제거하고 유일한 값만 반환
# =============================================================================





# 인덱싱(Indexing)
# =============================================================================

# 인덱싱
print('인덱싱')
s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = ['a', 'b', 'c', 'd', 'e'])
print(s['b'])
print('b' in s)
print(s.keys())
print(list(s.items()))
s['f'] = 1.25
print(s)
print(s[0:4])
print(s[(s > 0.4) & (s < 0.8)])
print()

# Series 인덱싱
print('Series 인덱싱')
s = pd.Series(['a', 'b', 'c', 'd', 'e'],
              index = [1, 3, 5, 7, 9])
print(s)
print(s[1])
print(s[2:4])
print()

# iloc : 인덱싱을 숫자로 한다.
print(s.iloc[1])
print(s.iloc[2:4])
print()

# reindex : 인덱스를 재구성
print(s.reindex(range(10)))
print(s.reindex(range(10), method = 'bfill')) # 'bfill' : 비어있다면 전값으로 채움
print()

# DataFrame 인덱싱

# df[ ] or df.~ : 일반 색인
print('df[ ] or df.~ : 일반 색인')
print(korea_df['남자인구수'])
print(korea_df.남자인구수)
print(korea_df.여자인구수)
korea_df['남여비율'] = (korea_df['남자인구수'] * 100 / korea_df['여자인구수'])
print(korea_df.남여비율)
print(korea_df.values)
print(korea_df.T)
print(korea_df.values[0])
print(korea_df['인구수'])
print()

# df.loc : 라벨값으로 색인
# df.loc[로우, 컬럼]
print('df.loc : 라벨값으로 색인')
print(korea_df.loc[:'인천광역시', :'남자인구수'])
print(korea_df.loc[(korea_df.여자인구수 > 1000000)])
print(korea_df.loc[(korea_df.인구수 < 2000000)])
print(korea_df.loc[(korea_df.인구수 > 2500000)])
print(korea_df.loc[korea_df.남여비율 > 100])
print(korea_df.loc[(korea_df.인구수 > 2500000) & (korea_df.남여비율 > 100)])
print()

# df.iloc : 정수로 색인
# df.iloc[로우, 컬럼]
print('df.iloc : 정수로 색인')
print(korea_df.iloc[:3, :2])
print()

# 다중인덱싱

# 다중 인덱스 Series
print('다중 인덱스 Series')
idx_tuples = [('서울특별시', 2010), ('서울특별시', 2020),
              ('부산광역시', 2010), ('부산광역시', 2020),
              ('인천광역시', 2010), ('인천광역시', 2020),
              ('대구광역시', 2010), ('대구광역시', 2020),
              ('대전광역시', 2010), ('대전광역시', 2020),
              ('광주광역시', 2010), ('광주광역시', 2020)]
print(idx_tuples)
pop_tuples = [10312545, 9720846,
              2567910, 3404423,
              2758296, 2947217,
              2511676, 2427954,
              1503664, 1471040,
              1454636, 1455048]
print(pop_tuples)
population = pd.Series(pop_tuples, index = idx_tuples)
print(population)
print()

midx = pd.MultiIndex.from_tuples(idx_tuples)
print(midx)
population = population.reindex(midx)
print(population)
print(population[:, 2020])
print(population['대전광역시', :])
print()

korea_mdf = population.unstack()
print(korea_mdf)
print(korea_mdf.stack())
print()

male_tuples = [5111259, 4732275,
              1773170, 1668618,
              1390356, 1476813,
              1255245, 1198815,
              753648, 734441,
              721780, 720060]
print(male_tuples)
print()

female_tuples = [5201286, 4988571,
                 1794740, 1735805,
                 1367940, 1470404,
                 1256431, 1229139,
                 750016, 736599,
                 732856, 734988]
print(female_tuples)
print()

korea_mdf = pd.DataFrame({'총인구수' : population,
                          '남자인구수' : male_tuples,
                          '여자인구수' : female_tuples})
print(korea_mdf)
print()

ratio = korea_mdf['남자인구수'] * 100 / korea_mdf['여자인구수']
print(ratio)
print(ratio.unstack())
print()

korea_mdf = pd.DataFrame({'총인구수' : population,
                          '남자인구수' : male_tuples,
                          '여자인구수' : female_tuples,
                          '남여비율' : ratio})
print(korea_mdf)
print()

# 다중 인덱스 생성

df = pd.DataFrame(np.random.rand(6, 3),
                  index = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]],
                  columns = ['c1', 'c2', 'c3'])
print(df)
print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]]))
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]))
print(pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]]))
print(pd.MultiIndex(levels = [['a', 'b', 'c'], [1, 2]],
                    codes = [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]))
# =============================================================================
