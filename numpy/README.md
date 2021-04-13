# 👀 Numpy Cheatsheet

- [1. Numpy Package](#1-numpy-package)
- [2. Array Creation](#2-array-creation)
  - [2.1. dtype](#21-dtype)
  - [2.2. shape](#22-shape)
  - [2.3. ndim](#23-ndim)
  - [2.4. nbytes](#24-nbytes)
- [3. Handling Shape](#3-handling-shape)
  - [3.1. reshape](#31-reshape)
  - [3.2. flatten](#32-flatten)
- [4. Indexing & Slicing](#4-indexing--slicing)
  - [4.1. Indexing](#41-indexing)
  - [4.2. Slicing](#42-slicing)
- [5. Creation Function](#5-creation-function)
  - [5.1. arange](#51-arange)
  - [5.2. zeros](#52-zeros)
  - [5.3. ones](#53-ones)
  - [5.3. empty](#53-empty)
  - [5.4. zeroslike, oneslike, emptylike](#54-zeroslike-oneslike-emptylike)
  - [5.5. identity](#55-identity)
  - [5.6. eye](#56-eye)
  - [5.7. Random Sampling](#57-random-sampling)
- [6. Operation Functions](#6-operation-functions)
  - [6.1. sum](#61-sum)
  - [6.2. mean](#62-mean)
  - [6.3. std](#63-std)
  - [6.3. concatenate](#63-concatenate)
- [7. Array Operations](#7-array-operations)
  - [7.1. Element-wise Operations](#71-element-wise-operations)
  - [7.2. Dot Product](#72-dot-product)
  - [7.3. Transpose](#73-transpose)
  - [7.4. Broadcasting](#74-broadcasting)
- [8. Comparisons](#8-comparisons)
  - [8.1. All, Any](#81-all-any)
  - [8.2. logicaland, logicalnot, logicalor](#82-logicaland-logicalnot-logicalor)
  - [8.3. np.where](#83-npwhere)
  - [8.4. np.isnan, np.isfinite](#84-npisnan-npisfinite)
  - [8.4. argmax, argmin](#84-argmax-argmin)
  - [8.5. argsort](#85-argsort)
- [9. Boolean & Fancy Index](#9-boolean--fancy-index)
  - [9.1. Boolean Index](#91-boolean-index)
  - [9.2. Fancy Index](#92-fancy-index)
    - [1 차원 배열](#1-차원-배열)
    - [2 차원 배열](#2-차원-배열)
- [10. Numpy Data I/O](#10-numpy-data-io)
  - [10.1. csv](#101-csv)
  - [10.2. npy](#102-npy)

## 1. Numpy Package

```python
import numpy as np
```

Numerical Python의 줄임말로 파이썬 고성능 과학 계산용 패키지이다.

- 파이썬의 리스트보다 빠르고 메모리 효율적이다.
- 선형대수와 관련된 다양한 기능을 제공한다.
- C, C++, Fortran 등의 언어와 통합이 가능하다.

## 2. Array Creation

`np.array`를 사용하여 배열을 생성할 수 있다.

- 타입은 `ndarray`형이다.
- 두 번째 인수로 요소의 자료형 `dtype`을 결정할 수 있다.
- 여러 자료형의 요소를 가질 수 있는 리스트와는 달리 모두 자료형이 같은 요소를 갖는다.

```python
#  np.array()로 ndarray 생성
arr = np.array([1, 2, 3, 4], dtype=np.float64)
arr
```

    array([1., 2., 3., 4.])

```python
# 중간에 문자열로 된 숫자가 있어도 OK
arr = np.array([1, 2, "3", 4], dtype=np.float64)
arr
```

    array([1., 2., 3., 4.])

### 2.1. dtype

numpy array의 **데이터 자료형**을 반환한다.

```python
# 요소의 자료형을 반환
arr.dtype
```

    dtype('float64')

### 2.2. shape

numpy array의 **dimension 구성** 즉, array의 크기, 형태 등에 관한 정보를 반환한다.

```python
# 1차원 배열
arr = np.array([1, 2, 3, 4], dtype=np.float64)
arr.shape
```

    (4,)

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int16)
arr_2d.shape
```

    (2, 4)

```python
# 3차원 배열
arr_3d = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                   [[5, 6, 7, 8], [5, 6, 7, 8]]])
arr_3d.shape
```

    (2, 2, 4)

### 2.3. ndim

**dimension의 개수**(=rank)를 반환한다.

```python
# 1차원 배열
arr_1d = np.array([1, 2, 3, 4], dtype=np.float64)
arr_1d.ndim
```

    1

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int16)
arr_2d.ndim
```

    2

```python
# 3차원 배열
arr_3d = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                   [[5, 6, 7, 8], [5, 6, 7, 8]]])
arr_3d.ndim
```

    3

### 2.4. nbytes

ndarray object의 **메모리 크기**를 반환한다.

- `np.float32`: 6 \* 4 = 24 bytes

```python
# float32
np.array([[1, 2, 3],
          [4, 5, 6]], dtype=np.float32).nbytes
```

    24

- `np.int8`: 6 \* 1 = 6 bytes

```python
# int8
np.array([[1, 2, 3],
          [4, 5, 6]], dtype=np.int8).nbytes
```

    6

- `np.float64`: 6 \* 8 = 48 bytes

```python
# float64
np.array([[1, 2, 3],
          [4, 5, 6]], dtype=np.float64).nbytes
```

    48

## 3. Handling Shape

### 3.1. reshape

array의 shape의 크기를 변경한다. 단, **element의 개수는 동일**해야 한다.

```python
# (2, 4) -> (1, 8)
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int8)
arr_2d.reshape((1, 8))
```

    array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=int8)

`reshape`의 인자로 넣어주는 튜플로 된 크기의 요소에 -1을 넣으면 알아서 해당 축의 개수를 계산한다.

```python
# -1을 넣어주면 알아서 계산
# (2, 4) -> (4, 2)
arr_2d.reshape((-1, 2))
```

    array([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8]], dtype=int8)

```python
# (2, 4) -> (2, 2, 2)
arr_2d.reshape((2, 2, -1))
```

    array([[[1, 2],
            [3, 4]],

           [[5, 6],
            [7, 8]]], dtype=int8)

### 3.2. flatten

다차원 array를 **1차원 array**로 변환한다.

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int8)
arr_2d.flatten()
```

    array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int8)

## 4. Indexing & Slicing

### 4.1. Indexing

`[0, 0]` 혹은 `[0][0]` 표기법 중 편할 것으로 사용하자.

```python
# 2차원 배열 인덱싱
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int8)
print(arr_2d[0, 0])
print(arr_2d[0][0])
```

    1
    1

```python
# 특정 인덱스에 값 할당도 가능
arr_2d[0][0] = 10
arr_2d
```

    array([[10,  2,  3,  4],
           [ 5,  6,  7,  8]], dtype=int8)

### 4.2. Slicing

**부분 집합**을 추출할 때 사용한다.

```python
# 2차원 배열 슬라이싱
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]], dtype=np.int8)

print(arr_2d[:, 2:])    # 전체 행의 2열 이상
print(arr_2d[1, 1:3])   # 1행의 1, 2열
print(arr_2d[1:3])      # 1, 2행 전체 열
```

    [[3 4]
     [7 8]]
    [6 7]
    [[5 6 7 8]]

`시작:끝:스텝`으로 스텝 지정이 가능하다.

```python
# 열에 대해 step=2로 준 경우
arr_2d[:, ::2]
```

    array([[1, 3],
           [5, 7]], dtype=int8)

`[]`이 인덱싱이냐 슬라이싱이냐에 따라서 같은 요소라도 형태가 다르다.

- `[0]`: 1개만 지정하므로 1차원
- `[0:1]`: 여러개를 지정하므로 1차원 (결과가 한 줄만 나올 뿐)

```python
arr_2d[1]        # 1차원 배열 반환
```

    array([5, 6, 7, 8], dtype=int8)

```python
arr_2d[1:2]    # 2차원 배열 반환
```

    array([[5, 6, 7, 8]], dtype=int8)

## 5. Creation Function

### 5.1. arange

`arange(시작, 끝, 스텝)`으로 시작부터 끝-1까지 스텝만큼 간격이 벌어진 1차원 배열을 반환한다.

```python
# 시작: 0, 끝: 29, 스텝: 1
np.arange(30)
```

    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

```python
# 시작: 0, 끝: 5, 스텝: 0.5
np.arange(0, 5, 0.5)
```

    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

### 5.2. zeros

인수로 받은 **배열의 크기만큼의 모두 0으로 초기화된 배열**을 반환한다.

```python
# 2차원 배열
np.zeros(shape=(2, 4), dtype=np.int8)
```

    array([[0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

```python
# 3차원 배열
np.zeros(shape=(3, 2, 1), dtype=np.int8)
```

    array([[[0],
            [0]],

           [[0],
            [0]],

           [[0],
            [0]]], dtype=int8)

### 5.3. ones

인수로 받은 **배열의 크기만큼의 모두 1로 초기화된 배열**을 반환한다.

```python
# 2차원 배열
np.ones(shape=(2, 4), dtype=np.int8)
```

    array([[1, 1, 1, 1],
           [1, 1, 1, 1]], dtype=int8)

```python
# 3차원 배열
np.ones(shape=(3, 2, 1), dtype=np.int8)
```

    array([[[1],
            [1]],

           [[1],
            [1]],

           [[1],
            [1]]], dtype=int8)

### 5.3. empty

인수로 받은 **배열의 크기만큼의 빈 배열**을 반환한다. 요소의 값은 메모리의 쓰레기값으로 채워진다.

```python
# 2차원 배열
np.empty(shape=(2, 4), dtype=np.int8)
```

    array([[ 40, 110,  10, 110],
           [101, 105, 116, 104]], dtype=int8)

```python
# 3차원 배열
np.empty(shape=(3, 2, 1), dtype=np.int8)
```

    array([[[ 40],
            [114]],

           [[111],
            [ 97]],

           [[100],
            [ 99]]], dtype=int8)

### 5.4. zeros_like, ones_like, empty_like

인수로 받은 **배열의 크기**만큼의 1, 0으로 채워진 혹은 빈 배열을 반환한다.

```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=np.int8)

print(np.zeros_like(arr_2d))    # 모두 0으로 초기화
print(np.ones_like(arr_2d))     # 모두 1로 초기화
print(np.empty_like(arr_2d))    # 쓰레기값으로 초기화
```

    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    [[1 1 1]
     [1 1 1]
     [1 1 1]]
    [[1 1 1]
     [1 1 1]
     [1 1 1]]

### 5.5. identity

$N×N$ 크기의 **단위 행렬**을 생성한다.

```python
# 크기 3의 단위행렬
np.identity(n=3, dtype=np.int8)
```

    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int8)

```python
# 크기 5의 단위행렬
np.identity(n=5)
```

    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])

### 5.6. eye

`np.identity`와는 달리 $N×N$ 크기가 아니어도 되며, `k`의 값으로 시작 인덱스를 설정할 수 있다.

- $k > 0$: 주 대각선(`k=0`)보다 k만큼 위에 위치한다.
- $k < 0$: 주 대각선(`k=0`)보다 k만큼 아래 위치한다.

```python
# np.identity(3)과 동일
np.eye(3)
```

    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

```python
# (3, 5) 크기의 대각선이 (0, 1)부터 시작
np.eye(N=3, M=5, k=1, dtype=np.int8)
```

    array([[0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0]], dtype=int8)

```python
# (3, 5) 크기의 대각선이 (0, 4)부터 시작
np.eye(N=3, M=5, k=4, dtype=np.int8)
```

    array([[0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=int8)

```python
# (3, 5) 크기의 대각선이 (1, 0)부터 시작
np.eye(N=3, M=5, k=-1, dtype=np.int8)
```

    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0]], dtype=int8)

### 5.7. Random Sampling

다양한 기준으로 랜덤 값을 생성할 수 있다. 보통 `(시작, 끝, 모수)`로 인자를 받는다.

```python
# 균등 분포
np.random.uniform(0, 1, 10)
```

    array([0.68964525, 0.14090274, 0.45853219, 0.61122881, 0.41077504,
           0.94554117, 0.33724925, 0.21587557, 0.68680998, 0.78315665])

```python
# 정규 분포
np.random.normal(0, 1, 10)
```

    array([-0.75317104, -0.95131911,  0.06679911,  1.83637756, -0.07257576,
            0.25116002, -0.69903599, -0.40847364,  0.57707482,  0.47669597])

## 6. Operation Functions

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=np.int8)
arr_2d.shape
```

    (2, 3)

```python
# 3차원 배열
arr_3d = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                   [[5, 6, 7, 8], [5, 6, 7, 8]]])
arr_3d.shape
```

    (2, 2, 4)

### 6.1. sum

#### 2차원 배열

```python
# 모든 요소의 합
arr_2d.sum(dtype=np.int8)
```

    21

```python
# axis=0 방향으로의 합 -> axis=0이 없어진다!
arr_2d.sum(axis=0)
```

    array([5, 7, 9])

```python
# axis=1 방향으로의 합 -> axis=1이 없어진다!
arr_2d.sum(axis=1)
```

    array([ 6, 15])

#### 3차원 배열

```python
# 모든 요소의 합
arr_3d.sum()
```

    72

```python
# axis=0 방향으로의 합 -> axis=0이 없어진다!
arr_3d.sum(axis=0)
```

    array([[ 6,  8, 10, 12],
           [ 6,  8, 10, 12]])

```python
# axis=1 방향으로의 합 -> axis=1이 없어진다!
arr_3d.sum(axis=1)
```

    array([[ 2,  4,  6,  8],
           [10, 12, 14, 16]])

```python
# axis=2 방향으로의 합 -> axis=2가 없어진다!
arr_3d.sum(axis=2)
```

    array([[10, 10],
           [26, 26]])

### 6.2. mean

**요소 간의 평균**을 반환한다.

```python
# 모든 요소의 평균
arr_2d.mean()
```

    3.5

```python
# axis=0 방향으로의 평균 -> axis=0이 없어진다!
arr_2d.mean(axis=0)
```

    array([2.5, 3.5, 4.5])

```python
# axis=1 방향으로의 평균 -> axis=1이 없어진다!
arr_2d.mean(axis=1)
```

    array([2., 5.])

### 6.3. std

**요소 간의 표준편차**를 반환한다.

```python
# 모든 요소의 표준편차
arr_2d.std()
```

    1.707825127659933

```python
# axis=0 방향으로의 표준편차 -> axis=0이 없어진다!
arr_2d.std(axis=0)
```

    array([1.5, 1.5, 1.5])

```python
# axis=1 방향으로의 표준편차 -> axis=1이 없어진다!
arr_2d.std(axis=1)
```

    array([0.81649658, 0.81649658])

### 6.3. concatenate

배열을 지정한 방향에 따라 합쳐준다.

#### vstack

axis=0 방향으로 배열을 붙여준다.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a, b))
```

    array([[1, 2, 3],
           [4, 5, 6]])

#### hstack

axis=1 방향으로 배열을 붙여준다.

```python
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
np.hstack((a, b))
```

    array([[1, 4],
           [2, 5],
           [3, 6]])

#### concatenate

axis를 지정해 원하는 축에서의 concat을 해준다.

```python
# concatenate(axis=0) == vstack
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])
np.concatenate((a, b), axis=0)
```

    array([[1, 2, 3],
           [4, 5, 6]])

```python
# concatenate(axis=1) == hstack
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
np.concatenate((a, b), axis=1)
```

    array([[1, 4],
           [2, 5],
           [3, 6]])

단, 인수로 받는 배열의 축 범위 내의 concat 결과만 나온다.

- (3, )와 (3, ) 벡터끼리의 axis=0 방향으로 concat을 하면, **1차원 벡터**가 반환된다.

```python
# a와 b의 크기가 (3, )
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate((a, b), axis=0)
```

    array([1, 2, 3, 4, 5, 6])

- (1, 3)과 (1, 3) 벡터끼리의 axis=0 방향으로 concat을 하면, **2차원 벡터**가 반환된다.

```python
# a와 b의 크기가 (1, 3)
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])
np.concatenate((a, b), axis=0)
```

    array([[1, 2, 3],
           [4, 5, 6]])

## 7. Array Operations

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=np.int8)
arr_2d.shape
```

    (2, 3)

```python
# 3차원 배열
arr_3d = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                   [[5, 6, 7, 8], [5, 6, 7, 8]]])
arr_3d.shape
```

    (2, 2, 4)

### 7.1. Element-wise Operations

대응되는 요소끼리의 사칙연산으로, 피연산자인 배열의 크기가 같아야 한다.

#### 덧셈

```python
arr_2d + arr_2d
```

    array([[ 2,  4,  6],
           [ 8, 10, 12]], dtype=int8)

```python
arr_3d + arr_3d
```

    array([[[ 2,  4,  6,  8],
            [ 2,  4,  6,  8]],

           [[10, 12, 14, 16],
            [10, 12, 14, 16]]])

#### 뺄셈

```python
arr_2d - arr_2d
```

    array([[0, 0, 0],
           [0, 0, 0]], dtype=int8)

```python
arr_3d - arr_3d
```

    array([[[0, 0, 0, 0],
            [0, 0, 0, 0]],

           [[0, 0, 0, 0],
            [0, 0, 0, 0]]])

#### 곱셈

```python
arr_2d * arr_2d
```

    array([[ 1,  4,  9],
           [16, 25, 36]], dtype=int8)

```python
arr_3d * arr_3d
```

    array([[[ 1,  4,  9, 16],
            [ 1,  4,  9, 16]],

           [[25, 36, 49, 64],
            [25, 36, 49, 64]]])

### 7.2. Dot Product

**두 배열의 내적**을 한 결과를 반환한다.

```python
arr1 = np.arange(10).reshape(2, 5)
arr2 = np.arange(11, 21).reshape(5, 2)
arr1.dot(arr2)
```

    array([[170, 180],
           [545, 580]])

### 7.3. Transpose

배열의 **전치 행렬**을 반환한다.

```python
arr = np.arange(10).reshape(2, 5)
arr
```

    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

```python
# arr의 전치 행렬
print(np.transpose(arr))
print(arr.T)
```

    [[0 5]
     [1 6]
     [2 7]
     [3 8]
     [4 9]]
    [[0 5]
     [1 6]
     [2 7]
     [3 8]
     [4 9]]

### 7.4. Broadcasting

크기가 다른 배열 간의 연산을 지원한다. 비는 공간만큼 같은 값, 행렬로 채운다.

```python
matrix = np.arange(10).reshape(2, 5)
matrix
```

    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

```python
vector = np.arange(5)
vector
```

    array([0, 1, 2, 3, 4])

#### 행렬 - 스칼라

```python
matrix + 3
```

    array([[ 3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12]])

```python
matrix * 3
```

    array([[ 0,  3,  6,  9, 12],
           [15, 18, 21, 24, 27]])

```python
matrix - 3
```

    array([[-3, -2, -1,  0,  1],
           [ 2,  3,  4,  5,  6]])

```python
matrix ** 3
```

    array([[  0,   1,   8,  27,  64],
           [125, 216, 343, 512, 729]], dtype=int32)

#### 행렬 - 벡터

```python
matrix + vector
```

    array([[ 0,  2,  4,  6,  8],
           [ 5,  7,  9, 11, 13]])

```python
matrix - vector
```

    array([[0, 0, 0, 0, 0],
           [5, 5, 5, 5, 5]])

```python
matrix * vector
```

    array([[ 0,  1,  4,  9, 16],
           [ 0,  6, 14, 24, 36]])

## 8. Comparisons

```python
arr = np.arange(10)
arr
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

### 8.1. All, Any

배열의 데이터 전부 혹은 일부가 조건에 만족하는지를 반환한다.

```python
# all -> 요소가 모든 조건을 만족하면 True
print("arr > -1 :", np.all(arr > -1))
print("arr > 5 :", np.all(arr > 5))
```

    arr > -1 : True
    arr > 5 : False

```python
# any -> 요소 중 하나라도 조건을 만족하면 True
print("arr > 5:", np.any(arr > 5))
print("arr < 0:", np.any(arr < 0))
```

    arr > 5: True
    arr < 0: False

### 8.2. logical_and, logical_not, logical_or

Bool로 이루어진 배열에 `and`, `or`, `not` 연산을 수행한다. 단, `logical_not`을 제외하고 두 배열의 크기는 같아야 한다.

```python
# logical_and: element-wise and
print("arr > 0:", arr > 0)
print("arr < 5:", arr > 5)
np.logical_and(arr > 0, arr > 5)
```

    arr > 0: [False  True  True  True  True  True  True  True  True  True]
    arr < 5: [False False False False False False  True  True  True  True]





    array([False, False, False, False, False, False,  True,  True,  True,
            True])

```python
# logical_or: element-wise or
print("arr > 0:", arr > 0)
print("arr < 5:", arr > 5)
np.logical_or(arr > 0, arr > 5)
```

    arr > 0: [False  True  True  True  True  True  True  True  True  True]
    arr < 5: [False False False False False False  True  True  True  True]





    array([False,  True,  True,  True,  True,  True,  True,  True,  True,
            True])

```python
# logical_not: Bool 요소를 반전
print("arr < 5:", arr > 5)
np.logical_not(arr > 5)
```

    arr < 5: [False False False False False False  True  True  True  True]





    array([ True,  True,  True,  True,  True,  True, False, False, False,
           False])

### 8.3. np.where

- `방법 1.` True인 요소에 특정 값 x, False인 요소에 특정 값 y를 할당한다.

```python
print("arr > 5:", arr > 5)
np.where(arr < 5, 10, 1)
```

    arr > 5: [False False False False False False  True  True  True  True]





    array([10, 10, 10, 10, 10,  1,  1,  1,  1,  1])

- `방법 2.` True인 요소의 인덱스 값을 반환한다.

```python
print("arr%2 != 0:", arr%2 != 0)
np.where(arr%2 != 0)
```

    arr%2 != 0: [False  True False  True False  True False  True False  True]





    (array([1, 3, 5, 7, 9], dtype=int64),)

### 8.4. np.isnan, np.isfinite

```python
arr = np.array([1, np.NaN, np.inf])
arr
```

    array([ 1., nan, inf])

- `np.isnan`: 각 요소가 Nan(Not a number)인지 검사한 Bool 배열을 반환한다.

```python
np.isnan(arr)
```

    array([False,  True, False])

- `np.isfinite`: 각 요소가 유한한지(발산하지 않은지) 검사한 Bool 배열을 반환한다.

```python
np.isfinite(arr)
```

    array([ True, False, False])

### 8.4. argmax, argmin

배열 내의 **최대값 혹은 최소값의 인덱스**를 반환한다. `axis`를 지정해서 해당 방향에서의 최대/최소값의 인덱스를 구할 수 있다.

#### 1차원 배열

```python
arr_1d = np.array([1, 3, 7, 2, 0, -5, 100])
arr_1d, arr_1d.shape
```

    (array([  1,   3,   7,   2,   0,  -5, 100]), (7,))

```python
np.argmax(arr_1d), np.argmin(arr_1d)
```

    (6, 5)

#### 2차원 배열

```python
arr_2d = np.array([[1, 2, 6, 2, -85, 100],
                   [7, 9, -20, 120, 0, 12]])
arr_2d, arr_2d.shape
```

    (array([[  1,   2,   6,   2, -85, 100],
            [  7,   9, -20, 120,   0,  12]]),
     (2, 6))

```python
np.argmax(arr_2d), np.argmin(arr_2d)    # flatten됐을 때의 기준인 듯
```

    (9, 4)

```python
# axis도 지정 가능
# 지정된 axis 방향으로 최대-> 해당 axis가 없어진다!
print("argmax(axis=0) :", np.argmax(arr_2d, axis=0))
print("argmin(axis=1) :", np.argmax(arr_2d, axis=1))
```

    argmax(axis=0) : [1 1 0 1 1 0]
    argmin(axis=1) : [5 3]

#### 3차원 배열

```python
arr_3d = np.array([[[1, 2, 6], [2, -85, 100]],
                   [[7, 9, -20], [120, 0, 12]]])
arr_3d, arr_3d.shape
```

    (array([[[  1,   2,   6],
             [  2, -85, 100]],

            [[  7,   9, -20],
             [120,   0,  12]]]),
     (2, 2, 3))

```python
np.argmax(arr_3d), np.argmin(arr_3d)
```

    (9, 4)

```python
# axis도 지정 가능
# 지정된 axis 방향으로 최대-> 해당 axis가 없어진다!
print("argmax(axis=0) :\n", np.argmax(arr_3d, axis=0))
print("argmin(axis=1) :\n", np.argmax(arr_3d, axis=1))
print("argmin(axis=2) :\n", np.argmax(arr_3d, axis=2))
```

    argmax(axis=0) :
     [[1 1 0]
     [1 1 0]]
    argmin(axis=1) :
     [[1 0 1]
     [1 0 1]]
    argmin(axis=2) :
     [[2 2]
     [1 0]]

### 8.5. argsort

배열을 정렬했을 때 원 배열의 인덱스를 기준으로 한 인덱스 배열을 반환한다.

```python
# 1차원 배열
arr_1d = np.array([1, 3, 7, 2, 0, -5, 100])
arr_1d, arr_1d.shape
```

    (array([  1,   3,   7,   2,   0,  -5, 100]), (7,))

```python
np.argsort(arr_1d)
```

    array([5, 4, 0, 3, 1, 2, 6], dtype=int64)

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 6, 2, -85, 100],
                   [7, 9, -20, 120, 0, 12]])
arr_2d, arr_2d.shape
```

    (array([[  1,   2,   6,   2, -85, 100],
            [  7,   9, -20, 120,   0,  12]]),
     (2, 6))

```python
np.argsort(arr_2d)
```

    array([[4, 0, 1, 3, 2, 5],
           [2, 4, 0, 1, 5, 3]], dtype=int64)

## 9. Boolean & Fancy Index

```python
# 1차원 배열
arr_1d = np.array([1, 3, 7, 2, 0, -5, 100])
arr_1d, arr_1d.shape
```

    (array([  1,   3,   7,   2,   0,  -5, 100]), (7,))

### 9.1. Boolean Index

특정 조건에 따른 값을 배열로 추출한다.

```python
arr_1d[arr_1d > 5]
```

    array([  7, 100])

```python
arr_1d[arr_1d%2 == 0]
```

    array([  2,   0, 100])

### 9.2. Fancy Index

정수로 된 배열을 인덱스에 넣어주면 해당 인덱스의 요소만 뽑은 배열을 반환한다.

#### 1차원 배열

```python
# 1차원 배열
arr_1d = np.array([1, 3, 7, 2, 0, -5, 100])
arr_1d, arr_1d.shape
```

    (array([  1,   3,   7,   2,   0,  -5, 100]), (7,))

```python
cond = np.array([1, 3, 5])
print("방법 1:", arr_1d[cond])
print("방법 2:", arr_1d.take(cond))
```

    방법 1: [ 3  2 -5]
    방법 2: [ 3  2 -5]

take의 경우 뽑은 요소를 **어떤 형태로 나타낼건지** 지정 가능하다.

```python
cond_2d = np.array([[1, 2], [3, 4]])
arr_1d.take(cond_2d)
```

    array([[3, 7],
           [2, 0]])

#### 2차원 배열

```python
# 2차원 배열
arr_2d = np.array([[1, 2, 6, 2, -85, 100],
                   [7, 9, -20, 120, 0, 12]])
arr_2d, arr_2d.shape
```

    (array([[  1,   2,   6,   2, -85, 100],
            [  7,   9, -20, 120,   0,  12]]),
     (2, 6))

```python
cond_axis0 = np.array([0, 1, 0])
cond_axis1 = np.array([1, 3, 5])
arr_2d[cond_axis0, cond_axis1]
```

    array([  2, 120, 100])

## 10. Numpy Data I/O

`.txt`, `.csv`, `.npy` 형태로 데이터를 저장하거나 읽어올 수 있다.

```python
arr = np.arange(24).reshape(3, 8)
arr
```

    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]])

### 10.1. csv

```python
# 데이터 csv로 저장
np.savetxt("numpy_sample.csv", arr, delimiter=",")
```

```python
# csv에서 데이터 로드
new_arr = np.loadtxt("./numpy_sample.csv", delimiter=",")
new_arr
```

    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11., 12., 13., 14., 15.],
           [16., 17., 18., 19., 20., 21., 22., 23.]])

### 10.2. npy

```python
# 데이터 npy로 저장
np.save("numpy_sample_2", arr)
```

```python
# npy로부터 데이터 로드
new_arr = np.load("./numpy_sample_2.npy")
new_arr
```

    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]])
