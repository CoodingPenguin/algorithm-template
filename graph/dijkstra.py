# title: 다익스트라 최단 경로 알고리즘
# src: 이것이 취업을 위한 코딩테스트다 p.248 - p.249
# time: O(ElogV) (E=간선 개수, V=노드 개수)

import heapq
INF = int(1e9)  # 무한

def dijkstra(graph, start, distance):
    '''
    특정 노드에서 다른 노드까지 가는 각각의 최단 경로를 구하는 그리디 알고리즘
    - graph: (연결 노드, 비용)으로 이루어진 2차원 인접 리스트
    - start: 시작 노드
    - distance: 시작노드 - i번째 노드까지의 최단 경로를 저장한 리스트
    '''
    q = []  # 현재 가장 가까운 노드를 저장하는 우선순위 큐
    # 요소는 (시작노드에서 해당 노드까지의 최단거리, 노드)
    # 시작노드 - 시작노드까지의 최단 경로는 0으로 설정
    heapq.heappush(q, (0, start))   
    distance[start] = 0

    while q:
        # dist: 시작노드에서 해당 노드까지의 최단거리
        # now: 최단거리가 가장 짧은 노드
        dist, now = heapq.heappop(q)
        
        # 이미 최단거리가 계산된 노드일 경우 패스
        if distance[now] < dist:
            continue
        # 현재 노드와 연결된 인접 노드 체크
        for i in graph[now]:
            # 인접 노드까지의 비용 다시 계산
            cost = dist + i[1]
            # 그 비용이 최단 거리보다 작으면 갱신 
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))


if __name__=='__main__':
    start = 1
    graph = [
        [], # 인덱스와 숫자를 맞추기 위한 빈 리스트
        [(2, 2), (3, 5), (4, 1)],   # (연결된  노드, 가는 데 필요한 비용)
        [(3, 3), (4, 2)],
        [(2, 3), (6, 5)],
        [(3, 3), (5, 1)],
        [(3, 1), (6, 2)],
        []
    ]
    distance = [INF] * len(graph)

    dijkstra(graph, start, distance)
    print(distance[1:])