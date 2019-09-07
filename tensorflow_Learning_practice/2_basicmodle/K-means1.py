# K-means Algorithm is a clustering algorithm
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1) 
 
def get_distance(p1, p2):
    diff = [x-y for x, y in zip(p1, p2)]
    distance = np.sqrt(sum(map(lambda x: x**2, diff)))
    return distance
 
 
# 计算多个点的中心
# cluster = [[1,2,3], [-2,1,2], [9, 0 ,4], [2,10,4]]
def calc_center_point(cluster):
    N = len(cluster)
    m = np.matrix(cluster).transpose().tolist()
    center_point = [sum(x)/N for x in m]
    return center_point
 
def calc_var_point(points):
    m = np.matrix(points).transpose().tolist()
    var_point = [np.array(x).var() for x in m]
    return var_point

 
# 检查两个点是否有差别
def check_center_diff(center, new_center):
    n = len(center)
    for c, nc in zip(center, new_center):
        if c != nc:
            return False
    return True
 
 
# K-means算法的实现
def K_means(points, center_points):
 
    N = len(points)         # 样本个数
    n = len(points[0])      # 单个样本的维度
    k = len(center_points)  # k值大小
 
    tot = 0
    while True:             # 迭代
        temp_center_points = [] # 记录中心点
 
        clusters = []       # 记录聚类的结果
        for c in range(0, k):
            clusters.append([]) # 初始化
 
        # 针对每个点，寻找距离其最近的中心点（寻找组织）
        for i, data in enumerate(points):
            distances = []
            for center_point in center_points:
                distances.append(get_distance(data, center_point))
            index = distances.index(min(distances)) # 找到最小的距离的那个中心点的索引，
            # print(index)
            clusters[index].append(data)    # 那么这个中心点代表的簇，里面增加一个样本
 
        tot += 1
        print(tot, '次迭代   ')
        k = len(clusters)
        colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
        for i, cluster in enumerate(clusters):
            # print(tot)
            if tot < 7:
                data = np.array(cluster)
                data_x = [x[0] for x in data]
                data_y = [x[1] for x in data]
                # fig = plt.figure()
                plt.title("K-means %d iter" % tot)
                plt.subplot(2, 3, tot)
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.plot(data_x, data_y, colors[random.randint(0, 6)])
                plt.axis([0, 1000, 0, 1000])
                plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
                plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])
 
        # 重新计算中心点（该步骤可以与下面判断中心点是否发生变化这个步骤，调换顺序）
        for cluster in clusters:
            temp_center_points.append(calc_center_point(cluster))
 
        # 在计算中心点的时候，需要将原来的中心点算进去
        for j in range(0, k):
            if len(clusters[j]) == 0:
                temp_center_points[j] = center_points[j]
 
        # 判断中心点是否发生变化：即，判断聚类前后样本的类别是否发生变化
        for c, nc in zip(center_points, temp_center_points):
            if not check_center_diff(c, nc):
                center_points = temp_center_points[:]   # 复制一份
                break
        else:   # 如果没有变化，那么退出迭代，聚类结束
            break
        # print(len(center_points))
    plt.show()
    return clusters # 返回聚类的结果
 
def show_phote(clusters):
    colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
    for i, cluster in enumerate(clusters):
        data = np.array(cluster)
        data_x = [x[0] for x in data]
        data_y = [x[1] for x in data]
                # fig = plt.figure()
    # plt.subplot(2, 3, tot)
        plt.plot(data_x, data_y, colors[random.randint(0, 6)])
        plt.axis([0, 1000, 0, 1000])
        plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
        plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])
    plt.scatter(np.array(center_points)[:,0],np.array(center_points)[:,1],marker="*",c="red"
                ,label="cluster center")
    plt.legend()
    plt.show()
 
def input_data():
    N = 1000
    # 产生点的区域
    # areas = [area_1, area_2, area_3, area_4, area_5]
    areas = []
    row = [x for x in range(0, N + 1, 50)]
    col = [y for y in range(0, N+1, 100)]

    flag = 0
    for i in range(0, len(row) - 1):
        sub_areas = []
        sub_areas.append(row[i])
        sub_areas.append(row[i+1])
        for j in range(1 if flag % 2 else 0, len(col) - 1, 2):
            sub_areas.append(col[j])
            sub_areas.append(col[j+1])
            # print(sub_areas)
            areas.append(sub_areas)
            sub_areas = sub_areas[0:2]
        flag += 1
    # print(areas)
    k = len(areas)
 
    # 在各个区域内，随机产生一些点
    points = []
    mean_points = []
    var_points = []
    for area in areas:
        # rnd_num_of_points = random.randint(50, 200)
        num_of_points = 10
        sub_points = []
        for r in range(0, num_of_points):
            # rnd_add = random.randint(0, 50)
            rnd_x = random.randint(area[0], area[1])
            rnd_y = random.randint(area[2], area[3])
            points.append([rnd_x, rnd_y])
            sub_points.append([rnd_x, rnd_y])
        var_points.append(calc_var_point(sub_points))
        mean_points.append(calc_center_point(sub_points))
 
    center_points = []
    for area in areas:
        sub_areas = []
        sub_areas.append((area[0] + area[1]) // 2)
        sub_areas.append((area[3] + area[2]) // 2)
        center_points.append(sub_areas)
    # print(len(points))
    # print(len(center_points))
    return points, center_points, mean_points, var_points

def plt_mean_photo(mean_clusters_points, mean_points):
    colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
    data = np.array(mean_points)
    data_x = [x[0] for x in data]
    data_y = [x[1] for x in data]
                # fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data_x, data_y, colors[random.randint(0, 6)])
    plt.axis([0, 1000, 0, 1000])
    plt.title("cluster front points means")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
    plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])
    data = np.array(mean_clusters_points)
    data_x = [x[0] for x in data]
    data_y = [x[1] for x in data]
                # fig = plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(data_x, data_y, colors[random.randint(0, 6)])
    plt.axis([0, 1000, 0, 1000])
    plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
    plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])
    plt.title("cluster back points means")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    # plt.grid()
    plt.show()

def plt_var_photo(var_clusters_points, var_points):
    colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
    data = np.array(var_points)
    data_x = [x[0] for x in data]
    data_y = [x[1] for x in data]
                # fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data_x, data_y, colors[random.randint(0, 6)])
    plt.axis([0, 500, 0, 1500])
    plt.title("cluster front points vars")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks([x for x in range(500 + 1) if x % 100 ==0])
    plt.yticks([y for y in range(1500 + 1) if y % 100 ==0])
    data = np.array(var_clusters_points)
    data_x = [x[0] for x in data]
    data_y = [x[1] for x in data]
                # fig = plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(data_x, data_y, colors[random.randint(0, 6)])
    plt.axis([0, 500, 0, 1500])
    plt.xticks([x for x in range(500 + 1) if x % 100 ==0])
    plt.yticks([y for y in range(1500 + 1) if y % 100 ==0])
    plt.title("cluster back points vars")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
 
    points, center_points, mean_points, var_points = input_data()
    clusters = K_means(points, center_points)
    show_phote(clusters)
    mean_clusters_points = []
    var_clusters_points = []
    for cluster in clusters:
        mean_clusters_points.append(calc_center_point(cluster))
        var_clusters_points.append(calc_var_point(cluster))
    # print(var_clusters_points)
    plt_mean_photo(mean_clusters_points, mean_points)
    plt_var_photo(var_clusters_points, var_points)