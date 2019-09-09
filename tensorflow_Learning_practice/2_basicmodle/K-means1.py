# K-means Algorithm is a clustering algorithm
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1) 
 
def get_distance(p1, p2):
    diff = [x-y for x, y in zip(p1, p2)]
    distance = np.sqrt(sum(map(lambda x: x**2, diff)))
    return distance
 
def calc_center_point(cluster):
    N = len(cluster)
    m = np.matrix(cluster).transpose().tolist()
    center_point = [sum(x)/N for x in m]
    return center_point
 
def calc_var_point(points):
    m = np.matrix(points).transpose().tolist()
    var_point = [np.array(x).var() for x in m]
    return var_point

 
def check_center_diff(center, new_center):
    # n = len(center)
    for c, nc in zip(center, new_center):
        if c != nc:
            return False
    return True
 
 
# K-means
def K_means(points, center_points):
 
    # N = len(points)         # 样本个数
    # n = len(points[0])      # 单个样本的维度
    k = len(center_points)  # k值大小
 
    tot = 0
    err = []
    while True:   
        sub_err = []         
        temp_center_points = [] 
 
        clusters = []       
        for c in range(0, k):
            clusters.append([]) 
 
        for i, data in enumerate(points):
            distances = []
            for center_point in center_points:
                distances.append(get_distance(data, center_point))
            index = distances.index(min(distances))
            # print(index)
            clusters[index].append(data)  
 
        tot += 1
        print(tot, '次迭代   ')
        k = len(clusters)
        colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
        # for i, cluster in enumerate(clusters):
        #     if tot < 6:
        #         data = np.array(cluster)
        #         data_x = [x[0] for x in data]
        #         data_y = [x[1] for x in data]
        #         tott = tot - 1
        #         plt.title("K-means %d iter" %tott)
        #         plt.subplot(2, 2, tot)
        #         plt.xlabel("X")
        #         plt.ylabel("Y")
        #         plt.plot(data_x, data_y, colors[random.randint(0, 6)])
        #         plt.axis([0, 1000, 0, 1000])
        #         plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
        #         plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])

        # 重新计算中心点
        for cluster in clusters:
            temp_center_points.append(calc_center_point(cluster))
 
        for j in range(0, k):
            if len(clusters[j]) == 0:
                temp_center_points[j] = center_points[j]
 
        # 判断中心点是否发生变化
        error = cal_error(clusters, points)
        print("第 %d 迭代, 错误率%f" %(tot,error))
        sub_err.append(tot)
        sub_err.append(error)
        err.append(sub_err)
        for c, nc in zip(center_points, temp_center_points):
            if not check_center_diff(c, nc):
                center_points = temp_center_points[:]   
                break
        else:   
            break
        
        # print(len(center_points))
    plt.title("K-means %d iter" %tot)
    plt.show()
    return clusters, err 
 
def show_phote(clusters, mean_clusters_points):
    colors = ['r.', 'g.', 'b.', 'k.', 'y.', 'c.', 'm.']
    for i, cluster in enumerate(clusters):
        data = np.array(cluster)
        data_x = [x[0] for x in data]
        data_y = [x[1] for x in data]
        plt.plot(data_x, data_y, colors[random.randint(0, 6)])
        plt.axis([0, 1000, 0, 1000])
        plt.xticks([x for x in range(1000 + 1) if x % 100 ==0])
        plt.yticks([y for y in range(1000 + 1) if y % 100 ==0])
    plt.scatter(np.array(mean_clusters_points)[:,0],np.array(mean_clusters_points)[:,1],marker="*",c="red"
                ,label="cluster center")
    plt.legend()
    plt.title('final cluster')
    plt.show()

def show_err(errs):
    x = [x[0] for x in errs]
    y = [y[1] for y in errs]
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color='blue', linewidth=1)
    plt.xlabel('iter num')
    plt.ylabel('error rate %')
    plt.title('error rate %')
    plt.ylim(0., 14.)
    plt.xlim(0, 15)
    plt.show()



def input_data():
    N = 1000
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
    # k = len(areas)
 
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
    i = 0
    for area in areas:
        sub_areas = []
        # sub_areas.append((area[0] + area[1]) // 2)
        # sub_areas.append((area[3] + area[2]) // 2)
        sub_areas.append(area[0] + random.randint(0, 50))
        sub_areas.append(area[2] + random.randint(0, 100))
        # sub_areas.append(random.randint(0, 1001))
        # sub_areas.append(random.randint(0, 1001))

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
        
    plt.subplot(2, 1, 1)
    plt.plot(data_x, data_y, colors[random.randint(0, 6)])
    plt.axis([0, 500, 0, 1500])
    plt.title("cluster front points vars")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.xticks([x for x in range(500 + 1) if x % 100 ==0])
    plt.yticks([y for y in range(1500 + 1) if y % 100 ==0])
    data = np.array(var_clusters_points)
    data_x = [x[0] for x in data]
    data_y = [x[1] for x in data]
            
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

def SSE(mean_points, points):
    SSE = 0
    pos = 0
    for mean_point in mean_points:
        for point in points[pos: pos+10]:
            SSE += get_distance(point, mean_point)
            pos += 10
    return SSE 

def cal_error(clusters, points):
    cnt = 0
    pos = 0
    for cluster in clusters:

        for po in points[pos: pos+10]:
            if po not in cluster:
                cnt += 1
            # SSE_back += get_distance(point,center_points_cluster)
        pos += 10
    return cnt / 1000 * 100

if __name__ == '__main__':
    points, center_points, mean_points, var_points = input_data()
    clusters, errs= K_means(points, center_points)
    print(errs)
  
    mean_clusters_points = []
    var_clusters_points = []
    SSE_back = 0
    error = cal_error(clusters, points)
    print(error)
    for cluster in clusters:
        center_points_cluster = calc_center_point(cluster)
        mean_clusters_points.append(center_points_cluster)
        var_clusters_points.append(calc_var_point(cluster))
    # print(var_clusters_points)
    # print(cnt)
    show_phote(clusters, mean_clusters_points)
    plt_mean_photo(mean_clusters_points, mean_points)
    plt_var_photo(var_clusters_points, var_points)
    SSE_front = SSE(mean_points, points)
    show_err(errs)
    # SSE_back = SSE(mean_clusters_points, clusters)
    # print("SSE 为: ", SSE_front, SSE_back)