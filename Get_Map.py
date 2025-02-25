
""" Ori Gat_map """

import numpy as np

def get_map(h, w):
    FoV_list = [19.45, 23.12, 30.08, 33.40, 39.60, 48.46, 65.47, 73.74]
    thick_list = np.array([3, 6, 9]) * 0.001

    while True:
        g_dis = 0
        while g_dis < 0.2:
            g_dis = np.random.rand(1) * (5 - 0.2) + 0.2
        g_size = 0
        while g_size < 0.4:
            g_size = np.random.rand(1) * (3 - 0.4) + 0.4
        g_angle1 = 90
        while np.abs(g_angle1) > 60:
            g_angle1 = 10 * (np.random.randn(1, 1))
        g_angle2 = 90
        while np.abs(g_angle2) > 15:
            g_angle2 = 4 * (np.random.randn(1, 1))
        FoV = np.random.rand(1) * (73.74 - 19.45) + 19.45

        thickness = np.random.rand(1) * 0.007 + 0.003

        cor_x2 = np.tan((-FoV / 2 + np.abs(g_angle1)) / 180 * np.pi) * g_dis
        cor_x3 = np.tan((FoV / 2 + np.abs(g_angle1)) / 180 * np.pi) * g_dis

        if cor_x3 - cor_x2 < g_size:
            break

    n2 = 1.474

    map_T = compute_theta_map(w, h, g_dis, g_angle1, g_angle2, FoV, g_size)
    [map_A, map_B] = compute_map(map_T, n2)

    cos_map_T2 = 1.0 / np.sqrt(1 - (1 / n2 * np.sin(map_T)) ** 2)
    k_c = np.random.rand(1) * (32 - 4) + 4
    map_alpha = np.exp(-k_c * thickness * cos_map_T2)

    map_coe_R = map_B + map_B * (map_A * map_A * map_alpha * map_alpha) / (
            1 - map_B * map_B * map_alpha * map_alpha)
    map_coe_T = (map_A * map_A * map_alpha) / (1 - map_B * map_B * map_alpha * map_alpha)

    return map_coe_T, map_coe_R

def compute_theta_map(w, h, g_dis, angle1, angle2, FoV, g_size):
    cor_x2 = np.tan((-FoV/2 + np.abs(angle1))/180*np.pi)*g_dis
    cor_x3 = np.tan((FoV/2 + np.abs(angle1))/180*np.pi)*g_dis
    centerX = 0
    centerY = 0
    centerZ = w/2/np.tan(FoV/2/180*np.pi)
    if cor_x3-cor_x2 > g_size:
        step = g_size*np.cos(angle1/180*np.pi)/w
    else:
        step = 1
    map_T = np.zeros((h, w))
    n = np.array([0.0, 0.0, 0.0], dtype=float)
    nv = np.array([np.sin(angle1/180*np.pi)*np.cos(angle2/180*np.pi),
                   np.sin(angle1/180*np.pi)*np.sin(angle2/180*np.pi),
                   np.cos(angle1/180*np.pi)])
    for i in range(-w//2, w//2):
        for j in range(-h//2, h//2):
            n[0] = (i+0.5)*step+centerX
            n[1] = (j+0.5)*step+centerY
            n[2] = centerZ
            n /= np.linalg.norm(n)
            map_T[j+h//2, i+w//2] = np.arccos(np.dot(nv.flatten(), n))

    return map_T

def compute_map(map_T, n2):
    n1 = 1

    def Rs1(x):
        return ((n1 * np.cos(x) - n2 * np.sqrt(1 - (n1 / n2 * np.sin(x)) ** 2)) /
                (n1 * np.cos(x) + n2 * np.sqrt(1 - (n1 / n2 * np.sin(x)) ** 2))) ** 2

    def Rp1(x):
        return ((n2 * np.cos(x) - n1 * np.sqrt(1 - (n1 / n2 * np.sin(x)) ** 2)) /
                (n2 * np.cos(x) + n1 * np.sqrt(1 - (n1 / n2 * np.sin(x)) ** 2))) ** 2

    R = lambda x: 0.5 * (Rs1(x) + Rp1(x))

    map_A = np.zeros_like(map_T)
    map_B = np.zeros_like(map_T)
    h, w = map_T.shape

    for i in range(h):
        for j in range(w):
            map_A[i, j] = 1 - R(map_T[i, j])
            map_B[i, j] = R(map_T[i, j])

    return map_A, map_B
