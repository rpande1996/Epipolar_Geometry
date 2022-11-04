import copy
import random

import cv2
import numpy as np
from scipy.io import loadmat

input = "../input/"

data = loadmat(input + 'matches.mat')
r1 = data['r1']
r2 = data['r2']
c1 = data['c1']
c2 = data['c2']
matches = data['matches']

x1 = c1[matches[:, 0] - 1]
y1 = r1[matches[:, 0] - 1]
x2 = c2[matches[:, 1] - 1]
y2 = r2[matches[:, 1] - 1]


def approxFunda(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    mn1 = np.sum(p1, axis=0) / p1.shape[0]
    mn2 = np.sum(p2, axis=0) / p2.shape[0]
    mn1 = np.reshape(mn1, (1, mn1.shape[0]))
    mn2 = np.reshape(mn2, (1, mn2.shape[0]))
    trnsl_p1 = p1 - mn1
    trnsl_p2 = p2 - mn2

    added_sqs1 = np.sum(trnsl_p1 ** 2, axis=1)
    added_sqs1 = np.reshape(added_sqs1, (trnsl_p1.shape[0], 1))

    mn_dist1 = np.sum(added_sqs1 ** (1 / 2), axis=0) / trnsl_p1.shape[0]
    factor1 = 2 ** (1 / 2) / mn_dist1[0]

    added_sqs2 = np.sum(trnsl_p2 ** 2, axis=1)
    added_sqs2 = np.reshape(added_sqs2, (trnsl_p2.shape[0], 1))

    mn_dist2 = np.sum(added_sqs2 ** (1 / 2), axis=0) / trnsl_p2.shape[0]
    factor2 = 2 ** (1 / 2) / mn_dist2[0]

    nmlz1 = factor1 * trnsl_p1
    nmlz2 = factor2 * trnsl_p2

    trnslmtrx1 = np.array([[1, 0, -mn1[0][0]], [0, 1, -mn1[0][1]], [0, 0, 1]])
    scmtrx1 = np.array([[factor1, 0, 0], [0, factor1, 0], [0, 0, 1]])

    trnslmtrx2 = np.array([[1, 0, -mn2[0][0]], [0, 1, -mn2[0][1]], [0, 0, 1]])
    scmtrx2 = np.array([[factor2, 0, 0], [0, factor2, 0], [0, 0, 1]])

    T1 = np.dot(scmtrx1, trnslmtrx1)
    T2 = np.dot(scmtrx2, trnslmtrx2)

    A = np.zeros((p1.shape[0], 9))
    for i in range(p1.shape[0]):
        A[i, :] = [nmlz2[i][0] * nmlz1[i][0], nmlz2[i][0] * nmlz1[i][1], nmlz2[i][0], nmlz2[i][1] * nmlz1[i][0],
                   nmlz2[i][1] * nmlz1[i][1], nmlz2[i][1], nmlz1[i][0], nmlz1[i][1], 1]

    U, S, Vt = np.linalg.svd(A)

    V = Vt.T
    V = V[:, -1]
    F = np.zeros((3, 3))
    count = 0
    for i in range(3):
        for j in range(3):
            F[i, j] = V[count]
            count += 1

    u, s, vt = np.linalg.svd(F)

    s[-1] = 0
    newS = np.zeros((3, 3))
    for i in range(3):
        newS[i, i] = s[i]

    newF = np.dot((np.dot(u, newS)), vt)

    regF = np.dot(np.dot(T2.T, newF), T1)
    regF = regF / regF[-1, -1]

    return regF


def RANSAC(feature1, feature2):
    thresh = 0.05
    prsntInliers = 0
    perfF = []
    p = 0.99
    N = 2000
    count = 0
    while count < N:
        inlier_count = 0
        random_feature1 = []
        random_feature2 = []
        RandList = np.random.randint(len(feature1), size=8)
        for k in RandList:
            random_feature1.append(feature1[k])
            random_feature2.append(feature2[k])
        F = approxFunda(random_feature1, random_feature2)

        One = np.ones((len(feature1), 1))
        X_1 = np.hstack((feature1, One))
        X_2 = np.hstack((feature2, One))
        E_1, E_2 = X_1 @ F.T, X_2 @ F
        err = np.sum(E_2 * X_1, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((E_1[:, :-1], E_2[:, :-1])) ** 2, axis=1,
                                                                     keepdims=True)
        Inl = err <= thresh
        InlCnt = np.sum(Inl)
        if prsntInliers < InlCnt:
            prsntInliers = InlCnt
            coor = np.where(Inl == True)
            x1_ar = np.array(feature1)
            x2_ar = np.array(feature2)
            inlier_x1 = x1_ar[coor[0][:]]
            inlier_x2 = x2_ar[coor[0][:]]

            perfF = F

        RatioInl = InlCnt / len(feature1)
        if np.log(1 - (RatioInl ** 8)) == 0: continue
        N = np.log(1 - p) / np.log(1 - (RatioInl ** 8))
        count += 1
    return perfF, inlier_x1, inlier_x2


def randomPts(pt1_list, pt2_list, val):
    index = []
    for i in range(val):
        num = random.randint(0, len(pt1_list) - 1)
        index.append(num)
    new_pts1 = []
    new_pts2 = []
    for i in range(len(index)):
        val = index[i]
        new_pts1.append(pt1_list[val])
        new_pts2.append(pt2_list[val])

    new_pts1 = np.asarray(new_pts1)
    new_pts2 = np.asarray(new_pts2)
    return new_pts1, new_pts2


def drawlines(img1, lines, pts):
    i1 = copy.deepcopy(img1)
    r, c, _ = img1.shape
    for r, pt1 in zip(lines, pts):
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        i1 = cv2.line(i1, (x0, y0), (x1, y1), color1, 1)
        i1 = cv2.circle(i1, tuple(pt1), 2, color2, -1)
    return i1


def genPts(x1, y1):
    p1 = []
    for i in range(len(x1)):
        temp = [x1[i][0], y1[i][0]]
        p1.append(temp)

    return p1


def computeEpilines(img1, img2, x1, x2, y1, y2):
    p1 = genPts(x1, y1)
    p2 = genPts(x2, y2)
    F, pt1, pt2 = RANSAC(p1, p2)

    points1, points2 = randomPts(pt1, pt2, 7)

    ones = np.ones((points1.shape[0], 1))
    final_p1 = np.append(points1, ones, axis=1)
    final_p2 = np.append(points2, ones, axis=1)

    lines1 = np.dot(F, final_p1.T)
    lines2 = np.dot(final_p2, F)

    i1 = drawlines(img2, lines1.T, points2)
    i2 = drawlines(img1, lines2, points1)

    final_image = np.hstack((i1, i2))
    return final_image, F


img1 = cv2.imread(input + "chapel00.png")
img2 = cv2.imread(input + "chapel01.png")

final_image, F = computeEpilines(img1, img2, x1, x2, y1, y2)
# print("Fundamenal Matrix: \n", F)

out = "../output/"

cv2.imshow("Epilines", final_image)
cv2.imwrite(out + "epilines.jpg", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
