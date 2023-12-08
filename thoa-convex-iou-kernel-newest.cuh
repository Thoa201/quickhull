// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONVEX_IOU_CUDA_KERNEL_CUH
#define CONVEX_IOU_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define MAXN 100
#define NMAX 512
__device__ const double EPS = 1E-8;

__device__ inline int sig(double d) { return (d > EPS) - (d < -EPS); }

struct Point {
    double x, y;
    __device__ Point() {}
    __device__ Point(double x, double y) : x(x), y(y) {}
};

__device__ inline bool point_same(Point& a, Point& b) {
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void swap1(Point* a, Point* b) {
    Point temp;
    temp.x = a->x;
    temp.y = a->y;

    a->x = b->x;
    a->y = b->y;

    b->x = temp.x;
    b->y = temp.y;
}

__device__ inline void reverse1(Point* a, const int n) {
    for (int i = 0; i < (n - 1) / 2.0; i++) {
        Point* j = &(a[i]);
        Point* k = &(a[n - 1 - i]);
        swap1(j, k);
    }
}

__device__ inline double cross(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

__device__ inline double dis(Point a, Point b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
__device__ inline double area(Point* ps, int n) {
    ps[n] = ps[0];
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}
__device__ inline double polygon_area_grad(Point* ps, int n,
                                           int* polygon_to_pred_index,
                                           int n_pred, double* grad_C) {
    ps[n] = ps[0];
    double partion_grad[4 * 30 + 2];
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
        partion_grad[i * 4 + 2] = ps[i + 1].y;
        partion_grad[i * 4 + 3] = -ps[i + 1].x;
        if (i != n - 1) {
            partion_grad[i * 4 + 4] = -ps[i].y;
            partion_grad[i * 4 + 5] = ps[i].x;
        } else {
            partion_grad[0] = -ps[i].y;
            partion_grad[1] = ps[i].x;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_pred; j++) {
            if (i == polygon_to_pred_index[j]) {
                grad_C[2 * polygon_to_pred_index[j + n_pred]] =
                        (partion_grad[i * 4] + partion_grad[i * 4 + 2]) / 2;
                break;
            }
        }
        for (int j = 0; j < n_pred; j++) {
            if (i == polygon_to_pred_index[j]) {
                grad_C[2 * polygon_to_pred_index[j + n_pred] + 1] =
                        (partion_grad[i * 4 + 1] + partion_grad[i * 4 + 1 + 2]) / 2;
                break;
            }
        }
    }

    return res / 2.0;
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p,
                                double* cut_grad, int m, int n, int i) {
    double s1, s2;
    double s2_s1_2;
    double ds1_dxc, ds1_dyc, ds2_dxd, ds2_dyd;
    double dxp_dxc, dxp_dyc, dxp_dxd, dxp_dyd, dyp_dxc, dyp_dyc, dyp_dxd, dyp_dyd;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);

    ds1_dxc = -(b.y - a.y);
    ds1_dyc = b.x - a.x;
    ds2_dxd = ds1_dxc;
    ds2_dyd = ds1_dyc;
    s2_s1_2 = (s2 - s1) * (s2 - s1);

    if (sig(s1) == 0 && sig(s2) == 0) return 2;
    if (sig(s2 - s1) == 0) return 0;

    dxp_dxc =
            ((s2 - d.x * ds1_dxc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dxc)) /
            (s2_s1_2);
    dxp_dyc =
            ((0 - d.x * ds1_dyc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dyc)) /
            (s2_s1_2);
    dxp_dxd =
            ((c.x * ds2_dxd - s1) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dxd)) /
            (s2_s1_2);
    dxp_dyd =
            ((c.x * ds2_dyd - 0) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dyd)) /
            (s2_s1_2);

    dyp_dxc =
            ((0 - d.y * ds1_dxc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dxc)) /
            (s2_s1_2);
    dyp_dyc =
            ((s2 - d.y * ds1_dyc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dyc)) /
            (s2_s1_2);
    dyp_dxd =
            ((c.y * ds2_dxd - 0) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dxd)) /
            (s2_s1_2);
    dyp_dyd =
            ((c.y * ds2_dyd - s1) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dyd)) /
            (s2_s1_2);

    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    if (i == n - 1) {
        cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
        cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
        cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
        cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
        cut_grad[4 * n * m + 0] = dxp_dxd;  // + dyp_dxd;
        cut_grad[4 * n * m + 1] = dyp_dxd;
        cut_grad[4 * n * m + 2] = dxp_dyd;  // + dyp_dyd;
        cut_grad[4 * n * m + 3] = dyp_dyd;
    } else {
        cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
        cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
        cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
        cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
        cut_grad[4 * n * m + 4 * (i + 1)] = dxp_dxd;  // + dyp_dxd;
        cut_grad[4 * n * m + 4 * (i + 1) + 1] = dyp_dxd;
        cut_grad[4 * n * m + 4 * (i + 1) + 2] = dxp_dyd;  // + dyp_dyd;
        cut_grad[4 * n * m + 4 * (i + 1) + 3] = dyp_dyd;
    }

    return 1;
}
__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b,
                                   double* cut_grad) {
    Point pp[MAXN];
    double ccur_grad[MAXN] = {};
    int m = 0;
    p[n] = p[0];
    int k = n;
    for (int i = 0; i < n; i++) {
        if (sig(cross(a, b, p[i])) > 0) {
            pp[m] = p[i];
            ccur_grad[4 * n * m + 4 * i] = 1.0;
            ccur_grad[4 * n * m + 4 * i + 3] = 1.0;
            m++;
        }
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
            lineCross(a, b, p[i], p[i + 1], pp[m], ccur_grad, m, n, i);
            m++;
        }
    }

    n = 0;
    for (int i = 0; i < m; i++) {
        if (!i || !(point_same(pp[i], pp[i - 1]))) {
            p[n] = pp[i];
            for (int j = 0; j < 4 * k; j++) {
                cut_grad[4 * k * n + j] = ccur_grad[4 * k * i + j];
            }
            n++;
        }
    }

    while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d,
                                       double* grad_AB, int order,
                                       int convex_n) {
    Point o(0, 0);
    int res_flag = 0;
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0) return 0.0;
    if (s1 == -1) {
        Point* i = &a;
        Point* j = &b;
        swap1(i, j);
        res_flag = 1;
    }
    if (s2 == -1) {
        Point* i = &c;
        Point* j = &d;
        swap1(i, j);
    }
    Point p[10] = {o, a, b};
    int n = 3, n0 = 3, n1, n2, n3;
    double cut_grad1[MAXN] = {};
    double cut_grad2[MAXN] = {};
    double cut_grad3[MAXN] = {};
    double p1_p_grad[10][10] = {};
    double p2_p1_grad[10][10] = {};
    double p3_p2_grad[10][10] = {};

    double p3_p1_grad[10][10] = {};
    double p3_p_grad[10][10] = {};

    // 1
    polygon_cut(p, n, o, c, cut_grad1);
    n1 = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4 * n0; j++) {
            if (!(j % 2)) {
                p1_p_grad[2 * i][j / 2] = cut_grad1[4 * n0 * i + j];
            } else {
                p1_p_grad[2 * i + 1][j / 2] = cut_grad1[4 * n0 * i + j];
            }
        }
    }

    // 2
    polygon_cut(p, n, c, d, cut_grad2);
    n2 = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4 * n1; j++) {
            if (!(j % 2)) {
                p2_p1_grad[2 * i][j / 2] = cut_grad2[4 * n1 * i + j];
            } else {
                p2_p1_grad[2 * i + 1][j / 2] = cut_grad2[4 * n1 * i + j];
            }
        }
    }
    // 3
    polygon_cut(p, n, d, o, cut_grad3);
    n3 = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4 * n2; j++) {
            if (!(j % 2)) {
                p3_p2_grad[2 * i][j / 2] = cut_grad3[4 * n2 * i + j];
            } else {
                p3_p2_grad[2 * i + 1][j / 2] = cut_grad3[4 * n2 * i + j];
            }
        }
    }

    // mul
    //  p3_p2(n3 * n2) * p2_p1(n2 * n1) = p3_p1 (n3 * n1)
    for (int i = 0; i < 2 * n3; i++) {
        for (int j = 0; j < 2 * n1; j++) {
            double sum = 0.0;
            for (int m = 0; m < 2 * n2; m++) {
                sum = sum + p3_p2_grad[i][m] * p2_p1_grad[m][j];
            }
            p3_p1_grad[i][j] = sum;
        }
    }

    // p3_p1 (n3 * n1) * p1_p (n1 * n0) = p3_p (n3 * n0)
    for (int i = 0; i < 2 * n3; i++) {
        for (int j = 0; j < 2 * n0; j++) {
            double sum = 0.0;
            for (int m = 0; m < 2 * n1; m++) {
                sum = sum + p3_p1_grad[i][m] * p1_p_grad[m][j];
            }
            p3_p_grad[i][j] = sum;
        }
    }

    // calculate S_grad
    int polygon_index_box_index[20];
    double grad_polygon[20];
    double S_grad[6];

    for (int i = 0; i < n3; i++) {
        polygon_index_box_index[i] = i;
        polygon_index_box_index[i + n3] = i;
    }

    double res =
            polygon_area_grad(p, n3, polygon_index_box_index, n3, grad_polygon);

    if (s1 * s2 == -1) {
        for (int j = 0; j < 2 * 3; j++) {
            double sum = 0.0;
            for (int m = 0; m < 2 * n3; m++) {
                sum = sum - grad_polygon[m] * p3_p_grad[m][j];
            }
            S_grad[j] = sum;
        }

        if (order != convex_n - 1) {
            if (res_flag) {
                grad_AB[2 * order] += S_grad[4];
                grad_AB[2 * order + 1] += S_grad[5];
                grad_AB[2 * order + 2] += S_grad[2];
                grad_AB[2 * order + 3] += S_grad[3];

            } else {
                grad_AB[2 * order] += S_grad[2];
                grad_AB[2 * order + 1] += S_grad[3];
                grad_AB[2 * order + 2] += S_grad[4];
                grad_AB[2 * order + 3] += S_grad[5];
            }
        } else {
            if (res_flag) {
                grad_AB[2 * order] += S_grad[4];
                grad_AB[2 * order + 1] += S_grad[5];
                grad_AB[0] += S_grad[2];
                grad_AB[1] += S_grad[3];

            } else {
                grad_AB[2 * order] += S_grad[2];
                grad_AB[2 * order + 1] += S_grad[3];
                grad_AB[0] += S_grad[4];
                grad_AB[1] += S_grad[5];
            }
        }
        res = -res;
    } else {
        for (int j = 0; j < 2 * 3; j++) {
            double sum = 0.0;
            for (int m = 0; m < 2 * n3; m++) {
                sum = sum + grad_polygon[m] * p3_p_grad[m][j];
            }
            S_grad[j] = sum;
        }

        if (order != convex_n - 1) {
            if (res_flag) {
                grad_AB[2 * order] += S_grad[4];
                grad_AB[2 * order + 1] += S_grad[5];
                grad_AB[2 * order + 2] += S_grad[2];
                grad_AB[2 * order + 3] += S_grad[3];
            } else {
                grad_AB[2 * order] += S_grad[2];
                grad_AB[2 * order + 1] += S_grad[3];
                grad_AB[2 * order + 2] += S_grad[4];
                grad_AB[2 * order + 3] += S_grad[5];
            }
        } else {
            if (res_flag) {
                grad_AB[2 * order] += S_grad[4];
                grad_AB[2 * order + 1] += S_grad[5];
                grad_AB[0] += S_grad[2];
                grad_AB[1] += S_grad[3];
            } else {
                grad_AB[2 * order] += S_grad[2];
                grad_AB[2 * order + 1] += S_grad[3];
                grad_AB[0] += S_grad[4];
                grad_AB[1] += S_grad[5];
            }
        }
    }
    return res;
}

__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2, int n2,
                                        double* grad_AB) {
    if (area(ps1, n1) < 0) reverse1(ps1, n1);
    if (area(ps2, n2) < 0) reverse1(ps2, n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    double res = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            res +=
                    intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1], grad_AB, i, n1);
        }
    }
    return res;
}
//================bat dau them code moi tu day================
__device__ inline void copyToResult(Point pPoint[MAXN], int n_arranged_point, Point *pPoint1, int &size) {
    size = n_arranged_point;
    for(int i = 0; i<size; i++){
        pPoint1[i].x = pPoint[i].x;
        pPoint1[i].y = pPoint[i].y;
    }
}
__device__ inline void removeDuplicatePoints(Point *points, int &n) {
    int new_n = 0;
    for (int i = 0; i < n; i++) {
        bool is_duplicate = false;
        for (int j = 0; j < new_n; j++) {
            if (points[i].x == points[j].x && points[i].y == points[j].y) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            points[new_n++] = points[i];
        }
    }

    n = new_n; // Update the original n value
}

__device__ inline double findMaxY(Point points[], int n) {
    double maxY = points[0].y;
    for (int i = 1; i < n; i++) {
        if (points[i].y > maxY) {
            maxY = points[i].y;
        }
    }
    return maxY;
}

__device__ inline double findMinY(Point points[], int n) {
    double minY = points[0].y;
    for (int i = 1; i < n; i++) {
        if (points[i].y < minY) {
            minY = points[i].y;
        }
    }
    return minY;
}

__device__ inline double findMaxX(Point points[], int n) {
    double maxX = points[0].y;
    for (int i = 1; i < n; i++) {
        if (points[i].x > maxX) {
            maxX = points[i].x;
        }
    }
    return maxX;
}

__device__ inline double findMinX(Point points[], int n) {
    double minX = points[0].x;
    for (int i = 1; i < n; i++) {
        if (points[i].x < minX) {
            minX = points[i].x;
        }
    }
    return minX;
}


__device__ inline void findPointsByY(Point point[], int size, double maxY, Point foundPoints[], int &foundPointsCount) {

    // Duyệt qua mảng
    for (int i = 0; i < size; i++) {
        // Nếu tọa độ y của phần tử tại vị trí i bằng với maxY
        if (point[i].y == maxY) {
            // Thêm phần tử vào mảng foundPoints
            foundPoints[foundPointsCount++] = point[i];
        }
    }

}

__device__ inline void findPointsByX(Point *points, int n, double x, Point result[], int &n_result) {


    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Nếu hoành độ của phần tử thứ i bằng x
        if (points[i].x == x) {
            // Lưu phần tử đó vào mảng result
            result[n_result++] = points[i];
        }
    }
}

__device__ inline void sortPointsByXDescending(Point *points, int n) {
    // Sắp xếp mảng theo hoành độ giảm dần
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của phần tử thứ j
            if (points[i].x < points[j].x) {
                // Đổi chỗ hai phần tử
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
}

__device__ inline void sortPointsByXAscending(Point *points, int n) {
    // Sắp xếp mảng theo hoành độ tăng dần
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của phần tử thứ j
            if (points[i].x > points[j].x) {
                // Đổi chỗ hai phần tử
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
}

__device__ inline void sortPointsByYDescending(Point *points, int n) {
    // Sắp xếp mảng theo tung độ giảm dần
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Nếu tung độ của phần tử thứ i nhỏ hơn tung độ của phần tử thứ j
            if (points[i].y < points[j].y) {
                // Đổi chỗ hai phần tử
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
}

__device__ inline void sortPointsByYAscending(Point *points, int n) {
    // Sắp xếp mảng theo tung độ tăng dần
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Nếu tung độ của phần tử thứ i lớn hơn tung độ của phần tử thứ j
            if (points[i].y > points[j].y) {
                // Đổi chỗ hai phần tử
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
}

__device__ inline void getPoints1(Point *points, int n, Point q1, Point qq1, Point *result, int &resultSize) {


    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của q1 và tung độ của phần tử thứ i lớn hơn tung độ của qq1
        if (points[i].x <= q1.x && points[i].y >= qq1.y) {
            // Lưu phần tử đó vào mảng result
            result[resultSize++] = points[i];
        }
    }
}

__device__ inline void getPoints2(Point *points, int n, Point qq2, Point q2, Point *result, int &resultSize) {

    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của q1 và tung độ của phần tử thứ i lớn hơn tung độ của qq1
        if (points[i].x <= qq2.x && points[i].y <= q2.y) {
            // Lưu phần tử đó vào mảng result
            result[resultSize++] = points[i];
        }
    }
}

__device__ inline void getPoints3(Point *points, int n, Point q3, Point qq3, Point *result, int &resultSize) {


    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của q1 và tung độ của phần tử thứ i lớn hơn tung độ của qq1
        if (points[i].x >= q3.x && points[i].y <= qq3.y) {
            // Lưu phần tử đó vào mảng result
            result[resultSize++] = points[i];
        }
    }

}

__device__ inline void getPoints4(Point *points, int n, Point qq4, Point q4, Point *result, int &resultSize) {


    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Nếu hoành độ của phần tử thứ i nhỏ hơn hoành độ của q1 và tung độ của phần tử thứ i lớn hơn tung độ của qq1
        if (points[i].x >= qq4.x && points[i].y >= q4.y) {
            // Lưu phần tử đó vào mảng result
            result[resultSize++] = points[i];
        }
    }
}

__device__ inline void findPointsWithYGreaterThan(Point points[], int n, double y, Point results[], int &n_result) {
    for (int i = 0; i < n; i++) {
        if (points[i].y > y) {
            results[n_result++] = points[i];
        }
    }
}

__device__ inline void findPointsWithYSmallerThan(Point points[], int n, double y, Point results[], int &n_result) {
    for (int i = 0; i < n; i++) {
        if (points[i].y < y) {
            results[n_result++] = points[i];
        }
    }
}

__device__ inline void findPointsWithXSmallerThan(Point points[], int n, double x, Point results[], int &n_result) {
    for (int i = 0; i < n; i++) {
        if (points[i].x < x) {
            results[n_result++] = points[i];
        }
    }
}

__device__ inline void findPointsWithXGreaterThan(Point points[], int n, double x, Point results[], int &n_result) {
    for (int i = 0; i < n; i++) {
        if (points[i].x > x) {
            results[n_result++] = points[i];
        }
    }
}

//Point* readPoints(const string& filename)
//{
//	ifstream file(filename);
//	Point* points;
//	double x, y;
//	int count = 0;
//	while (file >> x >> y)
//	{
//		points[count++] = { x, y };
//	}
//
//	return points;
//}
__device__ inline void calculateSquareDifference(Point points[], int n, double q1, double qq1, double results[]) {
    double tmp1[MAXN];
    double tmp2[MAXN];
    for (int i = 0; i < n; i++) {
        tmp1[i] = (points[i].x - q1) * (points[i].x - q1);
    }
    for (int i = 0; i < n; i++) {
        tmp2[i] = (points[i].y - qq1) * (points[i].y - qq1);
    }
    for (int i = 0; i < n; i++) {
        results[i] = tmp1[i] + tmp2[i];
    }
}

__device__ inline double findMax(double array[], int size) {
    double max = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

__device__ inline void findEqualElements(double key1[], int size, double maxset1, int results[]) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (key1[i] == maxset1) {
            results[count] = i;
            count++;
        }
    }
}

__device__ inline void findOHull1(Point set1[], int n_set1, Point q1, Point qq1, Point arrangedPoints[], int &n_arrangedPoints) {
    if (n_set1 == 0) {
        //n_arrangedPoints = 0;
        return;
    }
    double key1[MAXN];
    int n_key1 = n_set1;
    calculateSquareDifference(set1, n_set1, q1.x, qq1.y, key1);
    double maxset1 = findMax(key1, n_key1);
    int arr_index_newpoint1[100];
    findEqualElements(key1, n_key1, maxset1, arr_index_newpoint1);
    Point new_point1 = set1[arr_index_newpoint1[0]];
    Point new_set11[MAXN], new_set12[MAXN];
    int n_new_set11 = 0;
    int n_new_set12 = 0;
    findPointsWithYGreaterThan(set1, n_set1, new_point1.y, new_set11, n_new_set11);
    findPointsWithXSmallerThan(set1, n_set1, new_point1.x, new_set12, n_new_set12);

    findOHull1(new_set11, n_new_set11, q1, new_point1, arrangedPoints, n_arrangedPoints);
    arrangedPoints[n_arrangedPoints++] = new_point1;
    findOHull1(new_set12, n_new_set12, new_point1, qq1, arrangedPoints, n_arrangedPoints);

}

__device__ inline void findOHull2(Point set2[], int n_set2, Point q2, Point qq2, Point arrangedPoints[], int &n_arrangedPoints) {
    if (n_set2 == 0) {
        //n_arrangedPoints = 0;
        return;
    }
    double key2[MAXN];
    int n_key2 = n_set2;
    calculateSquareDifference(set2, n_set2, qq2.x, q2.y, key2);
    double maxset2 = findMax(key2, n_key2);
    int arr_index_newpoint2[200];

    findEqualElements(key2, n_key2, maxset2, arr_index_newpoint2);
    Point new_point2 = set2[arr_index_newpoint2[0]];
    Point new_set21[MAXN], new_set22[MAXN];
    int n_new_set21 = 0;
    int n_new_set22 = 0;
    findPointsWithXSmallerThan(set2, n_set2, new_point2.x, new_set21, n_new_set21);
    findPointsWithYSmallerThan(set2, n_set2, new_point2.y, new_set22, n_new_set22);

    findOHull2(new_set21, n_new_set21, q2, new_point2, arrangedPoints, n_arrangedPoints);
    arrangedPoints[n_arrangedPoints++] = new_point2;
    findOHull2(new_set22, n_new_set22, new_point2, qq2, arrangedPoints, n_arrangedPoints);

}

__device__ inline void findOHull3(Point set3[], int n_set3, Point q3, Point qq3, Point arrangedPoints[], int &n_arrangedPoints) {
    if (n_set3 == 0) {
        //n_arrangedPoints = 0;
        return;
    }
    double key3[MAXN];
    int n_key3 = n_set3;
    calculateSquareDifference(set3, n_set3, q3.x, qq3.y, key3);
    double maxset3 = findMax(key3, n_key3);
    int arr_index_newpoint3[100];
    findEqualElements(key3, n_key3, maxset3, arr_index_newpoint3);
    Point new_point3 = set3[arr_index_newpoint3[0]];
    Point new_set31[MAXN], new_set32[MAXN];
    int n_new_set31 = 0;
    int n_new_set32 = 0;
    findPointsWithYSmallerThan(set3, n_set3, new_point3.y, new_set31, n_new_set31);
    findPointsWithXGreaterThan(set3, n_set3, new_point3.x, new_set32, n_new_set32);

    findOHull3(new_set31, n_new_set31, q3, new_point3, arrangedPoints, n_arrangedPoints);
    arrangedPoints[n_arrangedPoints++] = new_point3;
    findOHull3(new_set32, n_new_set32, new_point3, qq3, arrangedPoints, n_arrangedPoints);
}

__device__ inline void findOHull4(Point set4[], int n_set4, Point q4, Point qq4, Point arrangedPoints[], int &n_arrangedPoints) {
    if (n_set4 == 0) {
        return;
    }
    double key4[MAXN];
    int n_key4 = n_set4;
    calculateSquareDifference(set4, n_set4, qq4.x, q4.y, key4);
    double maxset4 = findMax(key4, n_key4);
    int arr_index_newpoint4[100];
    findEqualElements(key4, n_key4, maxset4, arr_index_newpoint4);
    Point new_point4 = set4[arr_index_newpoint4[0]];
    Point new_set41[MAXN], new_set42[MAXN];
    int n_new_set41 = 0;
    int n_new_set42 = 0;
    findPointsWithXGreaterThan(set4, n_set4, new_point4.x, new_set41, n_new_set41);
    findPointsWithYGreaterThan(set4, n_set4, new_point4.y, new_set42, n_new_set42);

    findOHull4(new_set41, n_new_set41, q4, new_point4, arrangedPoints, n_arrangedPoints);
    arrangedPoints[n_arrangedPoints++] = new_point4;
    findOHull4(new_set42, n_new_set42, new_point4, qq4, arrangedPoints, n_arrangedPoints);

}

//================================


__device__ inline void Jarvis(Point* in_poly, int& n_poly) {
    Point arranged_points[1000];
    int n_arranged_point = 0;
    double maxY = findMaxY(in_poly, n_poly);
    double minY = findMinY(in_poly, n_poly);
    double maxX = findMaxX(in_poly, n_poly);
    double minX = findMinX(in_poly, n_poly);

    Point rightPoints[1000];
    int n_rightPoints = 0;
    findPointsByX(in_poly, n_poly, maxX, rightPoints, n_rightPoints);

    Point leftPoints[1000];
    int n_leftPoints = 0;
    findPointsByX(in_poly, n_poly, minX, leftPoints, n_leftPoints);

    Point topPoints[1000];
    int n_topPoints = 0;
    findPointsByY(in_poly, n_poly, maxY, topPoints, n_topPoints);

    Point bottomPoints[1000];
    int n_bottomPoints = 0;
    findPointsByY(in_poly, n_poly, minY, bottomPoints, n_bottomPoints);



    //top
    Point top[1000];
    int n_top = 0;
    if (n_topPoints == 1) {
        top[0] = topPoints[0];
        n_top = 1;
    } else {
        sortPointsByXAscending(topPoints, n_topPoints);
        n_top = 2;
        top[0] = topPoints[0];
        top[1] = topPoints[n_topPoints - 1];
    }


    //bottom

    Point bottom[1000];
    int n_bottom = 0;
    if (n_bottomPoints == 1) {
        bottom[0] = bottomPoints[0];
        n_bottom = 1;
    } else {
        sortPointsByXDescending(bottomPoints, n_bottomPoints);
        n_bottom = 2;
        bottom[0] = bottomPoints[0];
        bottom[1] = bottomPoints[n_bottomPoints - 1];
    }

    //right
    Point right[1000];
    int n_right = 0;
    if (n_rightPoints == 1) {
        right[0] = rightPoints[0];
        n_right = 1;
    } else {
        sortPointsByYDescending(rightPoints, n_rightPoints);
        n_right = 2;
        right[0] = rightPoints[0];
        right[1] = rightPoints[n_rightPoints - 1];
    }

    //left
    Point left[1000];
    int n_left = 0;
    if (n_leftPoints == 1) {
        left[0] = leftPoints[0];
        n_left = 1;
    } else {
        sortPointsByYAscending(leftPoints, n_leftPoints);
        n_left = 2;
        left[0] = leftPoints[0];
        left[1] = leftPoints[n_leftPoints - 1];
    }
    Point q1, qq1, q2, qq2, q3, qq3, q4, qq4;
    if (n_top == 1) {
        q1 = top[0];
        qq4 = top[0];
    } else {
        q1 = top[0];
        qq4 = top[1];
    }
    q4 = right[0];

    if (n_right == 1) {
        qq3 = right[0];
    } else {
        qq3 = right[1];
    }
    q3 = bottom[0];

    if (n_bottom == 1) {
        qq2 = bottom[0];
    } else {
        qq2 = bottom[1];
    }
    q2 = left[0];

    if (n_left == 1) {
        qq1 = left[0];
    } else {
        qq1 = left[1];
    }

    Point set1[1000], set2[1000], set3[1000], set4[1000];
    int n_set1 = 0;
    int n_set2 = 0;
    int n_set3 = 0;
    int n_set4 = 0;
    getPoints1(in_poly, n_poly, q1, qq1, set1, n_set1);
    getPoints2(in_poly, n_poly, qq2, q2, set2, n_set2);
    getPoints3(in_poly, n_poly, q3, qq3, set3, n_set3);
    getPoints4(in_poly, n_poly, qq4, q4, set4, n_set4);


    Point new_arranged_points[1000];
    new_arranged_points[0] = q1;
    new_arranged_points[1] = q1;
    int n_new_arranged_points = 2;
    findOHull1(set1, n_set1, q1, qq1, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq1;
    new_arranged_points[n_new_arranged_points++] = q2;
    findOHull2(set2, n_set2, q2, qq2, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq2;

    new_arranged_points[n_new_arranged_points++] = q3;
    findOHull3(set3, n_set3, q3, qq3, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq3;

    new_arranged_points[n_new_arranged_points++] = q4;
    findOHull4(set4, n_set4, q4, qq4, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq4;
    for (int i = 0; i < n_new_arranged_points; i++) {
        arranged_points[i] = new_arranged_points[i];
    }
    n_arranged_point = n_new_arranged_points;
    removeDuplicatePoints(arranged_points, n_arranged_point);
    copyToResult(arranged_points, n_arranged_point, in_poly, n_poly);

}

__device__ inline double intersectAreaPoly(Point* ps1, int n1, Point* ps2,
                                           int n2, double* grad_C) {
    Point polygon[MAXN];
    int n = n1 + n2, n_poly = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n - n1; j++) {
            if (point_same(ps1[i], ps2[j])) {
                for (int k = j; k < n - n1 - 1; k++) {
                    ps2[k] = ps2[k + 1];
                }
                n2--;
                break;
            }
        }
    }
    n_poly = n1 + n2;
    for (int i = 0; i < n_poly; i++) {
        if (i < n1) {
            polygon[i] = ps1[i];
        } else {
            polygon[i] = ps2[i - n1];
        }
    }

    Jarvis(polygon, n_poly);

    int polygon_to_pred_index[18] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int n_pred = 0;
    for (int i = 0; i < n_poly; i++) {
        for (int j = 0; j < n1; j++) {
            if (polygon[i].x == ps1[j].x && polygon[i].y == ps1[j].y) {
                polygon_to_pred_index[n_pred] = i;
                polygon_to_pred_index[n_pred + n1] = j;
                n_pred += 1;
                break;
            }
        }
    }
    if (n_pred == 0) {
        double polygon_area = fabs(area(polygon, n_poly));
        for (int i = 0; i < 18; i++) {
            grad_C[i] = 0.0;
        }
        return polygon_area;
    } else {
        double polygon_area =
                polygon_area_grad(polygon, n_poly, polygon_to_pred_index, n1, grad_C);
        if (polygon_area < 0) {
            for (int i = 0; i < 18; i++) {
                grad_C[i] = -grad_C[i];
            }
        }
        return fabs(polygon_area);
    }
}

// convex_find and get the polygon_index_box_index
__device__ inline void Jarvis_and_index(Point* in_poly, int& n_poly,
                                        int* points_to_convex_ind) {
    int n_input = n_poly;
    Point input_poly[20];
    for (int i = 0; i < n_input; i++) {
        input_poly[i].x = in_poly[i].x;
        input_poly[i].y = in_poly[i].y;
    }

    //=======================
    Point arranged_points[1000];
    int n_arranged_point = 0;
    double maxY = findMaxY(in_poly, n_poly);
    double minY = findMinY(in_poly, n_poly);
    double maxX = findMaxX(in_poly, n_poly);
    double minX = findMinX(in_poly, n_poly);

    Point rightPoints[1000];
    int n_rightPoints = 0;
    findPointsByX(in_poly, n_poly, maxX, rightPoints, n_rightPoints);

    Point leftPoints[1000];
    int n_leftPoints = 0;
    findPointsByX(in_poly, n_poly, minX, leftPoints, n_leftPoints);

    Point topPoints[1000];
    int n_topPoints = 0;
    findPointsByY(in_poly, n_poly, maxY, topPoints, n_topPoints);

    Point bottomPoints[1000];
    int n_bottomPoints = 0;
    findPointsByY(in_poly, n_poly, minY, bottomPoints, n_bottomPoints);



    //top
    Point top[1000];
    int n_top = 0;
    if (n_topPoints == 1) {
        top[0] = topPoints[0];
        n_top = 1;
    } else {
        sortPointsByXAscending(topPoints, n_topPoints);
        n_top = 2;
        top[0] = topPoints[0];
        top[1] = topPoints[n_topPoints - 1];
    }


    //bottom

    Point bottom[1000];
    int n_bottom = 0;
    if (n_bottomPoints == 1) {
        bottom[0] = bottomPoints[0];
        n_bottom = 1;
    } else {
        sortPointsByXDescending(bottomPoints, n_bottomPoints);
        n_bottom = 2;
        bottom[0] = bottomPoints[0];
        bottom[1] = bottomPoints[n_bottomPoints - 1];
    }

    //right
    Point right[1000];
    int n_right = 0;
    if (n_rightPoints == 1) {
        right[0] = rightPoints[0];
        n_right = 1;
    } else {
        sortPointsByYDescending(rightPoints, n_rightPoints);
        n_right = 2;
        right[0] = rightPoints[0];
        right[1] = rightPoints[n_rightPoints - 1];
    }

    //left
    Point left[1000];
    int n_left = 0;
    if (n_leftPoints == 1) {
        left[0] = leftPoints[0];
        n_left = 1;
    } else {
        sortPointsByYAscending(leftPoints, n_leftPoints);
        n_left = 2;
        left[0] = leftPoints[0];
        left[1] = leftPoints[n_leftPoints - 1];
    }
    Point q1, qq1, q2, qq2, q3, qq3, q4, qq4;
    if (n_top == 1) {
        q1 = top[0];
        qq4 = top[0];
    } else {
        q1 = top[0];
        qq4 = top[1];
    }
    q4 = right[0];

    if (n_right == 1) {
        qq3 = right[0];
    } else {
        qq3 = right[1];
    }
    q3 = bottom[0];

    if (n_bottom == 1) {
        qq2 = bottom[0];
    } else {
        qq2 = bottom[1];
    }
    q2 = left[0];

    if (n_left == 1) {
        qq1 = left[0];
    } else {
        qq1 = left[1];
    }

    Point set1[1000], set2[1000], set3[1000], set4[1000];
    int n_set1 = 0;
    int n_set2 = 0;
    int n_set3 = 0;
    int n_set4 = 0;
    getPoints1(in_poly, n_poly, q1, qq1, set1, n_set1);
    getPoints2(in_poly, n_poly, qq2, q2, set2, n_set2);
    getPoints3(in_poly, n_poly, q3, qq3, set3, n_set3);
    getPoints4(in_poly, n_poly, qq4, q4, set4, n_set4);


    Point new_arranged_points[1000];
    new_arranged_points[0] = q1;
    new_arranged_points[1] = q1;
    int n_new_arranged_points = 2;
    findOHull1(set1, n_set1, q1, qq1, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq1;
    new_arranged_points[n_new_arranged_points++] = q2;
    findOHull2(set2, n_set2, q2, qq2, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq2;

    new_arranged_points[n_new_arranged_points++] = q3;
    findOHull3(set3, n_set3, q3, qq3, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq3;

    new_arranged_points[n_new_arranged_points++] = q4;
    findOHull4(set4, n_set4, q4, qq4, new_arranged_points, n_new_arranged_points);
    new_arranged_points[n_new_arranged_points++] = qq4;
    for (int i = 0; i < n_new_arranged_points; i++) {
        arranged_points[i] = new_arranged_points[i];
    }
    n_arranged_point = n_new_arranged_points;
    removeDuplicatePoints(arranged_points, n_arranged_point);
    copyToResult(arranged_points, n_arranged_point, in_poly, n_poly);
//=================end modified code======================
    for (int i = 0; i < n_poly; i++) {
        for (int j = 0; j < n_input; j++) {
            if (point_same(in_poly[i], input_poly[j])) {
                points_to_convex_ind[i] = j;
                break;
            }
        }
    }
}

template <typename T>
__device__ inline float devrIoU(T const* const p, T const* const q,
                                T* point_grad, const int idx) {
    Point ps1[MAXN], ps2[MAXN];

    Point convex[MAXN];
    for (int i = 0; i < 9; i++) {
        convex[i].x = (double)p[i * 2];
        convex[i].y = (double)p[i * 2 + 1];
    }
    int n_convex = 9;
    int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    Jarvis_and_index(convex, n_convex, points_to_convex_ind);

    int n1 = n_convex;
    int n2 = 4;

    for (int i = 0; i < n1; i++) {
        ps1[i].x = (double)convex[i].x;
        ps1[i].y = (double)convex[i].y;
    }

    for (int i = 0; i < n2; i++) {
        ps2[i].x = (double)q[i * 2];
        ps2[i].y = (double)q[i * 2 + 1];
    }

    int polygon_index_box_index[18];
    for (int i = 0; i < n1; i++) {
        polygon_index_box_index[i] = i;
        polygon_index_box_index[i + n1] = i;
    }

    double grad_A[18] = {};
    double grad_AB[18] = {};
    double grad_C[18] = {};

    double inter_area = intersectAreaO(ps1, n1, ps2, n2, grad_AB);
    double S_pred =
            polygon_area_grad(ps1, n1, polygon_index_box_index, n1, grad_A);
    if (S_pred < 0) {
        for (int i = 0; i < n_convex * 2; i++) {
            grad_A[i] = -grad_A[i];
        }
    }
    double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;

    double iou = inter_area / union_area;
    double polygon_area = intersectAreaPoly(ps1, n1, ps2, n2, grad_C);

    //    printf("%d:live\n", idx);
    double rot_giou = iou - (polygon_area - union_area) / polygon_area;

    float grad_point_temp[18] = {};

    for (int i = 0; i < n_convex; i++) {
        int grad_point = points_to_convex_ind[i];
        grad_point_temp[2 * grad_point] =
                (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i] -
                        iou / union_area * grad_A[2 * i] -
                        1 / polygon_area * (grad_AB[2 * i] - grad_A[2 * i]) -
                        (union_area) / polygon_area / polygon_area * grad_C[2 * i]);
        grad_point_temp[2 * grad_point + 1] =
                (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i + 1] -
                        iou / union_area * grad_A[2 * i + 1] -
                        1 / polygon_area * (grad_AB[2 * i + 1] - grad_A[2 * i + 1]) -
                        (union_area) / polygon_area / polygon_area * grad_C[2 * i + 1]);
    }

    for (int i = 0; i < 9; i++) {
        point_grad[2 * i] = grad_point_temp[2 * i];
        point_grad[2 * i + 1] = grad_point_temp[2 * i + 1];
    }
    return (float)rot_giou;
}

template <typename T>
__global__ void convex_giou_cuda_kernel(const int ex_n_boxes,
                                        const int gt_n_boxes, const T* ex_boxes,
                                        const T* gt_boxes, T* point_grad) {
    CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
        const T* cur_box = ex_boxes + index * 18;
        const T* cur_gt_box = gt_boxes + index * 8;
        T* cur_grad = point_grad + index * 19;
        T giou = devrIoU(cur_box, cur_gt_box, cur_grad, threadIdx.x);
        cur_grad[18] = giou;
    }
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p) {
    double s1, s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);
    if (sig(s1) == 0 && sig(s2) == 0) return 2;
    if (sig(s2 - s1) == 0) return 0;
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}

__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b) {
    Point pp[MAXN];
    int m = 0;
    p[n] = p[0];
    for (int i = 0; i < n; i++) {
        if (sig(cross(a, b, p[i])) > 0) {
            pp[m] = p[i];
            m++;
        }
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
            lineCross(a, b, p[i], p[i + 1], pp[m]);
            m++;
        }
    }
    n = 0;
    for (int i = 0; i < m; i++) {
        if (!i || !(point_same(pp[i], pp[i - 1]))) {
            p[n] = pp[i];
            n++;
        }
    }

    while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d) {
    Point o(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0) return 0.0;
    if (s1 == -1) {
        Point* i = &a;
        Point* j = &b;
        swap1(i, j);
    }
    if (s2 == -1) {
        Point* i = &c;
        Point* j = &d;
        swap1(i, j);
    }
    Point p[10] = {o, a, b};
    int n = 3;

    polygon_cut(p, n, o, c);
    polygon_cut(p, n, c, d);
    polygon_cut(p, n, d, o);
    double res = area(p, n);
    if (s1 * s2 == -1) res = -res;
    return res;
}
__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2,
                                        int n2) {
    if (area(ps1, n1) < 0) reverse1(ps1, n1);
    if (area(ps2, n2) < 0) reverse1(ps2, n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    double res = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res;
}

template <typename T>
__device__ inline float devrIoU(T const* const p, T const* const q) {
    Point ps1[MAXN], ps2[MAXN];
    Point convex[MAXN];
    for (int i = 0; i < 9; i++) {
        convex[i].x = (double)p[i * 2];
        convex[i].y = (double)p[i * 2 + 1];
    }
    int n_convex = 9;
    int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    Jarvis_and_index(convex, n_convex, points_to_convex_ind);
    int n1 = n_convex;
    for (int i = 0; i < n1; i++) {
        ps1[i].x = (double)convex[i].x;
        ps1[i].y = (double)convex[i].y;
    }
    int n2 = 4;
    for (int i = 0; i < n2; i++) {
        ps2[i].x = (double)q[i * 2];
        ps2[i].y = (double)q[i * 2 + 1];
    }
    double inter_area = intersectAreaO(ps1, n1, ps2, n2);
    double S_pred = area(ps1, n1);
    double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
    double iou = inter_area / union_area;
    return (float)iou;
}

template <typename T>
__global__ void convex_iou_cuda_kernel(const int ex_n_boxes,
                                       const int gt_n_boxes, const T* ex_boxes,
                                       const T* gt_boxes, T* iou) {
    CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
        const T* cur_box = ex_boxes + index * 18;
        for (int i = 0; i < gt_n_boxes; i++) {
            iou[index * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
        }
    }
}
#endif  // CONVEX_IOU_CUDA_KERNEL_CUH
