//
// Created by Nguyen Xuan on 11/23/2023.
//
#include <iostream>

struct Point {
    int x, y;
};

void removeDuplicatePoints(Point* points, int& n, Point* res, int& n_res) {
    // Khởi tạo biến đếm số phần tử duy nhất
    int outputSize = 0;

    // Duyệt qua mảng points
    for (int i = 0; i < n; i++) {
        // Kiểm tra xem điểm points[i] có phải là phần tử duy nhất hay không
        bool isUnique = true;
        for (int j = 0; j < n; j++) {
            if (points[i].x == points[j].x && points[i].y == points[j].y ) {
                isUnique = false;
                break;
            }
        }

        // Nếu points[i] là phần tử duy nhất thì thêm nó vào mảng output
        if (isUnique) {
            points[n_res++] = points[i];
        }
    }
//created by dotx
}

int main() {
    // Khởi tạo mảng points
    Point points[] = {{1, 2},
                      {2, 3},
                      {3, 4},
                      {1, 2},
                      {4, 5},
                      {3, 4}};
    int n = sizeof(points) / sizeof(points[0]);
    Point res[n];
    int n_res = 0;
    // Gọi hàm removeDuplicatePoints()
    removeDuplicatePoints(points, n, res, n_res);

    // In các phần tử duy nhất trong mảng points
    for (int i = 0; i < n_res; i++) {
        std::cout << res[i].x << " " << res[i].y << std::endl;
    }

    return 0;
}
