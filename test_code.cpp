//#include <iostream>
//
//using namespace std;
//
//struct Point {
//    int x, y;
//};
//void removeDuplicatePoints(Point *points, int& n) {
//    int new_n = 0;
//    for (int i = 0; i < n; i++) {
//        bool is_duplicate = false;
//        for (int j = 0; j < new_n; j++) {
//            if (points[i].x == points[j].x && points[i].y == points[j].y) {
//                is_duplicate = true;
//                break;
//            }
//        }
//
//        if (!is_duplicate) {
//            points[new_n++] = points[i];
//        }
//    }
//
//    n = new_n; // Update the original n value
//}
//
//int main() {
//    Point points[] = {
//            {1, 2},
//            {2, 3},
//            {3, 4},
//            {3, 4},
//            {1, 2},
//            {2, 3},
//            {2, 3},
//            {2, 6},
//    };
//    int n = sizeof(points) / sizeof(points[0]);
//
//    removeDuplicatePoints(points, n);
//
//    for (int i = 0; i < n; i++) {
//        cout << points[i].x << " " << points[i].y << endl;
//    }
//
//    return 0;
//}
