//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <algorithm>
//#include <cmath>
//#include <chrono>
//
//using namespace std;
//
//struct Point
//{
//    double x, y;
//};
//
//vector<Point> readPoints(const string &filename);
//vector<Point> findConvexHull(const vector<Point> &points);
//void findOHull1(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints);
//void findOHull2(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints);
//void findOHull3(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints);
//void findOHull4(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints);
//
//bool comparePoints(const Point &p1, const Point &p2)
//{
//    return (p1.x < p2.x) || (p1.x == p2.x && p1.y < p2.y);
//}
//
//int main()
//{
//    // t?o 9 ði?m
//    vector<Point> points = {
//        {1.0, 1.0},
//        {2.0, 2.0},
//        {3.0, 1.5},
//        {4.0, 5.0},
//        {5.0, 4.0},
//        {6.0, 3.0},
//        {7.0, 2.0},
//        {8.0, 3.5},
//        {9.0, 1.0}};
//
//    vector<Point> convexHull = findConvexHull(points);
//
//    cout << "Convex Hull Points:" << endl;
//    for (const auto &point : convexHull)
//    {
//        cout << "(" << point.x << ", " << point.y << ")" << endl;
//    }
//
//    return 0;
//}
//
//vector<Point> readPoints(const string &filename)
//{
//    ifstream file(filename);
//    vector<Point> points;
//    double x, y;
//
//    while (file >> x >> y)
//    {
//        points.push_back({x, y});
//    }
//
//    return points;
//}
//
//vector<Point> findConvexHull(const vector<Point> &points)
//{
//    vector<Point> convexHull;
//
//    auto maxY = max_element(points.begin(), points.end(), [](const Point &p1, const Point &p2)
//                            { return p1.y < p2.y; });
//    auto minY = min_element(points.begin(), points.end(), [](const Point &p1, const Point &p2)
//                            { return p1.y < p2.y; });
//    auto maxX = max_element(points.begin(), points.end(), [](const Point &p1, const Point &p2)
//                            { return p1.x < p2.x; });
//    auto minX = min_element(points.begin(), points.end(), [](const Point &p1, const Point &p2)
//                            { return p1.x < p2.x; });
//
//    auto rightPoints = max_element(points.begin(), points.end(), [maxX](const Point &p1, const Point &p2)
//                                   { return p1.x == maxX->x; });
//    auto leftPoints = max_element(points.begin(), points.end(), [minX](const Point &p1, const Point &p2)
//                                  { return p1.x == minX->x; });
//    auto topPoints = max_element(points.begin(), points.end(), [maxY](const Point &p1, const Point &p2)
//                                 { return p1.y == maxY->y; });
//    auto bottomPoints = max_element(points.begin(), points.end(), [minY](const Point &p1, const Point &p2)
//                                    { return p1.y == minY->y; });
//
//    Point q1, qq1, q2, qq2, q3, qq3, q4, qq4;
//
//    if (topPoints->x == bottomPoints->x)
//    {
//        topPoints = minmax_element(topPoints, bottomPoints, comparePoints).first;
//        bottomPoints = minmax_element(topPoints, bottomPoints, comparePoints).second;
//    }
//
//    if (rightPoints->y == leftPoints->y)
//    {
//        rightPoints = minmax_element(rightPoints, leftPoints, comparePoints).first;
//        leftPoints = minmax_element(rightPoints, leftPoints, comparePoints).second;
//    }
//
//    q1 = qq4 = *topPoints;
//    q4 = *rightPoints;
//    qq3 = *rightPoints;
//    q3 = *bottomPoints;
//    qq2 = *bottomPoints;
//    q2 = *leftPoints;
//    qq1 = *leftPoints;
//
//    // Separate to 4 sets of points
//    vector<Point> set1, set2, set3, set4;
//    for (const auto &point : points)
//    {
//        if (point.x <= q1.x && point.y >= qq1.y)
//            set1.push_back(point);
//        if (point.x <= qq2.x && point.y <= q2.y)
//            set2.push_back(point);
//        if (point.x >= q3.x && point.y <= qq3.y)
//            set3.push_back(point);
//        if (point.x >= qq4.x && point.y >= q4.y)
//            set4.push_back(point);
//    }
//
//    // Find o.convex by O-Quickhull
//    vector<Point> arrangedPoints;
//    arrangedPoints.push_back(q1);
//    findOHull1(set1, q1, qq1, arrangedPoints);
//    arrangedPoints.push_back(qq1);
//    arrangedPoints.push_back(q2);
//    findOHull2(set2, q2, qq2, arrangedPoints);
//    arrangedPoints.push_back(qq2);
//    arrangedPoints.push_back(q3);
//    findOHull3(set3, q3, qq3, arrangedPoints);
//    arrangedPoints.push_back(qq3);
//    arrangedPoints.push_back(q4);
//    findOHull4(set4, q4, qq4, arrangedPoints);
//    arrangedPoints.push_back(qq4);
//
//    return arrangedPoints;
//}
//
//void findOHull1(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints)
//{
//    if (set.empty())
//    {
//        return;
//    }
//
//    double maxKey = -1;
//    Point newPoint;
//
//    for (const auto &point : set)
//    {
//        double key = pow((point.x - q.x), 2) + pow((point.y - qq.y), 2);
//        if (key > maxKey)
//        {
//            maxKey = key;
//            newPoint = point;
//        }
//    }
//
//    vector<Point> newSet11, newSet12;
//    for (const auto &point : set)
//    {
//        if (point.y > newPoint.y)
//            newSet11.push_back(point);
//        if (point.x < newPoint.x)
//            newSet12.push_back(point);
//    }
//
//    findOHull1(newSet11, q, newPoint, arrangedPoints);
//    arrangedPoints.push_back(newPoint);
//    findOHull1(newSet12, newPoint, qq, arrangedPoints);
//}
//
//void findOHull2(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints)
//{
//    if (set.empty())
//    {
//        return;
//    }
//
//    double maxKey = -1;
//    Point newPoint;
//
//    for (const auto &point : set)
//    {
//        double key = pow((point.x - qq.x), 2) + pow((point.y - q.y), 2);
//        if (key > maxKey)
//        {
//            maxKey = key;
//            newPoint = point;
//        }
//    }
//
//    vector<Point> newSet21, newSet22;
//    for (const auto &point : set)
//    {
//        if (point.x < newPoint.x)
//            newSet21.push_back(point);
//        if (point.y < newPoint.y)
//            newSet22.push_back(point);
//    }
//
//    findOHull2(newSet21, q, newPoint, arrangedPoints);
//    arrangedPoints.push_back(newPoint);
//    findOHull2(newSet22, newPoint, qq, arrangedPoints);
//}
//
//void findOHull3(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints)
//{
//    if (set.empty())
//    {
//        return;
//    }
//
//    double maxKey = -1;
//    Point newPoint;
//
//    for (const auto &point : set)
//    {
//        double key = pow((point.x - q.x), 2) + pow((point.y - qq.y), 2);
//        if (key > maxKey)
//        {
//            maxKey = key;
//            newPoint = point;
//        }
//    }
//
//    vector<Point> newSet31, newSet32;
//    for (const auto &point : set)
//    {
//        if (point.y < newPoint.y)
//            newSet31.push_back(point);
//        if (point.x > newPoint.x)
//            newSet32.push_back(point);
//    }
//
//    findOHull3(newSet31, q, newPoint, arrangedPoints);
//    arrangedPoints.push_back(newPoint);
//    findOHull3(newSet32, newPoint, qq, arrangedPoints);
//}
//
//void findOHull4(const vector<Point> &set, const Point &q, const Point &qq, vector<Point> &arrangedPoints)
//{
//    if (set.empty())
//    {
//        return;
//    }
//
//    double maxKey = -1;
//    Point newPoint;
//
//    for (const auto &point : set)
//    {
//        double key = pow((point.x - qq.x), 2) + pow((point.y - q.y), 2);
//        if (key > maxKey)
//        {
//            maxKey = key;
//            newPoint = point;
//        }
//    }
//
//    vector<Point> newSet41, newSet42;
//    for (const auto &point : set)
//    {
//        if (point.x > newPoint.x)
//            newSet41.push_back(point);
//        if (point.y > newPoint.y)
//            newSet42.push_back(point);
//    }
//
//    findOHull4(newSet41, q, newPoint, arrangedPoints);
//    arrangedPoints.push_back(newPoint);
//    findOHull4(newSet42, newPoint, qq, arrangedPoints);
//}
