
#ifndef RECTANGLE_H__
#define RECTANGLE_H__

#include "stdio.h"
#include "utils/emp-tool.h"
#include "RtreeBlocks/point.h"
#include <cmath>
#include <algorithm>
#ifndef Min
  #define Min std::min
#endif //Min
#ifndef Max
  #define Max std::max
#endif //Max


class Rectangle {
public:
  Point *low;
  Point *high;
  // Rectangle ();

  Rectangle(Point *p1, Point *p2) {
    this->low = p1->clone();
    this->high = p2->clone();
  }

  ~Rectangle() {
    delete low;
    delete high;
  }

  Rectangle* clone() {
    Point *p1 = (Point*) low->clone();
    Point *p2 = (Point*) high->clone();
    return new Rectangle(p1, p2);
  }

  int getDimension() {
    return low->getDimension();
  }

  Point* getLow() {
    return low;
  }

  Point* getHigh() {
    return high;
  }

  uint64_t getArea() {
    uint64_t area = 1;
    for (int i = 0; i < getDimension(); i++) {
        area *= high->get(i) - low->get(i);
    }
    return area;
  }


  Rectangle* getUnionRectangle(Rectangle* rectangle) {
    int size = getDimension();
    uint64_t *min = new uint64_t[size];
    uint64_t *max = new uint64_t[size];

    for (int i = 0; i < size; i++) {
        min[i] = Min(low->get(i), rectangle->low->get(i));
        max[i] = Max(high->get(i), rectangle->high->get(i));
    }
    return new Rectangle(new Point(min, size), new Point(max, size));
  }

  Rectangle* getUnionRectangle(Rectangle** rectangles, int dim) {
    Rectangle *r0 = (Rectangle*) rectangles[0]->clone();
    for (int i = 1; i < dim; i++) {
      r0 = r0->getUnionRectangle(rectangles[i]);
    }

    return r0; 
  }

  bool isIntersection(Rectangle *rectangle) {
    for (int i = 0; i < getDimension(); i++) {
      if (low->get(i) >= rectangle->high->get(i) || high->get(i) <= rectangle->low->get(i)) {
        return false; 
      }
    }
    return true;
  }

  uint64_t intersectingArea(Rectangle *rectangle) {
    if (!isIntersection(rectangle)) 
    {
      return 0;
    }
    uint64_t ret = 1;
    for (int i = 0; i < rectangle->getDimension(); i++) {
      uint64_t l1 = this->low->get(i);
      uint64_t h1 = this->high->get(i);
      uint64_t l2 = rectangle->low->get(i);
      uint64_t h2 = rectangle->high->get(i);
      ret *= Min(h1, h2) - Max(l1, l2);
    }
    return ret;
  }

  bool enclosure(Rectangle *rectangle) {
    for (int i = 0; i < getDimension(); i++) {
      if (rectangle->low->get(i) < low->get(i) || rectangle->high->get(i) > high->get(i))
        return false;
    }
    return true;
  }

  bool enclosure(Point *point) {
    for (int i = 0; i < getDimension(); i++) {
      if (point->get(i) < low->get(i) || point->get(i) > high->get(i))
        return false;
    }
    return true;
  }

  bool enclosureCua(Point *point) {
    for (int i = 0; i < getDimension(); i++) {
      if (point->get(i) <= low->get(i) || point->get(i) >= high->get(i))
        return false;
    }
    return true;
  }
};

#endif
