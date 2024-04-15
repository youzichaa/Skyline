
#ifndef POINT_H__
#define POINT_H__

#include "stdio.h"
#include "utils/emp-tool.h"
#include <cmath>


class Point {
public:
  int size;
  uint64_t *data;
  // Point();

  Point(uint64_t *dataT, int size) {
    this->size = size;
    this->data = new uint64_t[size];
    memcpy(this->data, dataT, size * sizeof(uint64_t));
  }

  Point(int size) {
    this->data = new uint64_t[size];
    memset(this->data, 0, size * sizeof(uint64_t));
  }

  ~Point();

  Point* clone() {
    uint64_t *copy = new uint64_t[this->size];
    memcpy(copy, this->data, this->size * sizeof(uint64_t));
    return new Point(copy, this->size);
  }

  void set(int index, uint64_t val) {
    this->data[index] = val;
  }

  int getDimension() {
    return this->size;
  }

  uint64_t get(int index) {
    return data[index];
  }

};

#endif
