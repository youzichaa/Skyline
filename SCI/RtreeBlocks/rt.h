
#ifndef RT_H__
#define RT_H__

#include <iostream>
#include "RtreeBlocks/rectangle.h"
#include "RtreeBlocks/rtdatanode.h"
#include "RtreeBlocks/rtnode.h"
using namespace std;

// class RTNode;
class RT
{
public:
  // friend RTNode;
  RTNode *root;
  int nodeCapacity = -1;
  float fillFactor = -1;
  int leafNum;
  int nodeNum;
  int dimension;

  // RT()=default;
  
  // RT(int capacity, float fillFactor, int dimension);

  RT(int capacity, int dimension);

  ~RT();

  int getDimension();

  void setRoot(RTNode *node);

  void getRoot(RTNode *node);

  float getFillFactor();

  int getNodeCapacity();

  bool insert(Rectangle* rectangle);

  void printRT(RTNode *rootnode, int lev);
  // unordered_set<Point> RTAddPoint(RTNode root, unordered_map<Rectangle, unordered_set<Point>> res);
  // unordered_set<Point> RTCuaAddPoint(RTNode root, unordered_map<Rectangle, vector<Point>> res);
};

#endif
