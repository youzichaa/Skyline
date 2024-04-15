

#include "RtreeBlocks/rtree.h"

using namespace std;
using namespace sci;


 RTree::RTree(int capacity, float fillFactor, int dimension)
  {
    fillFactor = fillFactor;
    nodeCapacity = capacity;
    dimension = dimension;
    root = new RTDataNode(this, (RTNode *)NULL);
  }

  RTree::RTree(int capacity, int dimension)
  {
    this->nodeCapacity = capacity;
    this->fillFactor = 0.4f;
    this->dimension = dimension;
    this->root = new RTDataNode(this, (RTNode *)NULL);
  }

  RTree::~RTree(){
    delete root;
  }

  int RTree::getDimension()
  {
    return this->dimension;
  }

  void RTree::setRoot(RTNode *root)
  {
    this->root = root;
  }

  float RTree::getFillFactor()
  {
    return this->fillFactor;
  }

  int RTree::getNodeCapacity()
  {
    return this->nodeCapacity;
  }

  bool RTree::insert(Rectangle* rectangle)
  {
    RTDataNode* leaf = this->root->chooseLeaf(rectangle);
    return leaf->insert(rectangle);
  }