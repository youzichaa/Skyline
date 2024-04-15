

#include "RtreeBlocks/rt.h"
#include "RtreeBlocks/rectangle.h"
#include "RtreeBlocks/rtdatanode.h"
#include "RtreeBlocks/rtnode.h"
#include <cstdlib>

using namespace std;
using namespace sci;


//  RT::RT(int capacity, float fillFactor, int dimension)
//   {
//     this->fillFactor = fillFactor;
//     this->nodeCapacity = capacity;
//     this->dimension = dimension;
//     this->root = new RTDataNode(capacity, (RTNode *)nullptr);
//     this->root->setRT(this, fillFactor);
//   }

  RT::RT(int capacity, int dimension)
  {
    this->nodeCapacity = capacity;
    this->fillFactor = 0.4f;
    this->dimension = dimension;
    this->root = new RTDataNode(capacity, (RTNode *)nullptr);
  }

  RT::~RT(){
    delete root;
  }

  int RT::getDimension()
  {
    return this->dimension;
  }

  void RT::setRoot(RTNode *node)
  {
    this->root = node;
  }

  void RT::getRoot(RTNode *node)
  {
    
    if (node->isRoot())
    {
      this->root = node;
    } 
    else
    {
      RTNode *parentNode = (RTNode *)node->getParent();
      getRoot(parentNode);
    }
  }

  float RT::getFillFactor()
  {
    return this->fillFactor;
  }

  int RT::getNodeCapacity()
  {
    return this->nodeCapacity;
  }

  bool RT::insert(Rectangle* rectangle)
  {
    RTDataNode* leaf = (RTDataNode*)(this->root)->chooseLeaf(rectangle);
    return leaf->insert(rectangle);
  }

  void RT::printRT(RTNode *rootnode, int lev)
  {
    if (rootnode == nullptr)
      throw std::invalid_argument("Node cannot be null.");
    if (!rootnode->isLeaf())
    {
      for (int i = 0; i < rootnode->usedSpace; i++)
      {
        for (int j = 0; j < lev - rootnode->level + 1; j++)
        {
          cout << "+";
        }
        // cout << root.datas[i] << endl;
        cout << rootnode->datas[i]->getLow()->get(0) << "," << rootnode->datas[i]->getLow()->get(1) << ";";
        cout << rootnode->datas[i]->getHigh()->get(0) << "," << rootnode->datas[i]->getHigh()->get(1) << std::endl;
        printRT(((RTDirNode *)(rootnode))->getChild(i), lev);
      }
    }
    else
    {
      // cout << "point_num:" << root.usedSpace << endl;
      // cout << "leaf:" << rootnode.level << endl;
      for (int i = 0; i < rootnode->usedSpace; i++)
      {
        for (int j = 0; j < lev - rootnode->level; j++)
        {
          std::cout << "=";
        }
        std::cout << rootnode->datas[i]->getLow()->get(0) << "," << rootnode->datas[i]->getLow()->get(1) << ";";
        std::cout << rootnode->datas[i]->getHigh()->get(0) << "," << rootnode->datas[i]->getHigh()->get(1) << ";";
        std::cout << rootnode->data[i]->get(0) << "," << rootnode->data[i]->get(1);
        if (i == (rootnode->usedSpace - 1))
        {
          std::cout << ";1" << std::endl;
        }
        else
        {
          std::cout << ";0" << std::endl;
        }
      }
    }
  }