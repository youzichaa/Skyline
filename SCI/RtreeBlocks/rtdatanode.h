
#ifndef RTDATANODE_H__
#define RTDATANODE_H__

#include <iostream>
#include <cstdlib>
#include "RtreeBlocks/rt.h"
#include "RtreeBlocks/rtdirnode.h"
#include <vector>
// class RTDirNode;
class RTDataNode : public RTNode
{
  
public:

  RTDataNode(int NodeCapacity, RTNode *parent) : RTNode(NodeCapacity, parent, 0)
  {
    
  }

  ~RTDataNode()
  {
    delete datas;
    delete data;
    delete parent;
  }

  std::vector<RTDataNode *> splitLeaf(Rectangle *rectangle)
  {
    int **group = quadraticSplit(rectangle);
    RTDataNode *l = new RTDataNode(Capacity, parent);
    RTDataNode *ll = new RTDataNode(Capacity, parent);

    int *group1 = group[0];
    int *group2 = group[1];
    int len1 = group[2][0], len2 = group[2][1];
    for (int i = 0; i < len1; i++)
    {
      l->addData(datas[group1[i]]);
    }
    for (int i = 0; i < len2; i++)
    {
      ll->addData(datas[group2[i]]);
    }
    std::vector<RTDataNode *> rdT;
    rdT.push_back(l);
    rdT.push_back(ll);
    return rdT;
  }

  bool insert(Rectangle *rectangle)
  {
    if (usedSpace < Capacity) // 已用节点小于节点容量
    {
      datas[usedSpace++] = rectangle;
      RTDirNode *parent = (RTDirNode *)getParent();

      if (parent != NULL)
        // 调整树，但不需要分裂节点，因为 节点小于节点容量，还有空间
        parent->adjustTree(this);
      return true;
    }
    // 超过结点容量
    else
    {
      std::vector<RTDataNode *> splitNodes = splitLeaf(rectangle);
      RTDataNode *l = splitNodes[0];
      RTDataNode *ll = splitNodes[1];

      if (isRoot())
      {
        // 根节点已满，需要分裂。创建新的根节点
        RTDirNode *rDirNode = new RTDirNode(Capacity, nullptr, level + 1);
        // rDirNode->rt->setRoot(rDirNode);
        // getNodeRectangle()返回包含结点中所有条目的最小Rectangle
        rDirNode->addData(l->getNodeRectangle());
        rDirNode->addData(ll->getNodeRectangle());

        ll->parent = rDirNode;
        l->parent = rDirNode;

        rDirNode->children.push_back(l);
        rDirNode->children.push_back(ll);
      }
      else
      { // 不是根节点
        RTDirNode *parentNode = (RTDirNode *)getParent();
        parentNode->adjustTree(l, ll);
      }
    }
    return true;
  }

  RTNode *chooseLeaf(Rectangle *rectangle) override 
  {
    insertIndex = usedSpace; 
    return this;
  }

  RTNode *clone() override 
  {
    RTDataNode *tmp = new RTDataNode(this->Capacity, this->parent);
    tmp->level = this->level;
    for (int i = 0; i < this->usedSpace; i++)
    {
      tmp->datas[i] = this->datas[i]->clone();
    }
    for (int i = 0; i < this->usedSpace; i++)
    {
      tmp->data[i] = this->data[i]->clone();
    }
    tmp->usedSpace = this->usedSpace;
    tmp->insertIndex = this->insertIndex;
    tmp->Factor = this->Factor;
    return tmp;
  }
};

#endif
