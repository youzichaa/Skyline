
#ifndef RTDIRNODE_H__
#define RTDIRNODE_H__

#include <iostream>
#include <cstdlib>
#include "RtreeBlocks/rtnode.h"
#include "RtreeBlocks/point.h"
#include <vector>

class RTDirNode : public RTNode
{
public:
  std::vector<RTNode *> children;

  RTDirNode(int NodeCapacity, RTNode *parent, int level) : RTNode(NodeCapacity, parent, level)
  {
  }

  ~RTDirNode()
  {
    delete datas;
    delete data;
    delete parent;
  }

  RTNode *getChild(int index)
  {
    return children[index];
  }

  int findLeastOverlap(Rectangle *rectangle)
  {
    uint64_t overlap = -1;
    int sel = -1;
    for (int i = 0; i < usedSpace; i++)
    {
      RTNode *node = getChild(i);
      uint64_t ol = 0; // 用于记录每个孩子的datas数据与传入矩形的重叠面积之和
      for (int j = 0; j < node->usedSpace; j++)
      {
        // 将传入矩形与各个矩形重叠的面积累加到ol中，得到重叠的总面积
        ol += rectangle->intersectingArea(node->datas[j]);
      }
      if (ol < overlap)
      {
        overlap = ol; // 记录重叠面积最小的
        sel = i;      // 记录第几个孩子的索引
      }
      // 如果重叠面积相等则选择加入此Rectangle后面积增量更小的,如果面积增量还相等则选择自身面积更小的
      else if (ol == overlap)
      {
        uint64_t area1 = datas[i]->getUnionRectangle(rectangle)->getArea() - datas[i]->getArea();
        uint64_t area2 = datas[sel]->getUnionRectangle(rectangle)->getArea() - datas[sel]->getArea();

        if (area1 == area2)
        {
          sel = (datas[sel]->getArea() <= datas[i]->getArea()) ? sel : i;
        }
        else
        {
          sel = (area1 < area2) ? i : sel;
        }
      }
    }
    return sel;
  }

  int findLeastEnlargement(Rectangle *rectangle)
  {
    uint64_t area = -1;
    int sel = -1;

    for (int i = 0; i < usedSpace; i++)
    {
      // 增量enlargement = 包含（datas[i]里面存储的矩形与查找的矩形）的最小矩形的面积 -
      // datas[i]里面存储的矩形的面积
      double enlargement = datas[i]->getUnionRectangle(rectangle)->getArea() - datas[i]->getArea();
      if (enlargement < area)
      {
        area = enlargement; // 记录增量
        sel = i;            // 记录引起增量的【包含（datas[i]里面存储的矩形与查找的矩形）的最小矩形】里面的datas[i]的索引
      }
      else if (enlargement == area)
      {
        sel = (datas[sel]->getArea() < datas[i]->getArea()) ? sel : i;
      }
    }

    return sel;
  }

  RTNode *chooseLeaf(Rectangle *rectangle) override
  {
    int index = findLeastEnlargement(rectangle);
    insertIndex = index;
    return getChild(index)->chooseLeaf(rectangle);
  }

  std::vector<Point *> getLeaf()
  {
    std::vector<Point *> sky;
    if (!this->isLeaf())
    {
      for (int i = 0; i < this->usedSpace; i++)
      {
        std::vector<Point *> skyt = ((RTDirNode *)this->getChild(i))->getLeaf();
        for (int i = 0; i < skyt.size(); i++)
        {
          sky.push_back(skyt[i]);
        }
        skyt.clear();
      }
    }
    else
    {
      for (int i = 0; i < this->usedSpace; i++)
      {
        sky.push_back(((RTNode *)this)->getdata(i));
      }
    }
    return sky;
  }

  std::vector<RTDirNode *> splitIndex(RTNode *node)
  {
    int **group = quadraticSplit(node->getNodeRectangle());
    children.push_back(node); // 新加的
    node->parent = this;      // 新加的
    // 新建两个非叶子节点
    RTDirNode *index1 = new RTDirNode(Capacity, parent, level);
    RTDirNode *index2 = new RTDirNode(Capacity, parent, level);

    int *group1 = group[0];
    int *group2 = group[1];
    int len1 = group[2][0], len2 = group[2][1];
    // 为index1添加数据和孩子
    for (int i = 0; i < len1; i++)
    {
      index1->addData(datas[group1[i]]);
      index1->children.push_back(children[group1[i]]); // 新加的
      // 让index1成为其父节点
      children[group1[i]]->parent = index1; // 新加的
    }
    for (int i = 0; i < len2; i++)
    {
      index2->addData(datas[group2[i]]);
      index2->children.push_back(children[group2[i]]); // 新加的
      children[group2[i]]->parent = index2;            // 新加的
    }
    std::vector<RTDirNode *> rdT;
    rdT.push_back(index1);
    rdT.push_back(index2);
    return rdT;
  }

  bool insert(RTNode *node)
  {
    // 已用结点小于树的节点容量，不需分裂，只需插入以及调整树
    if (usedSpace < Capacity)
    {
      datas[usedSpace++] = node->getNodeRectangle();
      children.push_back(node); // 新加的
      node->parent = this;      // 新加的
      RTDirNode *parent = (RTDirNode *)getParent();
      if (parent != nullptr) // 不是根节点
      {
        parent->adjustTree(this, nullptr);
      }
      return false;
    }
    else
    { // 非叶子结点需要分裂
      std::vector<RTDirNode *> a = splitIndex(node);
      RTDirNode *n = a[0];
      RTDirNode *nn = a[1];

      if (isRoot())
      {
        // 新建根节点，层数加1
        RTDirNode *newRoot = new RTDirNode(Capacity, nullptr, level + 1);

        // 把两个分裂的结点n和nn添加到根节点
        newRoot->addData(n->getNodeRectangle());
        newRoot->addData(nn->getNodeRectangle());

        newRoot->children.push_back(n);
        newRoot->children.push_back(nn);

        // 设置两个分裂的结点n和nn的父节点
        n->parent = newRoot;
        nn->parent = newRoot;

        // 最后设置rtree的根节点
        // this->rt->setRoot(newRoot); // 新加的
      }
      else
      {
        // 如果不是根结点，向上调整树
        RTDirNode *p = (RTDirNode *)getParent();
        p->adjustTree(n, nn);
      }
    }
    return true;
  }

  void adjustTree(RTNode *node1)
  {
    insert(node1); // 插入新的结点
  }

  void adjustTree(RTNode *node1, RTNode *node2)
  {
    // 先要找到指向原来旧的结点（即未添加Rectangle之前）的条目的索引
    datas[insertIndex] = node1->getNodeRectangle(); // 先用node1覆盖原来的结点
    children[insertIndex] = node1;                  // 替换旧的结点

    // 还没到达根节点
    if (!isRoot())
    {
      RTDirNode *parent = (RTDirNode *)getParent();
      parent->adjustTree(this); // 向上调整直到根节点
    }
  }

  RTNode *clone() override
  {
    RTDirNode *tmp = new RTDirNode(this->Capacity, this->parent, this->level);
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
    for (int i = 0; i < this->children.size(); i++)
    {
      tmp->children.push_back(this->children[i]);
    }
    return tmp;
  }
};

#endif
