
#ifndef RTNODE_H__
#define RTNODE_H__

#include <iostream>
#include <math.h>
#include <cstdlib>
#include "RtreeBlocks/rectangle.h"
#include "RtreeBlocks/point.h"

class RTNode
{
public:
  int level;
  Rectangle **datas;
  Point **data;
  RTNode *parent;
  int usedSpace;
  int insertIndex;
  int Capacity;
  float Factor;
  // RTNode()=default;

  RTNode(int NodeCapacity, RTNode *parent, int level)
  {
    this->parent = parent;
    this->level = level;
    this->Capacity = NodeCapacity;
    datas = new Rectangle *[NodeCapacity + 1]; // 多出的一个用于结点分裂
    data = new Point *[NodeCapacity];
    usedSpace = 0;
  }

  ~RTNode()
  {
    delete datas;
    delete data;
    delete parent;
  }

  void setRT(float fillFactor)
  {
    this->Factor = fillFactor;
  }

  bool isLeaf()
  {
    return (level == 0);
  }

  bool isIndex()
  {
    return (level != 0);
  }

  bool isRoot()
  {
    return (parent == nullptr);
  }

  RTNode *getParent()
  {
    return parent;
  }


  Rectangle *getdatas(int i)
  {
    return datas[i];
  }

  Point *getdata(int i)
  {
    return data[i];
  }

  int getSpace()
  {
    return usedSpace;
  }

  void addData(Rectangle *rectangle)
  {
    datas[usedSpace++] = rectangle;
  }

  void addDataP(Rectangle *rectangle, Point *p)
  {
    datas[usedSpace] = rectangle;
    data[usedSpace] = p;
    usedSpace++;
  }

  Rectangle *getNodeRectangle()
  {
    if (usedSpace > 0)
    {
      return (new Rectangle(nullptr, nullptr))->getUnionRectangle(datas, usedSpace);
    }
    else
    {
      return new Rectangle(nullptr, nullptr);
    }
  }

  int **quadraticSplit(Rectangle *rectangle)
  {
    datas[usedSpace] = rectangle;

    int total = usedSpace + 1;

    int *mask = new int[total];
    for (int i = 0; i < total; i++)
    {
      mask[i] = 1;
    }

    // 分裂后每个组只是有total/2个条目
    int c = total / 2 + 1;
    // 每个结点最小条目个数
    int minNodeSize = (int)((this->Capacity) * (this->Factor));
    // 至少有两个
    if (minNodeSize < 2)
      minNodeSize = 2;

    // 记录没有被检查的条目的个数
    int rem = total;

    int *group1 = new int[c]; // 记录分配的条目的索引
    int *group2 = new int[c]; // 记录分配的条目的索引
    // 跟踪被插入每个组的条目的索引
    int i1 = 0, i2 = 0;

    int *seed = pickSeeds();

    group1[i1++] = seed[0];
    group2[i2++] = seed[1];
    rem -= 2;
    mask[group1[0]] = -1;
    mask[group2[0]] = -1;

    while (rem > 0)
    {
      // 将剩余的所有条目全部分配到group1组中，算法终止
      if (minNodeSize - i1 == rem)
      {
        for (int i = 0; i < total; i++) // 总共rem个
        {
          if (mask[i] != -1) // 还没有被分配
          {
            if (i1 == c - 1)
              break;
            group1[i1++] = i;
            mask[i] = -1;
            rem--;
          }
        }
        // 将剩余的所有条目全部分配到group2组中，算法终止
      }
      else if (minNodeSize - i2 == rem)
      {
        for (int i = 0; i < total; i++) // 总共rem个
        {
          if (mask[i] != -1) // 还没有被分配
          {
            if (i2 == c - 1)
              break;
            group2[i2++] = i;
            mask[i] = -1;
            rem--;
          }
        }
      }
      else
      {
        // 求group1中所有条目的最小外包矩形
        Rectangle *mbr1 = (Rectangle *)datas[group1[0]]->clone();
        for (int i = 1; i < i1; i++)
        {
          mbr1 = mbr1->getUnionRectangle(datas[group1[i]]);
        }
        // 求group2中所有条目的外包矩形
        Rectangle *mbr2 = (Rectangle *)datas[group2[0]]->clone();
        for (int i = 1; i < i2; i++)
        {
          mbr2 = mbr2->getUnionRectangle(datas[group2[i]]);
        }

        // 找出下一个进行分配的条目
        uint64_t dif = 0;
        uint64_t areaDiff1 = 0, areaDiff2 = 0;
        int sel = -1;
        for (int i = 0; i < total; i++)
        {
          if (mask[i] != -1) // 还没有被分配的条目
          {
            // 计算把每个条目加入每个组之后面积的增量，选择两个组面积增量差最大的条目索引
            Rectangle *a = mbr1->getUnionRectangle(datas[i]);
            areaDiff1 = a->getArea() - mbr1->getArea();

            Rectangle *b = mbr2->getUnionRectangle(datas[i]);
            areaDiff2 = b->getArea() - mbr2->getArea();

            uint64_t tmp = 0;
            if (areaDiff1 > areaDiff2)
            {
              tmp = areaDiff1 - areaDiff2;
            }
            else
            {
              tmp = areaDiff2 - areaDiff1;
            }
            if (tmp > dif)
            {
              dif = tmp;
              sel = i;
            }
          }
        }
        int t = 0;
        if (areaDiff1 < areaDiff2) // 先比较面积增量
        {
          //                    group1[i1++] = sel;
        }
        else if (areaDiff1 > areaDiff2)
        {
          t = 1;
          //                    group2[i2++] = sel;
        }
        else if (mbr1->getArea() < mbr2->getArea()) // 再比较自身面积
        {
          //                    group1[i1++] = sel;
        }
        else if (mbr1->getArea() > mbr2->getArea())
        {
          t = 1;
          //                    group2[i2++] = sel;
        }
        else if (i1 < i2) // 最后比较条目个数
        {
          //                    group1[i1++] = sel;
        }
        else if (i1 > i2)
        {
          t = 1;
          //                    group2[i2++] = sel;
        }
        else
        {
          //                    group1[i1++] = sel;
        }
        if (t == 0)
        {
          if (i1 == c - 1)
          {
            group2[i2++] = sel;
          }
          else
          {
            group1[i1++] = sel;
          }
        }
        else
        {
          if (i2 == c - 1)
          {
            group1[i1++] = sel;
          }
          else
          {
            group2[i2++] = sel;
          }
        }
        mask[sel] = -1;
        rem--;
      }
    } // end while

    int **ret = new int *[3];
    ret[0] = new int[i1];
    ret[1] = new int[i2];
    ret[2] = new int[2];

    for (int i = 0; i < i1; i++)
    {
      ret[0][i] = group1[i];
    }
    for (int i = 0; i < i2; i++)
    {
      ret[1][i] = group2[i];
    }
    ret[2][0] = i1;
    ret[2][1] = i2;
    return ret;
  }

  int *pickSeeds()
  {
    uint64_t inefficiency = 0;
    int *res = new int[2];
    res[0] = 0, res[1] = 0;

    // 两个for循环对任意两个条目E1和E2进行组合
    for (int i = 0; i < usedSpace; i++)
    {
      for (int j = i + 1; j <= usedSpace; j++) // 注意此处的j值
      {
        Rectangle *rectangle = datas[i]->getUnionRectangle(datas[j]);
        uint64_t d = rectangle->getArea() - datas[i]->getArea() - datas[j]->getArea();
        if (d > inefficiency)
        {
          inefficiency = d;
          res[0] = i;
          res[1] = j;
        }
      }
    }
    return res;
  }

  virtual RTNode *chooseLeaf(Rectangle *rectangle) = 0;
  virtual RTNode *clone() = 0;
  // virtual RTNode *findLeaf(Rectangle *rectangle);
};

#endif
