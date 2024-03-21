/*
Authors: Deevashwer Rathee, Mayank Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions
:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "BuildingBlocks/aux-protocols.h"
#include "NonLinear/argmax.h"
#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include "OT/np.h"
#include "OT/ot.h"
#include "utils/prp.h"
#include <iostream>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <smmintrin.h>
#include <ctime>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <pthread.h>
using namespace sci;
using namespace std;

int party, port = 8000, dim = 1 << 16;
string address = "127.0.0.1";
int THs = 32;
NetIO *Iot[32];
OTPack<NetIO> *Otpackt[32];
AuxProtocols *Auxt[32];
LinearOT *Prodt[32];
PRG128 prg;
CRH crh;
int n = 9000;
int m = 2;
//"/small-correlated.txt";"/small-uniformly-distributed.txt";"/small-anti-correlated.txt";
string filename = "/small-correlated.txt";
string dataname = "corr-";
string data_path = "./data/input=10000/size=" + to_string(n) + filename;
unordered_map<uint32_t, vector<uint32_t>> G;
unordered_map<uint32_t, unordered_set<uint32_t>> Skyline;
unordered_set<uint32_t> H;
uint32_t **SS_G;
uint32_t * SS_C;
int *SS_L;
int *SS_L_itr;
uint64_t **SS_Skyline;
uint64_t **SS_Skyline_itr;
uint32_t *SS_Con;

int MAX = 14;
int lambda = 32;
uint64_t mask = (1ULL << lambda) - 1;
double com1, com2, plain_process, ss_upload, ss_process, ss_read, ss_dummy, ss_mask, ss_selectSky, ss_select;

void SS_Print(vector<uint32_t> H)
{
  int len = H.size();
  uint32_t *x = new uint32_t[len];
  copy(H.begin(), H.end(), x);
  if (party == ALICE)
  {
    Iot[0]->send_data(x, len * sizeof(uint32_t));
  }
  else
  {
    uint32_t *x0 = new uint32_t[len];
    Iot[0]->recv_data(x0, len * sizeof(uint32_t));
    cout << "[";
    for (int j = 0; j < len - 1; j++)
    {
      uint32_t t = (x[j] + x0[j]) & mask;
      cout << +t << ",";
    }
    cout << +((x[len - 1] + x0[len - 1]) & mask) << "]";
    delete[] x0;
  }
}

void SS_Print(uint32_t len, uint32_t *x)
{
  if (party == ALICE)
  {
    Iot[0]->send_data(x, len * sizeof(uint32_t));
    uint32_t *x0 = new uint32_t[len];
    Iot[0]->recv_data(x0, len * sizeof(uint32_t));
    cout << "[";
    for (int j = 0; j < len - 1; j++)
    {
      uint32_t t = (x[j] + x0[j]) & mask;
      cout << +t << ",";
    }
    cout << +((x[len - 1] + x0[len - 1]) & mask) << "]";
    delete[] x0;
  }
  else
  {
    uint32_t *x0 = new uint32_t[len];
    Iot[0]->recv_data(x0, len * sizeof(uint32_t));
    Iot[0]->send_data(x, len * sizeof(uint32_t));
    cout << "[";
    for (int j = 0; j < len - 1; j++)
    {
      uint32_t t = (x[j] + x0[j]) & mask;
      cout << +t << ",";
    }
    cout << +((x[len - 1] + x0[len - 1]) & mask) << "]";
    delete[] x0;
  }
}
void SS_Print(int len, uint64_t *x)
{
  // uint64_t *tt = new uint64_t[m];
  // tt[0] = 7000;
  // tt[1] = 5000;
  uint64_t *x0 = new uint64_t[len];
  if (party == ALICE)
  {
    Iot[0]->send_data(x, len * sizeof(uint64_t));
    Iot[0]->recv_data(x0, len * sizeof(uint64_t));
    for (int j = 0; j < len; j++)
    {
      x0[j] = (x[j] + x0[j]) & mask;
    }
  }
  else
  {
    Iot[0]->recv_data(x0, len * sizeof(uint64_t));
    Iot[0]->send_data(x, len * sizeof(uint64_t));
    for (int j = 0; j < len; j++)
    {
      x0[j] = (x[j] + x0[j]) & mask;
    }
  }
  uint64_t *s = new uint64_t[m];
  cout << "[";
  for (int i = 0; i < len - 1; i++)
  {
    s[0] = x0[i] & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      x0[i] = (x0[i] - s[j - 1]) >> MAX;
      s[j] = x0[i] & ((1ULL << MAX) - 1);
    }
    for (int j = m - 1; j >= 1; j--)
    {
      // cout << labs(s[j]-tt[0]) << ",";
      cout << s[j] << ",";
    }
    // cout << labs(s[0]-tt[1]) << ";";
    cout << s[0] << ";";
  }
  s[0] = x0[len - 1] & ((1ULL << MAX) - 1);
  for (int j = 1; j < m; j++)
  {
    x0[len - 1] = (x0[len - 1] - s[j - 1]) >> MAX;
    s[j] = x0[len - 1] & ((1ULL << MAX) - 1);
  }
  for (int j = m - 1; j >= 1; j--)
  {
    // cout << labs(s[j]-tt[0]) << ",";
      cout << s[j] << ",";
    }
  // cout << labs(s[0]-tt[1]) << ";";
  cout << s[0] << "]";
  delete[] s;
  delete[] x0;
}

void PrintSKyline(uint64_t *x)
{
  if (party == ALICE)
  {
    uint64_t *x0 = new uint64_t[m];
    Iot[0]->send_data(x, m * sizeof(uint64_t));
    Iot[0]->recv_data(x0, m * sizeof(uint64_t));
    for (int i = 0; i < m; i++)
    {
      x0[i] = ((x[i] + x0[i]) & mask);
    }
    cout<<"===="<<x0[0]<<","<<x0[1]<<"===="<<endl;
    delete[] x0;
  }
  else
  {
    uint64_t *x0 = new uint64_t[m];
    Iot[0]->recv_data(x0, m * sizeof(uint64_t));
    for (int i = 0; i < m; i++)
    {
      x0[i] = ((x[i] + x0[i]) & mask);
    }
    cout<<"===="<<x0[0]<<","<<x0[1]<<"===="<<endl;
    delete[] x0;
    Iot[0]->send_data(x, m * sizeof(uint64_t));
  }
}

void PrintSKyline(int th, uint64_t x)
{
  uint64_t x0 = 0;
  if (party == ALICE)
  {
    Iot[th]->send_data(&x, sizeof(uint64_t));
    Iot[th]->recv_data(&x0, sizeof(uint64_t));
    x0 = (x + x0) & mask;
    cout<<"===="<<x0<<"===="<<endl;
  }
  else
  {
    Iot[th]->recv_data(&x0, sizeof(uint64_t));
    Iot[th]->send_data(&x, sizeof(uint64_t));
    x0 = (x + x0) & mask;
    cout<<"===="<<x0<<"===="<<endl;
  }
}

void PrintSKyline(uint64_t *x, int th)
{
  if (party == ALICE)
  {
    uint64_t *x0 = new uint64_t[m];
    Iot[th]->send_data(x, m * sizeof(uint64_t));
    Iot[th]->recv_data(x0, m * sizeof(uint64_t));
    for (int i = 0; i < m; i++)
    {
      x0[i] = ((x[i] + x0[i]) & mask);
    }
    cout<<"===="<<x0[0]<<","<<x0[1]<<"===="<<endl;
    delete[] x0;
  }
  else
  {
    uint64_t *x0 = new uint64_t[m];
    Iot[th]->recv_data(x0, m * sizeof(uint64_t));
    for (int i = 0; i < m; i++)
    {
      x0[i] = ((x[i] + x0[i]) & mask);
    }
    cout<<"===="<<x0[0]<<","<<x0[1]<<"===="<<endl;
    delete[] x0;
    Iot[th]->send_data(x, m * sizeof(uint64_t));
  }
}

void Skyline_Print(vector<uint32_t> H)
{
  int skylineLen = H.size();
  uint32_t *s = new uint32_t[skylineLen];
  copy(H.begin(), H.end(), s);
  if (party == ALICE)
  {
    Iot[0]->send_data(s, skylineLen * sizeof(uint32_t));
    Iot[0]->recv_data(s, skylineLen * sizeof(uint32_t));
  }
  else
  {
    uint32_t *s0 = new uint32_t[skylineLen];
    Iot[0]->recv_data(s0, skylineLen * sizeof(uint32_t));
    for (int i = 0; i < skylineLen; i++)
    {
      s[i] = ((s[i] + s0[i]) & mask);
    }
    delete[] s0;
    Iot[0]->send_data(s, skylineLen * sizeof(uint32_t));
  }
  uint32_t *restmp = new uint32_t[skylineLen * m];
  int mm = 0;
  cout << skylineLen << endl;
  uint32_t *x = new uint32_t[m];
  for (int i = 0; i < skylineLen; i++)
  {
    x[0] = s[i] & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      s[i] = (s[i] - x[j - 1]) >> MAX;
      x[j] = s[i] & ((1ULL << MAX) - 1);
    }
    for (int j = m - 1; j >= 0; j--)
    {
      cout << x[j] << "\t";
    }
    cout << endl;
    copy(x, x + m, restmp + mm * m);
    mm++;
  }
  delete[] x;
}

void Skyline_Print(vector<uint64_t> &H)
{
  int skylineLen = H.size();
  uint64_t *s = new uint64_t[skylineLen];
  copy(H.begin(), H.end(), s);
  if (party == ALICE)
  {
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
    Iot[0]->recv_data(s, skylineLen * sizeof(uint64_t));
  }
  else
  {
    uint64_t *s0 = new uint64_t[skylineLen];
    Iot[0]->recv_data(s0, skylineLen * sizeof(uint64_t));
    for (int i = 0; i < skylineLen; i++)
    {
      s[i] = ((s[i] + s0[i]) & mask);
    }
    delete[] s0;
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
  }
  uint64_t *restmp = new uint64_t[skylineLen * m];
  int mm = 0;
  cout << skylineLen << endl;
  uint64_t *x = new uint64_t[m];
  for (int i = 0; i < skylineLen; i++)
  {
    x[0] = s[i] & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      s[i] = (s[i] - x[j - 1]) >> MAX;
      x[j] = s[i] & ((1ULL << MAX) - 1);
    }
    for (int j = m - 1; j >= 0; j--)
    {
      cout << x[j] << "\t";
    }
    cout << endl;
    copy(x, x + m, restmp + mm * m);
    mm++;
  }
  delete[] x;
  H.resize(skylineLen * m);
  copy(restmp, restmp + skylineLen * m, H.begin());
}

void PrintS(unordered_set<uint32_t> H)
{
  int len = H.size();
  uint64_t *x = new uint64_t[len];
  int i = 0;
  for (uint32_t b : H)
  {
    x[i] = b & mask;
    i++;
  }
  cout << "[";
  for (int j = 0; j < len - 1; j++)
  {
    uint64_t t = (x[j]) & mask;
    cout << +t << ",";
  }
  cout << +((x[len - 1]) & mask) << "]";
}

void shuffle(int len, uint32_t *&res)
{
  for (int i = 0; i < len; i++)
  {
    if (rand() % 2 == 1)
    {
      uint32_t t = rand() % len;
      uint32_t tmp = res[i];
      res[i] = res[t];
      res[t] = tmp;
    }
  }
}

void comparison(int dim, uint64_t *in, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = 32;
  uint8_t *y = new uint8_t[dim];
  Aux->comparison(in, y, dim, bw_x); // Bob less than Alice ,result is 1. Greater and equal is 0.
  Aux->B2A(y, out, dim, bw_x);        // binary share to arithmetic share
  delete[] y;
}

void comparison_with_eq(int dim, uint64_t *in, vector<uint64_t> &out, AuxProtocols *Aux)
{
  int bw_x = lambda + 1;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  uint64_t *z = new uint64_t[dim];
  Aux->comparison_with_eq(cmp, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ eq[i]) & 1;
  }
  Aux->B2A(cmp, z, dim, lambda); // binary share to arithmetic share
  copy(z, z + dim, out.begin());
  delete[] z;
  delete[] cmp;
  delete[] eq;
}

void comparison_with_eq(int dim, uint64_t *in, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = lambda + 1;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  Aux->comparison_with_eq(cmp, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ eq[i]) & 1;
  }
  // cout<<"11111"<<endl;
  // if (party == ALICE)
  // {
  //   uint8_t *tC1 =  new uint8_t[dim]; 
  //   Iot[0]->recv_data(tC1, dim * sizeof(uint8_t));
  //   Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //   for (int j = 0; j < dim; j++) {
  //     cout<< +cmp[j]<<","<< +tC1[j]<<",";
  //     tC1[j] = (cmp[j] ^ tC1[j]) & mask;
  //     cout<< +tC1[j]<<endl;
  //   }
  //   delete[] tC1;
  // }
  // else
  // {
  //   uint8_t *tC1 =  new uint8_t[dim]; 
  //   Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //   Iot[0]->recv_data(tC1, dim * sizeof(uint8_t));
  //   for (int j = 0; j < dim; j++) {
  //     cout<< +cmp[j]<<","<< +tC1[j]<<",";
  //     tC1[j] = (cmp[j] ^ tC1[j]) & mask;
  //     cout<< +tC1[j]<<endl;
  //   }
  //   delete[] tC1;
  // }  
  
  Aux->B2A(cmp, out, dim, lambda); // binary share to arithmetic share
  // cout<<"22222"<<endl;
  // if (party == ALICE)
  // {
  //   uint64_t *tC1 =  new uint64_t[dim]; 
  //   Iot[0]->recv_data(tC1, dim * sizeof(uint64_t));
  //   Iot[0]->send_data(out, dim * sizeof(uint64_t));
  //   for (int j = 0; j < dim; j++) {
  //     cout<< out[j]<<","<< tC1[j]<<",";
  //     tC1[j] = (out[j] + tC1[j]) & mask;
  //     cout<< tC1[j]<<endl;
  //   }
  //   delete[] tC1;
  // }
  // else
  // {
  //   uint64_t *tC1 =  new uint64_t[dim]; 
  //   Iot[0]->send_data(out, dim * sizeof(uint64_t));
  //   Iot[0]->recv_data(tC1, dim * sizeof(uint64_t));
  //   for (int j = 0; j < dim; j++) {
  //     cout<< out[j]<<","<< tC1[j]<<",";
  //     tC1[j] = (out[j] + tC1[j]) & mask;
  //     cout<< tC1[j]<<endl;
  //   }
  //   delete[] tC1;
  // }  
  delete[] cmp;
  delete[] eq;
}

void comparison_with_eq(int dim, uint64_t *fin, uint64_t *in, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = lambda;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  uint8_t *flp = new uint8_t[dim];
  Aux->comparison_with_eq(cmp, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ eq[i]) & mask;
  }
  Aux->equality(fin, flp, dim, bw_x);
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ flp[i]) & mask; //cmp \xor flp
  }
  Aux->B2A(cmp, out, dim, lambda); // binary share to arithmetic share
  delete[] cmp;
  delete[] eq;
  delete[] flp;
}

void comparison_with_eq2N(int dim, uint64_t *x, uint64_t *y, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = lambda + 32;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t BigPositive = 1ULL << (lambda + 31);
  uint64_t *in1 = new uint64_t[2 * dim];
  uint8_t *cmp1 = new uint8_t[2 * dim];
  uint8_t *eq1 = new uint8_t[2 * dim];
  uint64_t *out1 = new uint64_t[2 * dim];
  uint64_t *in = new uint64_t[dim];
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = x[j];
      in1[j+dim] = y[j];
    }
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = mask + 1 - x[j];
      in1[j+dim] = mask + 1 - y[j];
    }
  }
  Aux->comparison_with_eq(cmp1, eq1, in1, 2 * dim, lambda); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < 2 * dim; i++)
  {
    cmp1[i] = (cmp1[i] ^ eq1[i]) & 1;
  }
  Aux->B2A(cmp1, out1, 2 * dim, 32); // binary share to arithmetic share
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      in[j] = BigPositive + x[j] - y[j] + ((out1[j+dim]-out1[j])&mask)*(mask + 1);
    }
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in[j] = BigPositive + y[j] - x[j] + ((out1[j]-out1[j+dim])&mask)*(mask + 1);
    }
  }
  Aux->comparison_with_eq(cmp, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ eq[i]) & 1;
  }
  Aux->B2A(cmp, out, dim, lambda);
  delete[] cmp;
  delete[] eq;
  delete[] in;
  delete[] in1;
  delete[] cmp1;
  delete[] eq1;
  delete[] out1;
}

void comparison_with_eq2N(int dim, uint64_t *x, uint64_t *y, uint8_t *&out, AuxProtocols *Aux)
{
  int bw_x = lambda + 32;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t BigPositive = 1ULL << (lambda + 31);
  uint64_t *in1 = new uint64_t[2 * dim];
  uint8_t *cmp1 = new uint8_t[2 * dim];
  uint8_t *eq1 = new uint8_t[2 * dim];
  uint64_t *out1 = new uint64_t[2 * dim];
  uint64_t *in = new uint64_t[dim];
  uint8_t *eq = new uint8_t[dim];
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = x[j];
      in1[j+dim] = y[j];
    }
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = mask + 1 - x[j];
      in1[j+dim] = mask + 1 - y[j];
    }
  }
  Aux->comparison_with_eq(cmp1, eq1, in1, 2 * dim, lambda); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < 2 * dim; i++)
  {
    cmp1[i] = (cmp1[i] ^ eq1[i]) & 1;
  }
  Aux->B2A(cmp1, out1, 2 * dim, 32); // binary share to arithmetic share
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      in[j] = BigPositive + x[j] - y[j] + ((out1[j+dim]-out1[j])&mask)*(mask + 1);
    }
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in[j] = BigPositive + y[j] - x[j] + ((out1[j]-out1[j+dim])&mask)*(mask + 1);
    }
  }
  Aux->comparison_with_eq(out, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    out[i] = (out[i] ^ eq[i]) & 1;
  }
  delete[] eq;
  delete[] in;
  delete[] in1;
  delete[] cmp1;
  delete[] eq1;
  delete[] out1;
}

void comparison_with_eq2N(int dim, uint64_t *x, uint64_t *y, uint64_t *&out,uint64_t *&out4,uint64_t *&out2,uint8_t *&out5,uint8_t *&out3, AuxProtocols *Aux)
{
  int bw_x = lambda + 32;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t BigPositive = 1ULL << (lambda + 31);
  uint64_t *in1 = new uint64_t[2 * dim];
  uint8_t *cmp1 = new uint8_t[2 * dim];
  uint8_t *eq1 = new uint8_t[2 * dim];
  uint64_t *out1 = new uint64_t[2 * dim];
  // uint8_t *ba = new uint8_t[2 * dim];
  uint64_t *in = new uint64_t[dim];
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = x[j];
      in1[j+dim] = y[j];
    }
    // memset(ba, 1, 2*dim);
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in1[j] = (0 - x[j]) & mask;
      in1[j+dim] = (0 - y[j]) & mask;
    }
    // memset(ba, 0, 2*dim);
  }
  Aux->comparison_with_eq(cmp1, eq1, in1, 2 * dim, lambda); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < 2 * dim; i++)
  {
    cmp1[i] = (cmp1[i] ^ eq1[i]) & 1;
  }
  // Aux->AND(cmp1, ba, ba, 2 * dim);
  // for (int i = 0; i < dim; i++)
  // {
  //   cmp1[i] = (cmp1[i] - ba[i]) & 1;
  // }
  // if (party == ALICE)
  // {
  //   uint8_t *tC2 =  new uint8_t[2 * dim]; 
  //   Iot[2]->recv_data(tC2, 2 * dim * sizeof(uint8_t));
  //   Iot[3]->send_data(cmp1, 2 * dim * sizeof(uint8_t));
  //   for (int i = 0; i < 2 * dim; i++)
  //   {
  //     out3[i] = (cmp1[i]^tC2[i])&1;
  //     out5[i] = cmp1[i];
  //   }
  //   delete[] tC2;
  // }
  // else
  // {
  //   uint8_t *tC2 =  new uint8_t[2 * dim]; 
  //   Iot[2]->send_data(cmp1, 2 * dim * sizeof(uint8_t));
  //   Iot[3]->recv_data(tC2, 2 * dim * sizeof(uint8_t));
  //   for (int i = 0; i < 2 * dim; i++)
  //   {
  //     out3[i] = (cmp1[i]^tC2[i])&1;
  //     out5[i] = cmp1[i];
  //   }
  //   delete[] tC2;
  // }
  Aux->B2A(cmp1, out1, 2 * dim, 32); // binary share to arithmetic share
  // memcpy(out2, out1, 2*dim);
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++) {
      // {x1-y1-mask*(x1>=mask-x2)}>={y2-x2-mask*(y1>=mask-y2)}
      in[j] = BigPositive + x[j] - y[j] + ((out1[j+dim]-out1[j])&mask)*(mask + 1);
    }
    // uint64_t *tC1 =  new uint64_t[2 * dim]; 
    // Iot[0]->recv_data(tC1, 2 * dim * sizeof(uint64_t));
    // Iot[1]->send_data(out1, 2 * dim * sizeof(uint64_t));
    // for (int i = 0; i < 2 * dim; i++)
    // {
    //   out2[i] = (out1[i]+tC1[i]);
    //   out4[i] = out1[i];
    // }
    // delete[] tC1;
  }
  else
  {
    for (int j = 0; j < dim; j++) {
      in[j] = BigPositive + y[j] - x[j] + ((out1[j]-out1[j+dim])&mask)*(mask + 1);
    }
    // uint64_t *tC1 =  new uint64_t[2 * dim]; 
    // Iot[0]->send_data(out1, 2 * dim * sizeof(uint64_t));
    // Iot[1]->recv_data(tC1, 2 * dim * sizeof(uint64_t));
    // for (int i = 0; i < 2 * dim; i++)
    // {
    //   out2[i] = (out1[i]+tC1[i]);
    //   out4[i] = out1[i];
    // }
    // delete[] tC1;
  }
  Aux->comparison_with_eq(cmp, eq, in, dim, bw_x); // Bob less and equal than Alice ,result is 1. Greater  is 0.
  for (int i = 0; i < dim; i++)
  {
    cmp[i] = (cmp[i] ^ eq[i]) & 1;
  }
  Aux->B2A(cmp, out, dim, lambda);
  delete[] cmp;
  delete[] eq;
  delete[] in;
  delete[] in1;
  delete[] cmp1;
  delete[] eq1;
  delete[] out1;
}

void equality(int dim, uint64_t *in, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = 32;
  uint8_t *eq = new uint8_t[dim];

  Aux->equality(in, eq, dim, bw_x); // Bob equal than Alice ,result is 1. otherwise is 0.

  Aux->B2A(eq, out, dim, bw_x); // binary share to arithmetic share
  delete[] eq;
}

void loadP(vector<vector<uint32_t>> &p, string data_path)
{
  ifstream inf;
  inf.open(data_path);
  string line;
  while (getline(inf, line))
  {
    istringstream sin(line);
    // cout << "o:" << line << endl;
    vector<uint32_t> lineArray;
    string field;
    while (getline(sin, field, ','))
    {
      lineArray.push_back(stoi(field));
    }
    p.push_back(lineArray);
    lineArray.clear();
  }
  inf.close();
}

uint32_t Aggreate(vector<uint32_t> t)
{
  uint32_t x = t[0];
  for (int j = 1; j < m; j++)
  {
    //  x = (x *10000) + t[j];
    x = (x << MAX) + t[j];
  }
  return x;
}
uint32_t Aggreate(uint32_t *t)
{
  uint32_t x = t[0];
  for (int j = 1; j < m; j++)
  {
    x = (x << MAX) + t[j];
    // x = (x *10000) + t[j];
  }
  return x;
}

void Aggreate(uint32_t *t, uint32_t &res)
{
  res = t[0];
  for (int j = 1; j < m; j++)
  {
    res = (((res) << MAX) + t[j]) & mask;
    // *res = (((*res) *10000) + t[j]) & mask;
  }
}

void Aggreate(uint64_t *t, uint64_t &res)
{
  res = t[0];
  for (int j = 1; j < m; j++)
  {
    res = (((res) << MAX) + t[j]) & mask;
    // *res = (((*res) *10000) + t[j]) & mask;
  }
}

uint64_t Aggreate(uint64_t *t)
{
  uint64_t res = t[0];
  for (int j = 1; j < m; j++)
  {
    res = (((res) << MAX) + t[j]) & mask;
    // *res = (((*res) *10000) + t[j]) & mask;
  }
  return res;
}

void Aggreate(vector<vector<uint32_t>> p)
{
  int i = 0;
  for (vector<uint32_t> t : p)
  {
    uint32_t tmp = Aggreate(t);
    H.insert(tmp);
  }
}

void loadG(vector<vector<uint32_t>> p, unordered_map<uint32_t, vector<uint32_t>> &G)
{
  for (int j = 0; j < m; j++)
  {
    vector<uint32_t> tmp;
    for (vector<uint32_t> t : p)
    {
      tmp.push_back(t[j]);
    }
    sort(tmp.begin(), tmp.end());
    tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
    G[j + 1] = tmp;
  }
  Aggreate(p);
}

int indexG(vector<uint32_t> G, uint32_t q)
{
  int res = 0;
  for (uint32_t t : G)
  {
    if (t <= q)
      res++;
  }
  return res;
}

void plain(vector<vector<uint32_t>> p, vector<uint32_t> q)
{
  double start = omp_get_wtime();
  // Process
  int k = 1;
  // left and right add a row
  for (int k1 = G[k].size(); k1 >= 0; k1--)
  {
    unordered_set<uint32_t> hs;
    hs.insert(0);
    Skyline[(k1 << MAX) + G[k + 1].size()] = hs;
  }
  cout << "K1:" << G[k].size() << endl;
  for (int k2 = G[k + 1].size(); k2 >= 0; k2--)
  {
    unordered_set<uint32_t> hs;
    hs.insert(0);
    Skyline[(G[k].size() << MAX) + k2] = hs;
  }
  cout << "K2:" << G[k + 1].size() << endl;
  uint32_t *g = new uint32_t[2];
  int Vlen = 0;
  for (int k1 = G[k].size() - 1; k1 >= 0; k1--)
  {
    g[k - 1] = G[k][k1]; // right
    k++;
    for (int k2 = G[k].size() - 1; k2 >= 0; k2--)
    {
      g[k - 1] = G[k][k2]; // upper
      unordered_set<uint32_t> hs;
      uint32_t tmp = Aggreate(g);
      if (H.find(tmp) != H.end())
      { // point in the upper right corner of this grid
        hs.insert(tmp);
      }
      else
      {
        int t1 = ((k1 + 1) << MAX) + k2;
        int t2 = ((k1) << MAX) + k2 + 1;
        int t3 = ((k1 + 1) << MAX) + k2 + 1;
        hs.insert(Skyline[t1].begin(), Skyline[t1].end());
        unordered_set<uint32_t> h2(Skyline[t2]);
        for (uint32_t b : Skyline[t3])
        {
          if (hs.find(b) != hs.end())
          {
            hs.erase(b);
          }
          else
          {
            h2.erase(b);
          }
        }
        hs.insert(h2.begin(), h2.end());
        if (hs.size() > Vlen)
          Vlen = hs.size();
      }
      Skyline[(k1 << MAX) + k2] = hs;
      // cout << hs.size() <<"\t";
    }
    // cout<<endl;
    k--;
  }
  delete[] g;
  cout << "len:" << Vlen << endl;
  // for (int k2 = G[k + 1].size(); k2 >= 0; k2--)
  // {
  //   for (int k1 = 0; k1 <= G[k].size(); k1++)
  //   {
  //     PrintS(Skyline[(k1 << MAX) + k2]);
  //     cout << " ";
  //   }
  //   cout << endl;
  // }

  // for (int j = 0; j<= G[1].size(); j++) {
  //   for (int i = 0; i <=G[2].size(); i++) {
  //     PrintS(Skyline[(j<<MAX)+i]);
  //     cout << " ";
  //   }
  //   cout << endl;
  // }
  double end = omp_get_wtime();
  double time = end - start;
  plain_process = ((double)time);
  unordered_set<uint32_t> skyline(Skyline[(indexG(G[1], q[0]) << MAX) + indexG(G[2], q[1])]);
  cout << indexG(G[1], q[0]) << "," << indexG(G[2], q[1]) << endl;
  cout << skyline.size() << endl;
  for (uint32_t b : skyline)
  {
    vector<uint32_t> x(m);
    x[0] = b & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      b = (b - x[j - 1]) >> MAX;
      x[j] = b & ((1ULL << MAX) - 1);
    }
    // x[0] = b %10000;
    //  for (int j = 1; j < m; j++) {
    //     b = (b - x[j-1]) /10000;
    //     x[j] = b %10000;
    // }
    for (int j = m - 1; j >= 0; j--)
    {
      cout << x[j] << "\t";
    }
    cout << endl;
    ;
  }
}

void SSQ(vector<uint32_t> q, vector<uint32_t> &Q)
{
  int t = q.size();
  uint32_t *x = new uint32_t[t];
  uint32_t *y = new uint32_t[t];
  // PRG128 prg;
  prg.random_data(x, t * sizeof(uint32_t));
  if (party == ALICE)
  {
    for (int i = 0; i < t; i++)
    {
      x[i] = x[i] & mask;
      y[i] = (q[i] - x[i]) & mask;
    }
    copy(x, x + t, Q.begin());
    Iot[0]->send_data(y, t * sizeof(uint32_t));
  }
  else
  {
    uint32_t *y0 = new uint32_t[t];
    Iot[0]->recv_data(y0, t * sizeof(uint32_t));
    copy(y0, y0 + t, Q.begin());
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}
void SSG(unordered_map<uint32_t, vector<uint32_t>> G, uint32_t**&SS_G)
{
  uint32_t B = 0;
  SS_G = new uint32_t*[m + 1];
  SS_G[0] = new uint32_t[m];
  for (int k = 0; k < m; k++)
  {
    SS_G[0][k] = G[k + 1].size();
    SS_G[k+1] = new uint32_t[G[k + 1].size()];
    int i = 0;
    for (uint32_t b : G[k + 1])
    {
      // PRG128 prg;
      prg.random_data(&B, sizeof(uint32_t));
      if (party == ALICE)
      {
        SS_G[k+1][i] = B & mask;
        Iot[0]->send_data(&B, sizeof(uint32_t));
      }
      else
      {
        uint32_t B0 = 0;
        Iot[0]->recv_data(&B0, sizeof(uint32_t));
        SS_G[k+1][i] = (b - B0) & mask;
      }
      i++;
    }
  }
}
void SSH(unordered_set<uint32_t> H, vector<uint32_t> &SS_H)
{
  uint32_t B = 0;
  uint32_t *x = new uint32_t[H.size()];
  int i = 0;
  for (uint32_t b : H)
  {
    // PRG128 prg;
    prg.random_data(&B, sizeof(uint32_t));
    if (party == ALICE)
    {
      x[i] = B;
      Iot[0]->send_data(&B, sizeof(uint32_t));
    }
    else
    {
      uint32_t t = 0;
      Iot[0]->recv_data(&t, sizeof(uint32_t));
      x[i] = (b - (t & mask)) & mask;
    }
    i++;
  }
  copy(x, x + H.size(), SS_H.begin());
  delete[] x;
}
void SSC(unordered_set<uint32_t> H, unordered_map<uint32_t, vector<uint32_t>> G, uint32_t * &SS_C)
{
  uint32_t B = 0;
  uint32_t *g = new uint32_t[2];
  int len1 = G[1].size();
  int len2 = G[2].size();
  SS_C = new uint32_t[len1 * len2];
  // PRG128 prg;
  for (int k1 = len1 - 1; k1 >= 0; k1--)
  {
    g[0] = G[1][k1]; // right
    for (int k2 = len2 - 1; k2 >= 0; k2--)
    {
      g[1] = G[2][k2]; // upper
      uint32_t tmp = 0;
      Aggreate(g, tmp);
      uint32_t con = 0;
      if (H.find(tmp) != H.end())
      { // point in the upper right corner of this grid
        con = 1;
      }
      prg.random_data(&B, sizeof(uint32_t));
      if (party == ALICE)
      {
        uint32_t t = (con - B) & mask;
        Iot[0]->send_data(&t, sizeof(uint32_t));
      }
      else
      {
        Iot[0]->recv_data(&B, sizeof(uint32_t));
      }
      SS_C[k1 * len2 + k2] = B;
    }
  }
  delete[] g;
}

void Prod_H(int dim, uint64_t *inA, uint64_t *inB, uint64_t *&out, LinearOT *Prod)
{
  int bw_x = 32;
  Prod->hadamard_product(dim, inA, inB, out, bw_x, bw_x, bw_x, false);
}

void Prod_M(int dim1, int dim2, int dim3, uint64_t *inA, uint64_t *inB, uint64_t *&out, LinearOT *Prod, MultMode mode)
{
  uint8_t *msbA = nullptr;
  uint8_t *msbB = nullptr;
  int bw_x = 32;
  uint64_t *outC = new uint64_t[dim1 * dim2 * dim3];
  Prod->matrix_multiplication(dim1, dim2, dim3, inA, inB, outC, bw_x, bw_x, bw_x, false, true, false, mode, msbA, msbB);
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3; j++) {
      out[i * dim3 + j] = 0;
      for (int k = 0; k < dim2; k++) {
        out[i * dim3 + j] = (out[i * dim3 + j] + outC[dim2 * dim3 * i + dim2 * j + k]) & mask;
      }
    }
  }
  delete []outC;
}

uint64_t HashP(uint64_t in) 
{
  block128 inB = toBlock(in);
  // block128 CRH.H(block128)
  block128 outB = crh.H(inB);
  uint64_t out = (uint64_t)_mm_extract_epi64(outB, 0);
  // cout<<(uint64_t)_mm_extract_epi64(outB, 0)<<endl;
  // cout<<(uint64_t)_mm_extract_epi64(outB, 1)<<endl;
  return out;
}

void Pi_Gen(int dim, uint64_t *&res)
{
  uint32_t *t = new uint32_t[dim];
  for (int i = 0; i < dim; i++) {
    t[i] = i;
  }
  for (int i = 0; i < dim; i++) {
    if(rand()%2==1){
      uint32_t tmp = rand()%dim;
      // swap(&t[i], &t[tmp]);
      uint32_t tmp2 = t[i];
      t[i] = t[tmp];
      t[tmp] = tmp2;
    }
  }
  for (int i = 0; i < dim; i++)
  {
    res[i*dim+t[i]] = 1;
  }
  delete[] t;
}

void SkylineGen_1(uint64_t *H1, uint64_t *H2, uint64_t *H3, int len1, int len2, int len3, int &num, uint64_t *&res, AuxProtocols *Aux, LinearOT *Prod, NetIO *Io)
{
  int dim = len1 + len2 + len3;
  uint64_t *h = new uint64_t[dim];
  uint64_t *index = new uint64_t[dim]();
  memcpy(h, H1, len1 * sizeof(uint64_t));
  memcpy(h + len1, H2, len2 * sizeof(uint64_t));
  memcpy(h + len1 + len2, H3, len3 * sizeof(uint64_t));
  int len12 = len1 + len2;
  int len = len12 * len3;
  uint64_t *in = new uint64_t[len];
  uint64_t *out = new uint64_t[len];
  uint32_t tmp;
  if (party == ALICE)
  {
    for (int t2 = 0; t2 < len3; t2++)
    {
      tmp = t2 * len12;
      for (int t1 = 0; t1 < len12; t1++)
      {
        in[tmp+t1] = (H3[t2] - h[t1]) & mask;
      }
    }
  }
  else
  {
    for (int t2 = 0; t2 < len3; t2++)
    {
      tmp = t2 * len12;
      for (int t1 = 0; t1 < len12; t1++)
      {
        in[tmp+t1] = (h[t1] - H3[t2]) & mask;
      }
    }    
  }
  equality(len, in, out, Aux);
  for (int t2 = 0; t2 < len3; t2++)
  {
    tmp = t2 * len12;
    for (int t1 = 0; t1 < len12; t1++)
    {
      index[t1] = (index[t1] + out[tmp + t1]) & mask; // delete element for column
      index[len12 + t2] = (index[len12 + t2] + out[tmp + t1]) & mask; // delete element for row
    }
  }
  uint64_t *in2 = new uint64_t[dim];
  uint64_t *out2 = new uint64_t[dim];
  if (party == ALICE)
  {
    for (int t1 = 0; t1 < len12; t1++)
    {
      in2[t1] = (0 - index[t1]) & mask;
    }
    for (int t2 = len12; t2 < dim; t2++)
    {
      in2[t2] = (index[t2] - 1) & mask;
    }
  }
  else
  {
    for (int t1 = 0; t1 < len12; t1++)
    {
      in2[t1] = (index[t1] - 1) & mask;
    }
    for (int t2 = len12; t2 < dim; t2++)
    {
      in2[t2] = (0 - index[t2]) & mask;
    }
  }
  comparison(dim, in2, out2, Aux);
  memcpy(index, out2, dim * sizeof(uint64_t)); 
  uint64_t *skyT = new uint64_t[dim];
  Prod_H(dim, h, index, skyT, Prod);
  uint64_t *Pi = new uint64_t[dim*dim]();
  Pi_Gen(dim, Pi);
  MultMode mode = MultMode::Alice_has_A;
  uint64_t *skyTT = new uint64_t[dim];
  Prod_M(dim, dim, 1, Pi, skyT, skyTT, Prod, mode);
  mode = MultMode::Bob_has_A;
  uint64_t *sky = new uint64_t[dim];
  Prod_M(dim, dim, 1, Pi, skyTT, sky, Prod, mode);
  uint64_t *r = new uint64_t[dim];
  prg.random_data(r, dim * sizeof(uint64_t));
  Prod_H(dim, sky, r, skyT, Prod);
  if (party == ALICE){
    Io->send_data(skyT, dim * sizeof(uint64_t));
    Io->recv_data(skyTT, dim * sizeof(uint64_t));
  }
  else
  {
    Io->recv_data(skyTT, dim * sizeof(uint64_t));
    Io->send_data(skyT, dim * sizeof(uint64_t));
  }
  for (int t1 = 0; t1 < dim; t1++)
  { 
    r[t1] = (skyT[t1] + skyTT[t1]) & mask;
  }
  num = dim - count(r, r + dim, 0);
  int t2 = 0;
  res = new uint64_t[num];
  for (int t1 = 0; t1 < dim; t1++)
  {
    if (r[t1] != 0)
    {
      res[t2] = sky[t1] & mask;
      t2++;
    }
  }
  delete []skyT;
  delete []Pi;
  delete []skyTT;
  delete []sky;
  delete []r;
  delete[] h;
  delete[] index;
  delete[] in;
  delete[] out;
  delete[] in2;
  delete[] out2;
}

void SkylineGen_1(uint64_t *h, int len1, int len2, int len3, int &num, uint64_t *&res, AuxProtocols *Aux, LinearOT *Prod, NetIO *Io)
{
  int dim = len1 + len2 + len3;
  uint64_t *ind = new uint64_t[dim];
  memset(ind, 0, dim * sizeof(uint64_t));
  int len12 = len1 + len2;
  int len = len12 * len3;
  uint64_t *in = new uint64_t[len];
  uint64_t *out = new uint64_t[len];
  uint32_t tmp;
  if (party == ALICE)
  {
    for (int t2 = 0; t2 < len3; t2++)
    {
      tmp = t2 * len12;
      for (int t1 = 0; t1 < len12; t1++)
      {
        // in[tmp+t1] = (H3[t2] - h[t1]) & mask;
        in[tmp+t1] = (h[len12 + t2] - h[t1]) & mask;
      }
    }
  }
  else
  {
    for (int t2 = 0; t2 < len3; t2++)
    {
      tmp = t2 * len12;
      for (int t1 = 0; t1 < len12; t1++)
      {
        // in[tmp+t1] = (h[t1] - H3[t2]) & mask;
        in[tmp+t1] = (h[t1] - h[len12 + t2]) & mask;
      }
    }    
  }
  equality(len, in, out, Aux);
  for (int t2 = 0; t2 < len3; t2++)
  {
    tmp = t2 * len12;
    for (int t1 = 0; t1 < len12; t1++)
    {
      ind[t1] = (ind[t1] + out[tmp + t1]) & mask; // delete element for column
      ind[len12 + t2] = (ind[len12 + t2] + out[tmp + t1]) & mask; // delete element for row
    }
  }
  uint64_t *in2 = new uint64_t[dim];
  uint64_t *out2 = new uint64_t[dim];
  if (party == ALICE)
  {
    for (int t1 = 0; t1 < len12; t1++)
    {
      in2[t1] = (0 - ind[t1]) & mask;
    }
    for (int t2 = len12; t2 < dim; t2++)
    {
      in2[t2] = (ind[t2] - 1) & mask;
    }
  }
  else
  {
    for (int t1 = 0; t1 < len12; t1++)
    {
      in2[t1] = (ind[t1] - 1) & mask;
    }
    for (int t2 = len12; t2 < dim; t2++)
    {
      in2[t2] = (0 - ind[t2]) & mask;
    }
  }
  comparison(dim, in2, out2, Aux);
  // uint64_t *skyT = new uint64_t[dim];
  // Prod_H(dim, h, out2, skyT, Prod);
  Prod_H(dim, h, out2, ind, Prod);
  uint64_t *Pi = new uint64_t[dim*dim];
  memset(Pi, 0, dim*dim * sizeof(uint64_t));
  Pi_Gen(dim, Pi);
  MultMode mode = MultMode::Alice_has_A;
  // uint64_t *skyTT = new uint64_t[dim];
  // Prod_M(dim, dim, 1, Pi, skyT, skyTT, Prod, mode);
  Prod_M(dim, dim, 1, Pi, ind, in2, Prod, mode);
  mode = MultMode::Bob_has_A;
  // uint64_t *sky = new uint64_t[dim];
  // Prod_M(dim, dim, 1, Pi, skyTT, sky, Prod, mode);
  Prod_M(dim, dim, 1, Pi, in2, out2, Prod, mode);
  // uint64_t *r = new uint64_t[dim];
  // PRG128 prg;
  // prg.random_data(r, dim * sizeof(uint64_t));
  // Prod_H(dim, sky, r, skyT, Prod);
  prg.random_data(h, dim * sizeof(uint64_t));
  Prod_H(dim, out2, h, ind, Prod);
  if (party == ALICE){
    Io->send_data(ind, dim * sizeof(uint64_t));
    Io->recv_data(h, dim * sizeof(uint64_t));
  }
  else
  {
    Io->recv_data(h, dim * sizeof(uint64_t));
    Io->send_data(ind, dim * sizeof(uint64_t));
  }
  for (int t1 = 0; t1 < dim; t1++)
  { 
    in2[t1] = (ind[t1] + h[t1]) & mask;
  }
  num = dim - count(in2, in2 + dim, 0);
  int t2 = 0;
  res = new uint64_t[num];
  for (int t1 = 0; t1 < dim; t1++)
  {
    if (in2[t1] != 0)
    {
      res[t2] = out2[t1];
      t2++;
    }
  }
  // delete []skyT;
  delete []Pi;
  // delete []skyTT;
  // delete []sky;
  // delete []r;
  // delete[] h;
  delete[] ind;
  delete[] in;
  delete[] out;
  delete[] in2;
  delete[] out2;
}

void Pos(vector<uint32_t> Q, uint32_t** SS_G, vector<uint32_t> &pos)
{
  uint32_t *x = new uint32_t[m];
  uint64_t y = 1ULL << lambda;
  for (int k = 0; k < m; k++)
  {
    uint64_t *in = new uint64_t[SS_G[0][k]];
    vector<uint64_t> out(SS_G[0][k]);

    for (int i = 0; i < SS_G[0][k]; i++)
    {
      if (party == ALICE)
      {
        if (Q[k] < SS_G[k + 1][i])
        {
          in[i] = (mask + Q[k] - SS_G[k + 1][i]) & mask; //-2>-3==(mask-2)>(mask-3)
        }
        else
        {
          in[i] = (y + Q[k] - SS_G[k + 1][i]) & mask; // add the same 1
        }
      }
      else
      {
        if (SS_G[k + 1][i] < Q[k])
        {
          in[i] = (mask + SS_G[k + 1][i] - Q[k]) & mask;
        }
        else
        {
          in[i] = (y + SS_G[k + 1][i] - Q[k]) & mask;
        }
      }
    }
    comparison_with_eq(SS_G[0][k], in, out, Auxt[0]);
    uint32_t tmp = accumulate(out.begin(), out.end(), 0) & mask;
    x[k] = tmp;
    delete[] in;
    out.clear();
    vector<uint64_t>().swap(out);
  }
  copy(x, x + m, pos.begin());
  delete[] x;
}

void PosIndex(vector<uint32_t> Q, uint32_t** SS_G, uint32_t *&pos, unordered_map<uint32_t, uint32_t *> &posindex)
{
  // PRG128 prg;
  uint64_t y = 1ULL << lambda;
  uint32_t SS_zero = 0;
    prg.random_data(&SS_zero, sizeof(uint32_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(&SS_zero, sizeof(uint32_t));
    }
    else
    {
      Iot[0]->recv_data(&SS_zero, sizeof(uint32_t));
      SS_zero = (0 - SS_zero) & mask;
    }
    uint32_t SS_one = 0;
    prg.random_data(&SS_one, sizeof(uint32_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(&SS_one, sizeof(uint32_t));
    }
    else
    {
      Iot[0]->recv_data(&SS_one, sizeof(uint32_t));
      SS_one = (1 - SS_one) & mask;
    }
  for (int k = 0; k < m; k++)
  {
    uint64_t *in = new uint64_t[SS_G[0][k]];
    vector<uint64_t> out(SS_G[0][k]);
    for (int i = 0; i < SS_G[0][k]; i++)
    {
      if (party == ALICE)
      {
        if (Q[k] < SS_G[k + 1][i])
        {
          in[i] = (mask + Q[k] - SS_G[k + 1][i]) & mask; //-2>-3==(mask-2)>(mask-3)
        }
        else
        {
          in[i] = y + (Q[k] - SS_G[k + 1][i]) & mask; // add the same 1
        }
      }
      else
      {
        if (SS_G[k + 1][i] < Q[k])
        {
          in[i] = (mask + SS_G[k + 1][i] - Q[k]) & mask;
        }
        else
        {
          in[i] = y + (SS_G[k + 1][i] - Q[k]) & mask;
        }
      }
    }
    comparison_with_eq(SS_G[0][k], in, out, Auxt[0]); // larger and equal
    uint32_t postmp = accumulate(out.begin(), out.end(), 0) & mask;
    pos[k] = postmp;
    // make 1,1,0,0 become 0,1,0,0, index
    posindex[k] = new uint32_t[SS_G[0][k] + 1];
    posindex[k][0] = (SS_one - out[0]) & mask;
    for (int i = 1; i < SS_G[0][k]; i++)
    {
      posindex[k][i] = (out[i - 1] - out[i]) & mask;
    }
    posindex[k][SS_G[0][k]] = (out[SS_G[0][k] - 1] - SS_zero) & mask;
    delete[] in;
    out.clear();
    vector<uint64_t>().swap(out);
  }
}

void PosIndex_T(vector<uint32_t> Q, uint32_t** SS_G, uint32_t *&pos, unordered_map<uint32_t, uint32_t *> &posindex)
{
  // PRG128 prg;
  uint64_t y = 1ULL << lambda;
  uint32_t SS_zero = 0;
    prg.random_data(&SS_zero, sizeof(uint32_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(&SS_zero, sizeof(uint32_t));
    }
    else
    {
      Iot[0]->recv_data(&SS_zero, sizeof(uint32_t));
      SS_zero = (0 - SS_zero) & mask;
    }
    uint32_t SS_one = 0;
    prg.random_data(&SS_one, sizeof(uint32_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(&SS_one, sizeof(uint32_t));
    }
    else
    {
      Iot[0]->recv_data(&SS_one, sizeof(uint32_t));
      SS_one = (1 - SS_one) & mask;
    }
  for (int k = 0; k < m; k++)
  {
    uint64_t *out = new uint64_t[SS_G[0][k]];
    #pragma omp parallel num_threads(THs)
    {
      #pragma omp single 
      {
        for (int itr = 0; itr < THs; itr++)
        {
          #pragma omp task firstprivate(itr, THs, out)
          {
            int lendt = (((SS_G[0][k] - 1) * itr)/ THs) + 1;
            int lenut = ((SS_G[0][k] - 1) * (itr + 1))/ THs;
            if(itr == 0){
              lendt = 0;
            }
            int lent = lenut - lendt + 1;
            uint64_t *in1 = new uint64_t[lent];
            uint64_t *out1 = new uint64_t[lent];
            for (int i = lendt; i <= lenut; i++)
            {
              if (party == ALICE)
              {
                if (Q[k] < SS_G[k + 1][i])
                {
                  in1[i-lendt] = (mask + Q[k] - SS_G[k + 1][i]) & mask; //-2>-3==(mask-2)>(mask-3)
                }
                else
                {
                  in1[i-lendt] = y + (Q[k] - SS_G[k + 1][i]) & mask; // add the same 1
                }
              }
              else
              {
                if (SS_G[k + 1][i] < Q[k])
                {
                  in1[i-lendt] = (mask + SS_G[k + 1][i] - Q[k]) & mask;
                }
                else
                {
                  in1[i-lendt] = y + (SS_G[k + 1][i] - Q[k]) & mask;
                }
              }
            }
            comparison_with_eq(lent, in1, out1, Auxt[itr]); // larger and equal
            for (int i = lendt; i <= lenut; i++)
            {
              out[i] = out1[i-lendt] & mask;
            }
          }
        }
        #pragma omp taskwait
      }  
    }
    uint64_t postmp = 0;
    #pragma omp parallel for reduction(+:postmp)
    for (int i = 0; i < SS_G[0][k]; i++)
    {
      postmp = (postmp + out[i]) & mask;
    }
    pos[k] = postmp & mask;
    // make 1,1,0,0 become 0,1,0,0, index
    posindex[k] = new uint32_t[SS_G[0][k] + 1];
    posindex[k][0] = (SS_one - out[0]) & mask;
    for (int i = 1; i < SS_G[0][k]; i++)
    {
      posindex[k][i] = (out[i - 1] - out[i]) & mask;
    }
    posindex[k][SS_G[0][k]] = (out[SS_G[0][k] - 1] - SS_zero) & mask;
    delete[] out;
  }
}

void Poly(int len, uint64_t *a, int x, uint64_t &res)
{
  res = a[0];
  uint64_t t = x;
  for (int i = 1; i < len; i++)
  {
    res = (res + a[i] * t) & mask;
    t = (t * x) & mask;
  } 
}

uint64_t *SkylineRes_old(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 3;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t **a = new uint64_t*[len1];
  for (int i = 0; i < len1; i++)
  {
    a[i] = new uint64_t[polylen];
    prg.random_data(a[i], polylen * sizeof(uint64_t));
  } 
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  uint64_t *tmp0 = new uint64_t[len];
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
    }
  }
  delete[] tmp0;
  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // construct the same skyline
              // int slen = SS_L[(k1<<MAX)+k2];
              // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
              uint64_t offset = 0;
              Poly(polylen, a[k1], k2, offset);
              int lent = SS_L[(k1 * len2) + k2];
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
              }
              uint64_t *tmp1 = new uint64_t[lent];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
              }
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }

  // for (int k1 = 0; k1 < len1; k1++)
  // {
  //   for (int k2 = 0; k2 < len2; k2++)
  //   {
  //     // construct the same skyline
  //     // int slen = SS_L[(k1<<MAX)+k2];
  //     // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
  //     uint64_t offset = 0;
  //     Poly(polylen, a[k1], k2, offset);
  //     int slen = SS_L[(k1 * len2) + k2];
  //     for (int k = 0; k < slen; k++)
  //     {
  //       SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
  //     }
  //     uint64_t *tmp1 = new uint64_t[slen];
  //     if (party == ALICE)
  //     {
  //       Iot[0]->send_data(SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
  //       Iot[0]->recv_data(tmp1, slen * sizeof(uint64_t));
  //     }
  //     else
  //     {
  //       Iot[0]->recv_data(tmp1, slen * sizeof(uint64_t));
  //       Iot[0]->send_data(SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
  //     }
  //     for (int k = 0; k < slen; k++)
  //     {
  //       SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
  //     }
  //     delete[] tmp1;
  //   }
  // } 
  
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  cout << "select element" << endl;
  // select
  double starts = omp_get_wtime();
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  PosIndex(Q, SS_G, pos, posindex);
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[1];
  uint64_t *inB = new uint64_t[1];
  uint64_t *outC = new uint64_t[1];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = IT[i/2];
    inB[0] = IT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    IT[i] = outC[0];
  }
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD = new uint64_t[polylen];
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      int slen = SS_L[(k1 * len2) + k2];
      for (int k = 0; k < slen; k++)
      {
        tmp2[k1 * len + k] = (tmp2[k1 * len + k] + SS_Skyline[(k1 * len2) + k2][k] * posindex[1][k2]) & mask;
      }
    }
    //a[k1]*pos[1]+b[k1])
    uint64_t mk1 = 0;
    Prod_H(polylen, a[k1], JT, outD, Prodt[0]);
    for (int k = 0; k < polylen; k++)
    {
      mk1 = (mk1 + outD[k]) & mask;
    }
    // uint64_t offset = b[0] + b[1] * k1;  // a2*i+b2
    uint64_t offset = 0;
    Poly(polylen, b, k1, offset);
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset - mk1) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
    if (party == ALICE)
    {
      Iot[0]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
      Iot[0]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
    }
    else
    {
      Iot[0]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
      Iot[0]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
    }
    for (int k = 0; k < len; k++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  delete[] tmpt;
  delete[] tmp2;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  for (int k = 0; k < polylen; k++)
  {
      mk2 = (mk2 + outD[k]) & mask;
  }
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  reslen = len;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] pos;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return res;
}

uint64_t *SkylineRes(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 2;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t *rp2 = new uint64_t[len1];
  prg.random_data(rp2, len1 * sizeof(uint64_t));
  // memset(rp2, 0, len1 * sizeof(uint64_t));
  uint64_t rp1 = 0;
  prg.random_data(&rp1, sizeof(uint64_t));
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t)); 
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }

  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    // uint64_t mac = rp2[k1];
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset += a[k1 * polylen + i] * t;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      offset = offset & mask;
      // offset = mac + HashP(offset);
      offset = rp2[k1] + HashP(offset);
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  // exchange the copies
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[1];
  uint64_t *outD = new uint64_t[1];
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // double startpoly = omp_get_wtime();
  IT[0] = SS_one;
  IT[1] = pos[0];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
    IT[i] = outC[1];
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  // double endspoly = omp_get_wtime();
  // cout << "select element poly:" << (endspoly-startpoly) << endl;
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
  }
  MultMode mode1 = MultMode::Alice_has_A;
  MultMode mode2 = MultMode::Bob_has_B;
  uint64_t *coe_a = new uint64_t[polylen];
  uint64_t *outDA = new uint64_t[1];
  uint64_t *outDB = new uint64_t[1];
  for (int k1 = 0; k1 < len1; k1++)
  {
    uint64_t offset = b[0];
    uint64_t t = k1;
    for (int k = 1; k < polylen; k++)
    {
      offset += b[k] * t;
      t = (t * k1) & mask;
    } 
    offset = offset & mask;
    offset = rp1 + HashP(offset);
    memcpy(coe_a, a+k1*polylen, polylen * sizeof(uint64_t));
    uint64_t tmp = 0;
    Prod_M(1, polylen, 1, coe_a, JT, outDA, Prodt[0], mode1);
    Prod_M(1, polylen, 1, JT, coe_a, outDB, Prodt[0], mode2);
    if (party == ALICE)
    { // send outDA
      Iot[0]->send_data(outDA, sizeof(uint64_t));
      Iot[0]->recv_data(&tmp, sizeof(uint64_t));
      tmp = (outDB[0] + tmp) & mask;
    }
    else
    { //send outDB
      Iot[0]->recv_data(&tmp, sizeof(uint64_t));
      Iot[0]->send_data(outDB, sizeof(uint64_t));
      tmp = (outDA[0] + tmp) & mask;
    }
    tmp = HashP(tmp);
    offset = (offset - rp2[k1] - tmp) & mask;
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
  }
  delete[] coe_a;
  delete[] outDA;
  delete[] outDB;
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  uint64_t tmp = 0;
  Prod_M(1, polylen, 1, b, IT, outD0, Prodt[0], mode1);
  Prod_M(1, polylen, 1, IT, b, outD, Prodt[0], mode2);
  if (party == ALICE)
  {
    Iot[0]->send_data(outD0, sizeof(uint64_t));
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    tmp = (outD[0] + tmp) & mask;
  }
  else
  {
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    Iot[0]->send_data(outD, sizeof(uint64_t));
    tmp = (outD0[0] + tmp) & mask;
  }
  tmp = HashP(tmp);
  mk2 = (rp1 + tmp) & mask;
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  reslen = len;
  // SS_Print(len, res);

  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] rp2;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return res;
}

uint64_t *SkylineRes_T_old(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 3;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t **a = new uint64_t*[len1];
  for (int i = 0; i < len1; i++)
  {
    a[i] = new uint64_t[polylen];
    prg.random_data(a[i], polylen * sizeof(uint64_t));
  } 
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // construct the same skyline
              // int slen = SS_L[(k1<<MAX)+k2];
              // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
              uint64_t offset = 0;
              Poly(polylen, a[k1], k2, offset);
              int lent = SS_L[(k1 * len2) + k2];
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
              }
              uint64_t *tmp1 = new uint64_t[lent];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
              }
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  cout << "select element" << endl;
  // select
  double starts = omp_get_wtime();
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  PosIndex(Q, SS_G, pos, posindex);
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[1];
  uint64_t *inB = new uint64_t[1];
  uint64_t *outC = new uint64_t[1];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = IT[i/2];
    inB[0] = IT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    IT[i] = outC[0];
  }
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD = new uint64_t[polylen];
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      int lent = SS_L[(k1 * len2) + k2];
      for (int k = 0; k < lent; k++)
      {
        tmp2[k1 * len + k] = (tmp2[k1 * len + k] + SS_Skyline[(k1 * len2) + k2][k] * posindex[1][k2]) & mask;
      }
    }
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            //a[k1]*pos[1]+b[k1])
            uint64_t mk1 = 0;
            Prod_H(polylen, a[k1], JT, outD, Prodt[itr]);
            for (int k = 0; k < polylen; k++)
            {
              mk1 = (mk1 + outD[k]) & mask;
            }
            // uint64_t offset = b[0] + b[1] * k1;  // a2*i+b2
            uint64_t offset = 0;
            Poly(polylen, b, k1, offset);
            for (int k = 0; k < len; k++)
            {
              tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset - mk1) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
            }
            if (party == ALICE)
            {
              Iot[itr]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
              Iot[itr]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
            }
            else
            {
              Iot[itr]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
              Iot[itr]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  // #pragma omp parallel for reduction(+:res)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k = 0; k < len; k++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }

  delete[] tmpt;
  delete[] tmp2;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  for (int k = 0; k < polylen; k++)
  {
      mk2 = (mk2 + outD[k]) & mask;
  }
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  reslen = len;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] pos;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return res;
}

uint64_t *SkylineRes_T(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 2;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t *rp2 = new uint64_t[len1];
  prg.random_data(rp2, len1 * sizeof(uint64_t));
  // memset(rp2, 0, len1 * sizeof(uint64_t));
  uint64_t rp1 = 0;
  prg.random_data(&rp1, sizeof(uint64_t));
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }
  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    // uint64_t mac = rp2[k1];
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset += a[k1 * polylen + i] * t;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      offset = offset & mask;
      // offset = mac + HashP(offset);
      offset = rp2[k1] + HashP(offset);
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  // exchange the copies
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[1];
  uint64_t *outD = new uint64_t[1];
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // double startpoly = omp_get_wtime();
  IT[0] = SS_one;
  IT[1] = pos[0];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
    IT[i] = outC[1];
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  // double endspoly = omp_get_wtime();
  // cout << "select element poly:" << (endspoly-startpoly) << endl;
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
  }
  // double endsaggr22 = omp_get_wtime();
  // cout << "select element aggr22:" << (endsaggr22-endspoly) << endl;
  MultMode mode1 = MultMode::Alice_has_A;
  MultMode mode2 = MultMode::Bob_has_B;
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, a, rp1, rp2)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          uint64_t *coe_a = new uint64_t[polylen];
          uint64_t *outDA = new uint64_t[1];
          uint64_t *outDB = new uint64_t[1];
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            uint64_t offset = b[0];
            uint64_t t = k1;
            for (int k = 1; k < polylen; k++)
            {
              offset += b[k] * t;
              t = (t * k1) & mask;
            } 
            offset = offset & mask;
            offset = rp1 + HashP(offset);
            memcpy(coe_a, a+k1*polylen, polylen * sizeof(uint64_t));
            uint64_t tmp = 0;
            Prod_M(1, polylen, 1, coe_a, JT, outDA, Prodt[itr], mode1);
            Prod_M(1, polylen, 1, JT, coe_a, outDB, Prodt[itr], mode2);
            if (party == ALICE)
            { // send outDA
              Iot[itr]->send_data(outDA, sizeof(uint64_t));
              Iot[itr]->recv_data(&tmp, sizeof(uint64_t));
              tmp = (outDB[0] + tmp) & mask;
            }
            else
            { //send outDB
              Iot[itr]->recv_data(&tmp, sizeof(uint64_t));
              Iot[itr]->send_data(outDB, sizeof(uint64_t));
              tmp = (outDA[0] + tmp) & mask;
            }
            tmp = HashP(tmp);
            offset = (offset - rp2[k1] - tmp) & mask;
            for (int k = 0; k < len; k++)
            {
              tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
            }
          }
          delete[] coe_a;
          delete[] outDA;
          delete[] outDB;
        }
      }
      #pragma omp taskwait
    }  
  }
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[1]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[1]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  #pragma omp parallel for /* reduction(+ : res) */
  // for (int k1 = 0; k1 < len1; k1++)
  // {
  //   uint64_t pindx = posindex[0][k1];
  //   for (int k = 0; k < len; k++)
  //   {
  //     res[k] += (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * pindx;
  //   }
  // }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  uint64_t tmp = 0;
  Prod_M(1, polylen, 1, b, IT, outD0, Prodt[0], mode1);
  Prod_M(1, polylen, 1, IT, b, outD, Prodt[0], mode2);
  if (party == ALICE)
  {
    Iot[0]->send_data(outD0, sizeof(uint64_t));
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    tmp = (outD[0] + tmp) & mask;
  }
  else
  {
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    Iot[0]->send_data(outD, sizeof(uint64_t));
    tmp = (outD0[0] + tmp) & mask;
  }
  tmp = HashP(tmp);
  mk2 = (rp1 + tmp) & mask;
  // #pragma omp parallel for
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  // SS_Print(len, res);
  reslen = len;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] rp2;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return res;
}

uint64_t *SkylineRes_T2(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 3;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t **a = new uint64_t*[len1];
  for (int i = 0; i < len1; i++)
  {
    a[i] = new uint64_t[polylen];
    prg.random_data(a[i], polylen * sizeof(uint64_t));
  } 
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // construct the same skyline
              // int slen = SS_L[(k1<<MAX)+k2];
              // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
              uint64_t offset = 0;
              Poly(polylen, a[k1], k2, offset);
              int lent = SS_L[(k1 * len2) + k2];
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
              }
              uint64_t *tmp1 = new uint64_t[lent];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
              }
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  cout << "select element" << endl;
  // select
  double starts = omp_get_wtime();
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  PosIndex_T(Q, SS_G, pos, posindex);
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[1];
  uint64_t *inB = new uint64_t[1];
  uint64_t *outC = new uint64_t[1];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = IT[i/2];
    inB[0] = IT[i - i/2];
    outC[0] = 0;
    Prod_H(1, inA, inB, outC, Prodt[0]);
    IT[i] = outC[0];
  }
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD = new uint64_t[polylen];
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      int lent = SS_L[(k1 * len2) + k2];
      for (int k = 0; k < lent; k++)
      {
        tmp2[k1 * len + k] = (tmp2[k1 * len + k] + SS_Skyline[(k1 * len2) + k2][k] * posindex[1][k2]) & mask;
      }
    }
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            //a[k1]*pos[1]+b[k1])
            uint64_t mk1 = 0;
            Prod_H(polylen, a[k1], JT, outD, Prodt[itr]);
            for (int k = 0; k < polylen; k++)
            {
              mk1 = (mk1 + outD[k]) & mask;
            }
            // uint64_t offset = b[0] + b[1] * k1;  // a2*i+b2
            uint64_t offset = 0;
            Poly(polylen, b, k1, offset);
            for (int k = 0; k < len; k++)
            {
              tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset - mk1) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
            }
            if (party == ALICE)
            {
              Iot[itr]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
              Iot[itr]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
            }
            else
            {
              Iot[itr]->recv_data(tmpt + k1 * len, len * sizeof(uint64_t));
              Iot[itr]->send_data(tmp2 + k1 * len, len * sizeof(uint64_t));
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  // #pragma omp parallel for reduction(+:res)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k = 0; k < len; k++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }

  delete[] tmpt;
  delete[] tmp2;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  for (int k = 0; k < polylen; k++)
  {
      mk2 = (mk2 + outD[k]) & mask;
  }
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  reslen = len;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] pos;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return res;
}

void SMIN(int dim, uint64_t *in, int th, uint64_t &r) {
  uint64_t y = 1ULL << lambda;
  if(dim == 1){
    r = in[0];
  }else if(dim == 2){
    uint64_t *C = new uint64_t[1];
    uint64_t *in1 = new uint64_t[1];
    uint64_t *res = new uint64_t[1];
    uint64_t *in2 = new uint64_t[1];
    if (party == ALICE)
    {
      // if ((in[0] < in[1]) && ((in[1] - in[0])<y/2))
      // {
      //   in1[0] = (mask + in[0] - in[1]) & mask; //-2>-3==(mask-2)>(mask-3)
      // }
      // else
      // {
      //   in1[0] = y + ((in[0] - in[1]) & mask); // add the same 1
      // }
      if (in[0] < in[1]){
        in1[0] = (mask + in[0] - in[1]) & mask;
      }
      else{
        in1[0] = (in[0] - in[1]) & mask; 
      }
      // in1[0] = (mask + in[0] - in[1]) & mask; 
      if (in1[0]<(y/2)){
        in1[0] = y + in1[0]; 
      }
    }
    else
    {
      // if ((in[1] < in[0]) && ((in[0] - in[1])<y/2))
      // {
      //   in1[0] = (mask + in[1] - in[0]) & mask;
      // }
      // else
      // {
      //   in1[0] = y + ((in[1] - in[0]) & mask);
      // }
      if (in[1] < in[0]){
        in1[0] = (mask + in[1] - in[0]) & mask;
      }else{
        in1[0] = (in[1] - in[0]) & mask;
      }
      // in1[0] = (mask + in[1] - in[0]) & mask;
      if (in1[0]<(y/2))
      {
        in1[0] = y + in1[0];
      }
    }
    comparison_with_eq(1, in1, C, Auxt[th]); // larger
    in2[0] = (in[1] - in[0]) & mask;
    Prod_H(1, C, in2, res, Prodt[th]);
    res[0] = (res[0] + in[0]) & mask;
    r = res[0];
    delete[] C;
    delete[] in1;
    delete[] in2;
    delete[] res;
  }else {
    uint64_t *X1 = new uint64_t[dim/2];
    memcpy(X1, in, (dim/2) * sizeof(uint64_t));
    uint64_t *X2 = new uint64_t[dim-dim/2];
    memcpy(X2, in + (dim/2), (dim-dim/2) * sizeof(uint64_t));
    uint64_t C1 = 0;
    SMIN(dim/2, X1, th, C1);
    uint64_t C2 = 0;
    SMIN(dim-dim/2, X2, th, C2);
    uint64_t *C = new uint64_t[1];
    uint64_t *in1 = new uint64_t[1];
    uint64_t *in2 = new uint64_t[1];
    uint64_t *res = new uint64_t[1];
    if (party == ALICE)
    {
      // if (C1 < C2)
      // if ((C1 < C2) && ((C2 - C1)<y/2))
      // {
      //   in1[0] = (mask + C1 - C2) & mask; //-2>-3==(mask-2)>(mask-3)
      // }
      // else
      // {
      //   in1[0] = y + ((C1 - C2) & mask); // add the same 1
      // }
      if (C1 < C2){
        in1[0] = (mask + C1 - C2) & mask;
      }else{
        in1[0] = (C1 - C2) & mask;
      }
      // in1[0] = (mask + C1 - C2) & mask;
      if (in1[0]<(y/2))
      {
        in1[0] = y + in1[0]; 
      }
    }
    else
    {
      // if (C2 < C1)
      // if ((C2 < C1) && ((C1 - C2)<y/2))
      // {
      //   in1[0] = (mask + C2 - C1) & mask;
      // }
      // else
      // {
      //   in1[0] = y + ((C2 - C1) & mask);
      // }
      if (C2 < C1){
        in1[0] = (mask + C2 - C1) & mask;
      }else{
        in1[0] = (C2 - C1) & mask;
      }
      // in1[0] = (mask + C2 - C1) & mask;
      if (in1[0]<(y/2))
      {
        in1[0] = y + in1[0];
      }
    }
    comparison_with_eq(1, in1, C, Auxt[th]); // larger
    in2[0] = (C2 - C1) & mask;
    Prod_H(1, C, in2, res, Prodt[th]);
    res[0] = (res[0] + C1) & mask;
    r = res[0];
    delete[] C;
    delete[] in1;
    delete[] in2;
    delete[] res;
    delete[] X1;
    delete[] X2;
  }
}

void SMIN1(int dim, uint64_t *in, int th, uint64_t &r) {
  uint64_t y = 1ULL << (lambda - 1);
  if(dim == 1){
    r = in[0];
  }else if(dim == 2){
    uint64_t *C = new uint64_t[1];
    uint64_t *in1 = new uint64_t[1];
    uint64_t *res = new uint64_t[1];
    uint64_t *in2 = new uint64_t[1];
    uint64_t *fin1 = new uint64_t[1];
    memset(in2, 0, sizeof(uint64_t));
    uint64_t *fout1 = new uint64_t[1];
    if (party == ALICE)
    {
      in1[0] = (in[0] - in[1]) & mask;
      if ((in[0] < in[1]) && (in[1] - in[0] < y))
      {
        fin1[0] = 1;
      }
    }
    else
    {
      in1[0] = (in[1] - in[0]) & mask;
      if ((in[1] < in[0]) && (in[0] - in[1] < y))
      {
        fin1[0] = 1;
      }
    }
    comparison_with_eq(1, in1, C, Auxt[th]); // larger
    comparison_with_eq(1, in1, C, Auxt[th]);
    in2[0] = (in[1] - in[0]) & mask;
    Prod_H(1, C, in2, res, Prodt[th]);
    res[0] = (res[0] + in[0]) & mask;
    r = res[0];
    delete[] C;
    delete[] in1;
    delete[] in2;
    delete[] res;
  }else {
    uint64_t *X1 = new uint64_t[dim/2];
    memcpy(X1, in, (dim/2) * sizeof(uint64_t));
    uint64_t *X2 = new uint64_t[dim-dim/2];
    memcpy(X2, in + (dim/2), (dim-dim/2) * sizeof(uint64_t));
    uint64_t C1 = 0;
    SMIN(dim/2, X1, th, C1);
    uint64_t C2 = 0;
    SMIN(dim-dim/2, X2, th, C2);
    uint64_t *C = new uint64_t[1];
    uint64_t *in1 = new uint64_t[1];
    uint64_t *in2 = new uint64_t[1];
    uint64_t *res = new uint64_t[1];
    if (party == ALICE)
    {
      if (C1 < C2)
      {
        in1[0] = (mask + C1 - C2) & mask; //-2>-3==(mask-2)>(mask-3)
      }
      else
      {
        in1[0] = y + ((C1 - C2) & mask); // add the same 1
      }
    }
    else
    {
      if (C2 < C1)
      {
        in1[0] = (mask + C2 - C1) & mask;
      }
      else
      {
        in1[0] = y + ((C2 - C1) & mask);
      }
    }
    comparison_with_eq(1, in1, C, Auxt[th]); // larger
    in2[0] = (C2 - C1) & mask;
    Prod_H(1, C, in2, res, Prodt[th]);
    res[0] = (res[0] + C1) & mask;
    r = res[0];
    delete[] C;
    delete[] in1;
    delete[] in2;
    delete[] res;
    delete[] X1;
    delete[] X2;
  }
}

void TwoMIN(int th, uint64_t s1, uint64_t s2, uint64_t *pt1, uint64_t *pt2, uint64_t *St1, uint64_t *St2, uint64_t &rS, uint64_t * &rP, uint64_t * &rT) {
  uint64_t *C = new uint64_t[1];
  uint64_t *in0 = new uint64_t[1];
  uint64_t *in1 = new uint64_t[1];
  in0[0] = s1;
  in1[0] = s2;
  uint64_t *res = new uint64_t[1 + 2 * m];
  uint64_t *in2 = new uint64_t[1 + 2 * m];
  uint64_t *Ct = new uint64_t[1 + 2 * m];
  comparison_with_eq2N(1, in0, in1, C, Auxt[th]); // s1 larger s2
  // if (party == ALICE)
  // {
  //   int t = 1;
  //   uint64_t *tC1 =  new uint64_t[t]; 
  //   uint64_t *tC2 =  new uint64_t[t]; 
  //   uint64_t *tC3 =  new uint64_t[t]; 
  //   Iot[th]->recv_data(tC1, t * sizeof(uint64_t));
  //   Iot[th]->recv_data(tC2, t * sizeof(uint64_t));
  //   Iot[th]->recv_data(tC3, t * sizeof(uint64_t));
  //   Iot[th]->send_data(C, t * sizeof(uint64_t));
  //   Iot[th]->send_data(&s1, t * sizeof(uint64_t));
  //   Iot[th]->send_data(&s2, t * sizeof(uint64_t));
  //   for (int j = 0; j < t; j++) {
  //     tC1[j] = (C[j] + tC1[j]) & mask;
  //     // cout<< (C1[j])<<",";
  //   }
  //   for (int j = 0; j < t; j++) {
  //     tC2[j] = (s1 + tC2[j]) & mask;
  //     // cout<< (C2[j])<<",";
  //   }
  //   for (int j = 0; j < t; j++) {
  //     tC3[j] = (s2 + tC3[j]) & mask;
  //     // cout<< (C3[j])<<",";
  //   }
  //   if(tC1[0]!=(tC2[0]>=tC3[0])){
  //     cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
  //   }
  //   delete[] tC1;
  //   delete[] tC2;
  //   delete[] tC3;
  // }
  // else
  // {
  //   int t = 1;
  //   uint64_t *tC1 =  new uint64_t[t]; 
  //   uint64_t *tC2 =  new uint64_t[t]; 
  //   uint64_t *tC3 =  new uint64_t[t]; 
  //   Iot[th]->send_data(C, t * sizeof(uint64_t));
  //   Iot[th]->send_data(&s1, t * sizeof(uint64_t));
  //   Iot[th]->send_data(&s2, t * sizeof(uint64_t));
  //   Iot[th]->recv_data(tC1, t * sizeof(uint64_t));
  //   Iot[th]->recv_data(tC2, t * sizeof(uint64_t));
  //   Iot[th]->recv_data(tC3, t * sizeof(uint64_t));
  //   for (int j = 0; j < t; j++) {
  //     tC1[j] = (C[j] + tC1[j]) & mask;
  //     // cout<< (C1[j])<<",";
  //   }
  //   for (int j = 0; j < t; j++) {
  //     tC2[j] = (s1 + tC2[j]) & mask;
  //     // cout<< (C2[j])<<",";
  //   }
  //   for (int j = 0; j < t; j++) {
  //     tC3[j] = (s2 + tC3[j]) & mask;
  //     // cout<< (C3[j])<<",";
  //   }
  //   if(tC1[0]!=(tC2[0]>=tC3[0])){
  //     cout<<"!!!!!!!!!!!!!!!"<<endl;
  //   }
  //   delete[] tC1;
  //   delete[] tC2;
  //   delete[] tC3;
  // }

  in2[0] = (s2 - s1) & mask;
  Ct[0] = C[0] & mask;
  for (int k = 0; k < m; k++)
  {
    in2[1 + k] = (pt2[k] - pt1[k]) & mask;
    in2[1 + m + k] = (St2[k] - St1[k]) & mask;
    Ct[1 + k] = Ct[0];
    Ct[1 + m + k] = Ct[0];
  }
  Prod_H(1 + 2 * m, Ct, in2, res, Prodt[th]);
  rS = (res[0] + s1) & mask;
  for (int k = 0; k < m; k++)
  {
    rP[k] = (res[1 + k] + pt1[k]) & mask;
    rT[k] = (res[1 + m + k] + St1[k]) & mask;
  }
  delete[] C;
  delete[] Ct;
  delete[] in0;
  delete[] in1;
  delete[] in2;
  delete[] res;
}

void SMIN(int dims, int dime, uint64_t *S, uint64_t **pt, uint64_t **St, uint64_t &rS, uint64_t * &rP, uint64_t * &rT) {
  // [dims, dime)
  int dimt = dime - dims;
  // cout<< dimt << endl;
  if(dimt <= 32){
    int lt = dimt / 2;
    if(dimt == 1){
      rS = S[dims];
      memcpy(rP, pt[dims], m * sizeof(uint64_t));
      memcpy(rT, St[dims], m * sizeof(uint64_t));
      // PrintSKyline(0, rS);
    }else if(dimt == 2){
      TwoMIN(0, S[dims], S[dims+1], pt[dims], pt[dims+1], St[dims], St[dims+1], rS, rP, rT);
      // PrintSKyline(0, rS);
    }else if(dimt % 2 == 1){
      uint64_t rSt = 0;
      uint64_t *rPt = new uint64_t[m];
      uint64_t *rTt = new uint64_t[m];
      // PrintSKyline(pt[dime-1]);
      SMIN(dims, dime - 1, S, pt, St, rSt, rPt, rTt);
      TwoMIN(0, rSt, S[dime-1], rPt, pt[dime-1], rTt, St[dime-1], rS, rP, rT);
      // PrintSKyline(0, rS);
      delete[] rPt;
      delete[] rTt;
    }
    else {
      uint64_t *rSt = new uint64_t[lt];
      uint64_t **rPt = new uint64_t*[lt];
      uint64_t **rTt = new uint64_t*[lt];
      for (int j = 0; j < lt; j++)
      {
        int itr = dims + 2 * j;
        rSt[j] = 0;
        rPt[j] = new uint64_t[m];
        rTt[j] = new uint64_t[m];
        TwoMIN(j, S[itr], S[itr+1], pt[itr], pt[itr+1], St[itr], St[itr+1], rSt[j], rPt[j], rTt[j]);
        // cout<<itr<<","<<(itr+1)<<",";
        // PrintSKyline(j, rSt[j]);
      }
      SMIN(0, lt, rSt, rPt, rTt, rS, rP, rT);
      // cout<<"32 all:";
      // PrintSKyline(0, rS);
      delete rSt;
      for(int itr = 0; itr< lt; itr++)
      {
        delete[] rPt[itr];
        delete[] rTt[itr];
      }
      delete[] rPt;
      delete[] rTt;
    }
  }else {
    int lt = dimt / 32;
    uint64_t *rSt;
    uint64_t **rPt;
    uint64_t **rTt;
    if(dimt % 32 == 0){
      rSt = new uint64_t[lt];
      rPt = new uint64_t*[lt];
      rTt = new uint64_t*[lt];
    }else {
      rSt = new uint64_t[lt+1];
      rPt = new uint64_t*[lt+1];
      rTt = new uint64_t*[lt+1];
    }
    for (int itr = 0; itr < lt; itr++)
    {
      rSt[itr] = 0;
      rPt[itr] = new uint64_t[m];
      rTt[itr] = new uint64_t[m];
      SMIN(32*itr, 32*(itr+1), S, pt, St, rSt[itr], rPt[itr], rTt[itr]);
    }
    if(dimt % 32 == 0){
      SMIN(0, lt, rSt, rPt, rTt, rS, rP, rT);
    } else {
      rSt[lt] = 0;
      rPt[lt] = new uint64_t[m];
      rTt[lt] = new uint64_t[m];
      SMIN(32*lt, dime, S, pt, St, rSt[lt], rPt[lt], rTt[lt]);
      SMIN(0, lt+1, rSt, rPt, rTt, rS, rP, rT);
    }
    // cout<<"all:";
    // PrintSKyline(0, rS);
    delete rSt;
    for(int itr = 0; itr< lt; itr++)
    {
      delete[] rPt[itr];
      delete[] rTt[itr];
    }
    delete[] rPt;
    delete[] rTt;
  }
}

//old SMIN_T
// void SMIN_T(int dims, int dime, uint64_t *S, uint64_t **pt, uint64_t **St, uint64_t &rS, uint64_t * &rP, uint64_t * &rT) {
//   // [dims, dime)
//   int thr = 2*THs;
//   int dimt = dime - dims;
//   // cout<< dimt << endl;
//   if(dimt <= thr){
//     int lt = dimt / 2;
//     if(dimt == 1){
//       rS = S[dims];
//       memcpy(rP, pt[dims], m * sizeof(uint64_t));
//       memcpy(rT, St[dims], m * sizeof(uint64_t));
//       // PrintSKyline(0, rS);
//     }else if(dimt == 2){
//       TwoMIN(0, S[dims], S[dims+1], pt[dims], pt[dims+1], St[dims], St[dims+1], rS, rP, rT);
//       // PrintSKyline(0, rS);
//     }else if(dimt % 2 == 1){
//       // uint64_t rSt = 0;
//       // uint64_t *rPt = new uint64_t[m];
//       // uint64_t *rTt = new uint64_t[m];
//       // // PrintSKyline(pt[dime-1]);
//       // SMIN_T(dims, dime - 1, S, pt, St, rSt, rPt, rTt);
//       // TwoMIN(0, rSt, S[dime-1], rPt, pt[dime-1], rTt, St[dime-1], rS, rP, rT);
//       // // PrintSKyline(0, rS);
//       // delete[] rPt;
//       // delete[] rTt;
//       uint64_t *rSt = new uint64_t[lt+1];
//       uint64_t **rPt = new uint64_t*[lt+1];
//       uint64_t **rTt = new uint64_t*[lt+1];
//       #pragma omp parallel num_threads(lt)
//       {
//         #pragma omp single 
//         {
//           for (int j = 0; j < lt; j++)
//           {
//             #pragma omp task firstprivate(j, lt)
//             {
//               int itr = dims + 2 * j;
//               rSt[j] = 0;
//               rPt[j] = new uint64_t[m];
//               rTt[j] = new uint64_t[m];
//               TwoMIN(j, S[itr], S[itr+1], pt[itr], pt[itr+1], St[itr], St[itr+1], rSt[j], rPt[j], rTt[j]);
//             }
//           }
//           #pragma omp taskwait
//         }  
//       }
//       rSt[lt] = S[dime-1];
//       rPt[lt] = new uint64_t[m];
//       rTt[lt] = new uint64_t[m];
//       memcpy(rPt[lt], pt[dime-1], m * sizeof(uint64_t));
//       memcpy(rTt[lt], St[dime-1], m * sizeof(uint64_t));
//       SMIN_T(0, lt+1, rSt, rPt, rTt, rS, rP, rT);
//       // PrintSKyline(0, rS);
//       delete rSt;
//       for(int itr = 0; itr< lt; itr++)
//       {
//         delete[] rPt[itr];
//         delete[] rTt[itr];
//       }
//       delete[] rPt;
//       delete[] rTt;
//     }
//     else {
//       uint64_t *rSt = new uint64_t[lt];
//       uint64_t **rPt = new uint64_t*[lt];
//       uint64_t **rTt = new uint64_t*[lt];
//       #pragma omp parallel num_threads(lt)
//       {
//         #pragma omp single 
//         {
//           for (int j = 0; j < lt; j++)
//           {
//             #pragma omp task firstprivate(j, lt)
//             {
//               int itr = dims + 2 * j;
//               rSt[j] = 0;
//               rPt[j] = new uint64_t[m];
//               rTt[j] = new uint64_t[m];
//               TwoMIN(j, S[itr], S[itr+1], pt[itr], pt[itr+1], St[itr], St[itr+1], rSt[j], rPt[j], rTt[j]);
//               // cout<<itr<<","<<(itr+1)<<",";
//               // PrintSKyline(j, rSt[j]);
//             }
//           }
//           #pragma omp taskwait
//         }  
//       }
//       SMIN_T(0, lt, rSt, rPt, rTt, rS, rP, rT);
//       // cout<<"32 all:";
//       // PrintSKyline(0, rS);
//       delete rSt;
//       for(int itr = 0; itr< lt; itr++)
//       {
//         delete[] rPt[itr];
//         delete[] rTt[itr];
//       }
//       delete[] rPt;
//       delete[] rTt;
//     }
//   }else {
//     int lt = dimt / thr;
//     uint64_t *rSt;
//     uint64_t **rPt;
//     uint64_t **rTt;
//     if(dimt % thr == 0){
//       rSt = new uint64_t[lt];
//       rPt = new uint64_t*[lt];
//       rTt = new uint64_t*[lt];
//     }else {
//       rSt = new uint64_t[lt+1];
//       rPt = new uint64_t*[lt+1];
//       rTt = new uint64_t*[lt+1];
//     }
//     for (int itr = 0; itr < lt; itr++)
//     {
//       rSt[itr] = 0;
//       rPt[itr] = new uint64_t[m];
//       rTt[itr] = new uint64_t[m];
//       SMIN_T(thr*itr, thr*(itr+1), S, pt, St, rSt[itr], rPt[itr], rTt[itr]);
//     }
//     if(dimt % thr == 0){
//       SMIN_T(0, lt, rSt, rPt, rTt, rS, rP, rT);
//     } else {
//       rSt[lt] = 0;
//       rPt[lt] = new uint64_t[m];
//       rTt[lt] = new uint64_t[m];
//       SMIN_T(thr*lt, dime, S, pt, St, rSt[lt], rPt[lt], rTt[lt]);
//       SMIN_T(0, lt+1, rSt, rPt, rTt, rS, rP, rT);
//     }
//     // cout<<"all:";
//     // PrintSKyline(0, rS);
//     delete rSt;
//     for(int itr = 0; itr< lt; itr++)
//     {
//       delete[] rPt[itr];
//       delete[] rTt[itr];
//     }
//     delete[] rPt;
//     delete[] rTt;
//   }
// }

void SMIN_T(int dims, int dime, uint64_t *S, uint64_t **pt, uint64_t **St, uint64_t &rS, uint64_t * &rP, uint64_t * &rT) {
  // [dims, dime)
  // int thr = 2 * THs;
  int dimt = dime - dims;
  // cout<< dimt << endl;
  if(dimt <= THs){
    // double xx3 = omp_get_wtime();
    int lt = dimt / 2;
    if(dimt == 1){
      rS = S[dims];
      memcpy(rP, pt[dims], m * sizeof(uint64_t));
      memcpy(rT, St[dims], m * sizeof(uint64_t));
      // PrintSKyline(0, rS);
    }else if(dimt == 2){
      TwoMIN(0, S[dims], S[dims+1], pt[dims], pt[dims+1], St[dims], St[dims+1], rS, rP, rT);
      // PrintSKyline(0, rS);
    }else if(dimt % 2 == 1){
      // uint64_t rSt = 0;
      // uint64_t *rPt = new uint64_t[m];
      // uint64_t *rTt = new uint64_t[m];
      // // PrintSKyline(pt[dime-1]);
      // SMIN_T(dims, dime - 1, S, pt, St, rSt, rPt, rTt);
      // TwoMIN(0, rSt, S[dime-1], rPt, pt[dime-1], rTt, St[dime-1], rS, rP, rT);
      // // PrintSKyline(0, rS);
      // delete[] rPt;
      // delete[] rTt;
      uint64_t *rSt = new uint64_t[lt+1];
      uint64_t **rPt = new uint64_t*[lt+1];
      uint64_t **rTt = new uint64_t*[lt+1];
      #pragma omp parallel num_threads(lt)
      {
        #pragma omp single 
        {
          for (int j = 0; j < lt + 1; j++)
          {
            #pragma omp task firstprivate(j, lt)
            {
              if(j < lt){
                int itr = dims + 2 * j;
                rSt[j] = 0;
                rPt[j] = new uint64_t[m];
                rTt[j] = new uint64_t[m];
                TwoMIN(j, S[itr], S[itr+1], pt[itr], pt[itr+1], St[itr], St[itr+1], rSt[j], rPt[j], rTt[j]);
              }else {
                rSt[lt] = S[dime-1];
                rPt[lt] = new uint64_t[m];
                rTt[lt] = new uint64_t[m];
                memcpy(rPt[lt], pt[dime-1], m * sizeof(uint64_t));
                memcpy(rTt[lt], St[dime-1], m * sizeof(uint64_t));     
              }
            }
          }
          #pragma omp taskwait
        }  
      }
      SMIN_T(0, lt+1, rSt, rPt, rTt, rS, rP, rT);
      // PrintSKyline(0, rS);
      delete rSt;
      for(int itr = 0; itr< lt; itr++)
      {
        delete[] rPt[itr];
        delete[] rTt[itr];
      }
      delete[] rPt;
      delete[] rTt;
    }
    else {
      uint64_t *rSt = new uint64_t[lt];
      uint64_t **rPt = new uint64_t*[lt];
      uint64_t **rTt = new uint64_t*[lt];
      #pragma omp parallel num_threads(lt)
      {
        #pragma omp single 
        {
          for (int j = 0; j < lt; j++)
          {
            #pragma omp task firstprivate(j, lt)
            {
              int itr = dims + 2 * j;
              rSt[j] = 0;
              rPt[j] = new uint64_t[m];
              rTt[j] = new uint64_t[m];
              TwoMIN(j, S[itr], S[itr+1], pt[itr], pt[itr+1], St[itr], St[itr+1], rSt[j], rPt[j], rTt[j]);
              // cout<<itr<<","<<(itr+1)<<",";
              // PrintSKyline(j, rSt[j]);
            }
          }
          #pragma omp taskwait
        }  
      }
      SMIN_T(0, lt, rSt, rPt, rTt, rS, rP, rT);
      // cout<<"32 all:";
      // PrintSKyline(0, rS);
      delete rSt;
      for(int itr = 0; itr< lt; itr++)
      {
        delete[] rPt[itr];
        delete[] rTt[itr];
      }
      delete[] rPt;
      delete[] rTt;
    }
    // rS = S[dims];
    // memcpy(rP, pt[dims], m * sizeof(uint64_t));
    // memcpy(rT, St[dims], m * sizeof(uint64_t));
    // for (int i = dims + 1; i < dime; i++)
    // {
    //   TwoMIN(0, rS, S[i], rP, pt[i], rT, St[i], rS, rP, rT);
    // }
    // double xx4 = omp_get_wtime();
    // cout << "dy element Smin2:" << (xx4 - xx3) << " "<< dimt << endl;
  }else {
    // double xx3 = omp_get_wtime();
    uint64_t *rSt = new uint64_t[THs];
    uint64_t **rPt = new uint64_t*[THs];
    uint64_t **rTt = new uint64_t*[THs];
    #pragma omp parallel num_threads(THs)
    {
      #pragma omp single 
      {
        for (int itr = 0; itr < THs; itr++)
        {
          #pragma omp task firstprivate(itr, THs)
          {
            int lendt = (((dimt) * itr)/ THs) + dims;
            int lenut = ((dimt) * (itr + 1))/ THs + dims;
            rSt[itr] = S[lendt];
            rPt[itr] = new uint64_t[m];
            rTt[itr] = new uint64_t[m];
            memcpy(rPt[itr], pt[lendt], m * sizeof(uint64_t));
            memcpy(rTt[itr], St[lendt], m * sizeof(uint64_t));
            for (int i = lendt + 1; i < lenut; i++)
            {
              TwoMIN(itr, rSt[itr], S[i], rPt[itr], pt[i], rTt[itr], St[i], rSt[itr], rPt[itr], rTt[itr]);
            }
          }
        }
        #pragma omp taskwait
      }  
    }
    // double xx4 = omp_get_wtime();
    // cout << "dy element Smin1:" << (xx4 - xx3) << endl;
    SMIN_T(0, THs, rSt, rPt, rTt, rS, rP, rT);
    // cout<<"all:";
    // PrintSKyline(0, rS);
    delete rSt;
    for(int itr = 0; itr< THs; itr++)
    {
      delete[] rPt[itr];
      delete[] rTt[itr];
    }
    delete[] rPt;
    delete[] rTt;
  }
}

void SDOMbyMin(int dim, uint64_t *a, uint64_t *b, int th, uint64_t &r){
  // A and B
  uint64_t y = 1ULL << lambda;
  uint64_t *in1 = new uint64_t[m];
  uint64_t *sleq = new uint64_t[m];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *res = new uint64_t[1];
  for (int j = 0; j < dim; j++) {
    if (party == ALICE)
    {
      // if (b[j] < a[j])
      // if ((b[j] < a[j]) && ((a[j] - b[j])<y/2))
      // {
      //   in1[j] = (mask + b[j] - a[j]) & mask; //-2>-3==(mask-2)>(mask-3)
      // }
      // else
      // {
      //   in1[j] = y + ((b[j] - a[j]) & mask); // add the same 1
      // }
      if (b[j] < a[j]){
        in1[j] = (mask + b[j] - a[j]) & mask;
      }else{
        in1[j] = (b[j] - a[j]) & mask;
      }
      // in1[j] = (mask + b[j] - a[j]) & mask;
      if (in1[j]<(y/2))
      {
        in1[j] = y + in1[j];
      }
    }
    else
    {
      // if (a[j] < b[j])
      // if ((a[j] < b[j]) && ((b[j] - a[j])<y/2))
      // {
      //   in1[j] = (mask + a[j] - b[j]) & mask;
      // }
      // else
      // {
      //   in1[j] = y + ((a[j] - b[j]) & mask);
      // }
      if (a[j] < b[j]){
        in1[j] = (mask + a[j] - b[j]) & mask;
      }else{
        in1[j] = (a[j] - b[j]) & mask;
      }
      // in1[j] = (mask + a[j] - b[j]) & mask;
      if (in1[j]<(y/2))
      {
        in1[j] = y + in1[j];
      }
    }
  }
  comparison_with_eq(dim, in1, sleq, Auxt[th]); //  b larger and leq a
  uint64_t *in3 = new uint64_t[1];
  res[0] = sleq[0];
  for (int j = 1; j < m; j++) {
    in2[0] = sleq[j];
    in3[0] = res[0];
    Prod_H(1, in2, in3, res, Prodt[th]);
  }
  //res:*sleq
  // A
  uint64_t alpha = a[0];
  uint64_t beta = b[0];
  for (int j = 0; j < m; j++) {
    alpha = (alpha + a[j]) & mask;
    beta = (beta + b[j]) & mask;
  }
  // A and B
  if (party == ALICE)
  {
    // if (alpha < beta)
    // if ((alpha < beta) && ((beta - alpha)<y/2))
    // {
    //   in2[0] = (mask + alpha - beta) & mask; //-2>-3==(mask-2)>(mask-3)
    // }
    // else
    // {
    //   in2[0] = y + ((alpha - beta) & mask); // add the same 1
    // }
    if (alpha < beta){
      in2[0] = (mask + alpha - beta) & mask;
    }else{
      in2[0] = (alpha - beta) & mask;
    }
    // in2[0] = (mask + alpha - beta) & mask;
    if (in2[0]<(y/2))
    {
      in2[0] = y + in2[0]; 
    }
  }
  else
  {
    // if (beta < alpha)
    // if ((beta < alpha) && ((alpha - beta)<y/2))
    // {
    //   in2[0] = (mask + beta - alpha) & mask;
    // }
    // else
    // {
    //   in2[0] = y + ((beta - alpha) & mask);
    // }
    if (beta < alpha){
      in2[0] = (mask + beta - alpha) & mask;
    }else{
      in2[0] = (beta - alpha) & mask;
    }
    // in2[0] = (mask + beta - alpha) & mask;
    if (in2[0]<(y/2))
    {
      in2[0] = y + in2[0];
    }
  }
  comparison_with_eq(1, in2, in3, Auxt[th]); //  a<b=(1-(a>=b))
  if (party == ALICE)
  {
    in3[0] = (1 - in3[0]) & mask; 
  }
  else
  {
    in3[0] = (0 - in3[0]) & mask; 
  }
  in2[0] = res[0];
  Prod_H(1, in2, in3, res, Prodt[th]);
  r = res[0];
  delete[] in1;
  delete[] in2;
  delete[] in3;
  delete[] sleq;
  delete[] res;
}

void SDOMbyMin(int dimt, uint64_t Tmax, uint64_t *STmin, uint64_t **St, uint64_t Smin, uint64_t * &rS) {
  uint64_t *Sig = new uint64_t[dimt];
  uint64_t *Dlt = new uint64_t[dimt];
  uint64_t *Phi = new uint64_t[dimt];
  uint64_t *T = new uint64_t[dimt];
  uint64_t *res = new uint64_t[dimt];
  uint64_t SS_one = 0;
  // uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    // Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    // SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    // prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    // Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;    
  }
  // uint64_t Fla = SS_zero;
  uint64_t Fla = 0;
  //Print S
  // if (party == ALICE)
  // {
  //   Iot[2]->send_data(rS, dimt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[dimt]; 
  //   Iot[2]->recv_data(STTT, dimt * sizeof(uint64_t));
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((STTT[i] + rS[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(STmin, m * sizeof(uint64_t));
  //   Iot[1]->send_data(&Smin, sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STminT =  new uint64_t[m];    
  //   Iot[0]->recv_data(STminT, m * sizeof(uint64_t));
  //   for (int i = 0; i < m; i++) {
  //     cout<< ((STminT[i] + STmin[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STminT;
  //   uint64_t SminT =  0; 
  //   Iot[1]->recv_data(&SminT, sizeof(uint64_t));
  //   cout<< ((Smin + SminT) & mask) << endl;
  // }
  int thr = (dimt > THs) ? THs : dimt;
  for (int itr = 0; itr < thr; itr++)
  {
    int lendt = (((dimt - 1) * itr)/ thr) + 1;
    int lenut = ((dimt - 1) * (itr + 1))/ thr;
    if(itr == 0){
      lendt = 0;
    }
    int kk = lenut-lendt+1;
    // cout<<itr<<" "<<kk<<endl;
    uint64_t *in0 = new uint64_t[kk*(m+1)];
    uint64_t *in1 = new uint64_t[kk*(m+1)];
    uint64_t *in2 = new uint64_t[kk];
    uint64_t *in3 = new uint64_t[kk];
    uint64_t *sleq1 = new uint64_t[kk*(m+1)];
    uint64_t *sleq2 = new uint64_t[kk];
    for (int j = lendt; j <= lenut; j++) {
      in0[j-lendt] = Smin;
      in1[j-lendt] = rS[j];
      for (int k = 0; k < m; k++) {
        in0[(k+1)*kk+j-lendt] = St[j][k];
        in1[(k+1)*kk+j-lendt] = STmin[k];
      }
    }
    comparison_with_eq2N(kk*(m+1), in0, in1, sleq1, Auxt[itr]);
    memcpy(in2, sleq1+kk, kk * sizeof(uint64_t));
    memcpy(in3, sleq1+kk*2, kk * sizeof(uint64_t));
    Prod_H(kk, in2, in3, sleq2, Prodt[itr]);
    memcpy(Sig+lendt, sleq1, kk * sizeof(uint64_t));
    memcpy(Dlt+lendt, sleq2, kk * sizeof(uint64_t));
    // for (int j = lendt; j <= lenut; j++) {
    //   Sig[j] = sleq1[j-lendt];
    //   Dlt[j] = sleq2[j-lendt];
    // }
    delete[] in0;
    delete[] in1;
    delete[] in2;
    delete[] in3;
    delete[] sleq1;
    delete[] sleq2;
  }
  uint64_t *in1 = new uint64_t[2];
  uint64_t *in2 = new uint64_t[2];
  uint64_t *in3 = new uint64_t[2];
  for (int j = 0; j < dimt; j++) {
    in1[0] = Sig[j] & mask;
    in1[1] = Dlt[j] & mask;
    in2[0] = (SS_one - Fla) & mask;
    in2[1] = (SS_one - Sig[j]) & mask;
    Prod_H(2, in1, in2, in3, Prodt[0]);//first = in3[0], Dom = in3[1];
    Fla = (Fla + in3[0]) & mask;
    Phi[j] = (in3[0] + in3[1]) & mask;
    T[j] = (Tmax - rS[j]) & mask;
  }
  //Print Phi
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(Phi, dimt * sizeof(uint64_t));
  //   Iot[1]->send_data(Sig, dimt * sizeof(uint64_t));
  //   Iot[2]->send_data(Dlt, dimt * sizeof(uint64_t));
  //   // Iot[0]->send_data(&Tmax, sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *PhiT =  new uint64_t[dimt]; 
  //   uint64_t *SigT =  new uint64_t[dimt]; 
  //   uint64_t *DltT =  new uint64_t[dimt]; 
  //   // uint64_t TmaxT =  0; 
  //   Iot[0]->recv_data(PhiT, dimt * sizeof(uint64_t));
  //   Iot[1]->recv_data(SigT, dimt * sizeof(uint64_t));
  //   Iot[2]->recv_data(DltT, dimt * sizeof(uint64_t));
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((PhiT[i] + Phi[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((SigT[i] + Sig[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((DltT[i] + Dlt[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] PhiT;
  //   delete[] SigT;
  //   delete[] DltT;
  //   // Iot[0]->recv_data(&TmaxT, sizeof(uint64_t));
  //   // cout<< ((Tmax + TmaxT) & mask) << endl;
  // }
  Prod_H(dimt, T, Phi, res, Prodt[0]);
  for (int j = 0; j < dimt; j++) {
    rS[j] = (rS[j] + res[j]) & mask;
  }
  delete[] in1;
  delete[] in2;
  delete[] in3;
  delete[] Sig;
  delete[] Dlt;
  delete[] Phi;
  delete[] T;
  delete[] res;
}

//old SDOM
// void SDOMbyMin_T(int dimt, uint64_t Tmax, uint64_t *STmin, uint64_t **St, uint64_t Smin, uint64_t * &rS) {
//   uint64_t *Sig = new uint64_t[dimt];
//   uint64_t *Dlt = new uint64_t[dimt];
//   uint64_t *Phi = new uint64_t[dimt];
//   uint64_t *T = new uint64_t[dimt];
//   uint64_t *res = new uint64_t[dimt];
//   uint64_t SS_one = 0;
//   // uint64_t SS_zero = 0;
//   if (party == ALICE)
//   {
//     prg.random_data(&SS_one, sizeof(uint64_t));
//     Iot[0]->send_data(&SS_one, sizeof(uint64_t));
//     // Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
//     // SS_zero = (0 - SS_zero) & mask;
//   }
//   else
//   {
//     // prg.random_data(&SS_zero, sizeof(uint64_t));
//     Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
//     // Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
//     SS_one = (1 - SS_one) & mask;    
//   }
//   // uint64_t Fla = SS_zero;
//   uint64_t Fla = 0;
//   //Print S
//   // if (party == ALICE)
//   // {
//   //   Iot[2]->send_data(rS, dimt * sizeof(uint64_t));
//   // }
//   // else
//   // {
//   //   uint64_t *STTT =  new uint64_t[dimt]; 
//   //   Iot[2]->recv_data(STTT, dimt * sizeof(uint64_t));
//   //   for (int i = 0; i < dimt; i++) {
//   //     cout<< ((STTT[i] + rS[i]) & mask) << ",";
//   //   }
//   //   cout<<endl;
//   //   delete[] STTT;
//   // }
//   // if (party == ALICE)
//   // {
//   //   Iot[0]->send_data(STmin, m * sizeof(uint64_t));
//   //   Iot[1]->send_data(&Smin, sizeof(uint64_t));
//   // }
//   // else
//   // {
//   //   uint64_t *STminT =  new uint64_t[m];    
//   //   Iot[0]->recv_data(STminT, m * sizeof(uint64_t));
//   //   for (int i = 0; i < m; i++) {
//   //     cout<< ((STminT[i] + STmin[i]) & mask) << ",";
//   //   }
//   //   cout<<endl;
//   //   delete[] STminT;
//   //   uint64_t SminT =  0; 
//   //   Iot[1]->recv_data(&SminT, sizeof(uint64_t));
//   //   cout<< ((Smin + SminT) & mask) << endl;
//   // } 
//   int thr = (dimt > THs) ? THs : dimt;
//   #pragma omp parallel num_threads(thr)
//   {
//     #pragma omp single 
//     {
//       for (int itr = 0; itr < thr; itr++)
//       {
//         #pragma omp task firstprivate(itr, thr)
//         {
//           int lendt = (((dimt) * itr)/ thr);
//           int lenut = ((dimt) * (itr + 1))/ thr;
//           // int lendt = (((dimt - 1) * itr)/ thr) + 1;
//           // int lenut = ((dimt - 1) * (itr + 1))/ thr;
//           // if(itr == 0){
//           //   lendt = 0;
//           // }
//           // int kk = lenut-lendt+1;
//           int kk = lenut-lendt;
//           // cout<<itr<<" "<<kk<<endl;
//           uint64_t *in0 = new uint64_t[kk*(m+1)];
//           uint64_t *in1 = new uint64_t[kk*(m+1)];
//           uint64_t *in2 = new uint64_t[kk];
//           uint64_t *in3 = new uint64_t[kk];
//           uint64_t *sleq1 = new uint64_t[kk*(m+1)];
//           uint64_t *sleq2 = new uint64_t[kk];
//           for (int j = lendt; j < lenut; j++) {
//           // for (int j = lendt; j <= lenut; j++) {
//             in0[j-lendt] = Smin;
//             in1[j-lendt] = rS[j];
//             for (int k = 0; k < m; k++) {
//               in0[(k+1)*kk+j-lendt] = St[j][k];
//               in1[(k+1)*kk+j-lendt] = STmin[k];
//             }
//           }
//           // memcpy(in1, rS+lendt, kk * sizeof(uint64_t));
//           comparison_with_eq2N(kk*(m+1), in0, in1, sleq1, Auxt[itr]);
//           memcpy(in2, sleq1+kk, kk * sizeof(uint64_t));
//           memcpy(in3, sleq1+kk*2, kk * sizeof(uint64_t));
//           Prod_H(kk, in2, in3, sleq2, Prodt[itr]);
//           memcpy(Sig+lendt, sleq1, kk * sizeof(uint64_t));
//           memcpy(Dlt+lendt, sleq2, kk * sizeof(uint64_t));
//           // for (int j = lendt; j <= lenut; j++) {
//           //   Sig[j] = sleq1[j-lendt];
//           //   Dlt[j] = sleq2[j-lendt];
//           // }
//           delete[] in0;
//           delete[] in1;
//           delete[] in2;
//           delete[] in3;
//           delete[] sleq1;
//           delete[] sleq2;
//         }
//       }
//       #pragma omp taskwait
//     }  
//   } 
//   uint64_t *in1 = new uint64_t[2];
//   uint64_t *in2 = new uint64_t[2];
//   uint64_t *in3 = new uint64_t[2];
//   for (int j = 0; j < dimt; j++) {
//     in1[0] = Sig[j] & mask;
//     in1[1] = Dlt[j] & mask;
//     in2[0] = (SS_one - Fla) & mask;
//     in2[1] = (SS_one - Sig[j]) & mask;
//     Prod_H(2, in1, in2, in3, Prodt[0]);//first = in3[0], Dom = in3[1];
//     Fla = (Fla + in3[0]) & mask;
//     Phi[j] = (in3[0] + in3[1]) & mask;
//     T[j] = (Tmax - rS[j]) & mask;
//   }
//   // uint64_t *in0 = new uint64_t[dimt];
//   // uint64_t *in1 = new uint64_t[dimt];
//   // for (int j = 0; j < dimt; j++) {
//   //   in0[j] = (SS_one - Sig[j]) & mask;
//   //   T[j] = (Tmax - rS[j]) & mask;
//   // }
//   // Prod_H(dimt, Dlt, in0, in1, Prodt[0]);//Dom = in1;
//   // uint64_t *in2 = new uint64_t[1];
//   // uint64_t *in3 = new uint64_t[1];
//   // uint64_t *in4 = new uint64_t[1];
//   // for (int j = 0; j < dimt; j++) {
//   //   in2[0] = Sig[j] & mask;
//   //   in3[0] = (SS_one - Fla) & mask;
//   //   Prod_H(1, in2, in3, in4, Prodt[0]);//first = in4[0];
//   //   Fla = (Fla + in4[0]) & mask;
//   //   Phi[j] = (in4[0] + in1[j]) & mask;
//   // }
//   // delete[] in0;
//   // delete[] in4;
//   //Print Phi
//   // if (party == ALICE)
//   // {
//   //   Iot[0]->send_data(Phi, dimt * sizeof(uint64_t));
//   //   Iot[1]->send_data(Sig, dimt * sizeof(uint64_t));
//   //   Iot[2]->send_data(Dlt, dimt * sizeof(uint64_t));
//   //   // Iot[0]->send_data(&Tmax, sizeof(uint64_t));
//   // }
//   // else
//   // {
//   //   uint64_t *PhiT =  new uint64_t[dimt]; 
//   //   uint64_t *SigT =  new uint64_t[dimt]; 
//   //   uint64_t *DltT =  new uint64_t[dimt]; 
//   //   // uint64_t TmaxT =  0; 
//   //   Iot[0]->recv_data(PhiT, dimt * sizeof(uint64_t));
//   //   Iot[1]->recv_data(SigT, dimt * sizeof(uint64_t));
//   //   Iot[2]->recv_data(DltT, dimt * sizeof(uint64_t));
//   //   for (int i = 0; i < dimt; i++) {
//   //     cout<< ((PhiT[i] + Phi[i]) & mask) << ",";
//   //   }
//   //   cout<<endl;
//   //   for (int i = 0; i < dimt; i++) {
//   //     cout<< ((SigT[i] + Sig[i]) & mask) << ",";
//   //   }
//   //   cout<<endl;
//   //   for (int i = 0; i < dimt; i++) {
//   //     cout<< ((DltT[i] + Dlt[i]) & mask) << ",";
//   //   }
//   //   cout<<endl;
//   //   delete[] PhiT;
//   //   delete[] SigT;
//   //   delete[] DltT;
//   //   // Iot[0]->recv_data(&TmaxT, sizeof(uint64_t));
//   //   // cout<< ((Tmax + TmaxT) & mask) << endl;
//   // }
//   Prod_H(dimt, T, Phi, res, Prodt[0]);
//   for (int j = 0; j < dimt; j++) {
//     rS[j] = (rS[j] + res[j]) & mask;
//   }
//   delete[] in1;
//   delete[] in2;
//   delete[] in3;
//   delete[] Sig;
//   delete[] Dlt;
//   delete[] Phi;
//   delete[] T;
//   delete[] res;
// }

void SDOMbyMin_T(int dimt, uint64_t Tmax, uint64_t *STmin, uint64_t **St, uint64_t Smin, uint64_t * &rS) {
  uint64_t *Sig = new uint64_t[dimt];
  uint64_t *Dlt = new uint64_t[dimt];
  uint64_t *Phi = new uint64_t[dimt];
  uint64_t *T = new uint64_t[dimt];
  uint64_t *res = new uint64_t[dimt];
  uint64_t SS_one = 0;
  // uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    // Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    // SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    // prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    // Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;    
  }
  // uint64_t Fla = SS_zero;
  uint64_t Fla = 0;
  //Print S
  // if (party == ALICE)
  // {
  //   Iot[2]->send_data(rS, dimt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[dimt]; 
  //   Iot[2]->recv_data(STTT, dimt * sizeof(uint64_t));
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((STTT[i] + rS[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(STmin, m * sizeof(uint64_t));
  //   Iot[1]->send_data(&Smin, sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STminT =  new uint64_t[m];    
  //   Iot[0]->recv_data(STminT, m * sizeof(uint64_t));
  //   for (int i = 0; i < m; i++) {
  //     cout<< ((STminT[i] + STmin[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STminT;
  //   uint64_t SminT =  0; 
  //   Iot[1]->recv_data(&SminT, sizeof(uint64_t));
  //   cout<< ((Smin + SminT) & mask) << endl;
  // }  
  int thr = (dimt > THs) ? THs : dimt;
  #pragma omp parallel num_threads(thr)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < thr; itr++)
      {
        #pragma omp task firstprivate(itr, thr)
        {
          int lendt = (((dimt) * itr)/ thr);
          int lenut = ((dimt) * (itr + 1))/ thr;
          // int lendt = (((dimt - 1) * itr)/ thr) + 1;
          // int lenut = ((dimt - 1) * (itr + 1))/ thr;
          // if(itr == 0){
          //   lendt = 0;
          // }
          // int kk = lenut-lendt+1;
          int kk = lenut-lendt;
          // cout<<itr<<" "<<kk<<endl;
          uint64_t *in0 = new uint64_t[kk*(m+1)];
          uint64_t *in1 = new uint64_t[kk*(m+1)];
          uint8_t *in2 = new uint8_t[kk];
          uint8_t *in3 = new uint8_t[kk];
          uint8_t *sleq1 = new uint8_t[kk*(m+1)];
          uint8_t *sleq2 = new uint8_t[kk];
          uint64_t *sleq3 = new uint64_t[kk];
          for (int j = lendt; j < lenut; j++) {
          // for (int j = lendt; j <= lenut; j++) {
            in0[j-lendt] = Smin;
            in1[j-lendt] = rS[j];
            for (int k = 0; k < m; k++) {
              in0[(k+1)*kk+j-lendt] = St[j][k];
              in1[(k+1)*kk+j-lendt] = STmin[k];
            }
          }
          // memcpy(in1, rS+lendt, kk * sizeof(uint64_t));
          comparison_with_eq2N(kk*(m+1), in0, in1, sleq1, Auxt[itr]);
          memcpy(in2, sleq1+kk, kk * sizeof(uint8_t));
          memcpy(in3, sleq1+kk*2, kk * sizeof(uint8_t));
          Auxt[itr]->AND(in2, in3, sleq2, kk);
          Auxt[itr]->B2A(sleq2, sleq3, kk, lambda);
          memcpy(Dlt+lendt, sleq3, kk * sizeof(uint64_t));
          memcpy(sleq2, sleq1, kk * sizeof(uint8_t));
          Auxt[itr]->B2A(sleq2, sleq3, kk, lambda);
          memcpy(Sig+lendt, sleq3, kk * sizeof(uint64_t));
          // for (int j = lendt; j <= lenut; j++) {
          //   Sig[j] = sleq1[j-lendt];
          //   Dlt[j] = sleq2[j-lendt];
          // }
          delete[] in0;
          delete[] in1;
          delete[] in2;
          delete[] in3;
          delete[] sleq1;
          delete[] sleq2;
          delete[] sleq3;
        }
      }
      #pragma omp taskwait
    }  
  }  
  uint64_t *in1 = new uint64_t[2];
  uint64_t *in2 = new uint64_t[2];
  uint64_t *in3 = new uint64_t[2];
  for (int j = 0; j < dimt; j++) {
    in1[0] = Sig[j] & mask;
    in1[1] = Dlt[j] & mask;
    in2[0] = (SS_one - Fla) & mask;
    in2[1] = (SS_one - Sig[j]) & mask;
    Prod_H(2, in1, in2, in3, Prodt[0]);//first = in3[0], Dom = in3[1];
    Fla = (Fla + in3[0]) & mask;
    Phi[j] = (in3[0] + in3[1]) & mask;
    T[j] = (Tmax - rS[j]) & mask;
  }
  // uint64_t *in0 = new uint64_t[dimt];
  // uint64_t *in1 = new uint64_t[dimt];
  // for (int j = 0; j < dimt; j++) {
  //   in0[j] = (SS_one - Sig[j]) & mask;
  //   T[j] = (Tmax - rS[j]) & mask;
  // }
  // Prod_H(dimt, Dlt, in0, in1, Prodt[0]);//Dom = in1;
  // uint64_t *in2 = new uint64_t[1];
  // uint64_t *in3 = new uint64_t[1];
  // uint64_t *in4 = new uint64_t[1];
  // for (int j = 0; j < dimt; j++) {
  //   in2[0] = Sig[j] & mask;
  //   in3[0] = (SS_one - Fla) & mask;
  //   Prod_H(1, in2, in3, in4, Prodt[0]);//first = in4[0];
  //   Fla = (Fla + in4[0]) & mask;
  //   Phi[j] = (in4[0] + in1[j]) & mask;
  // }
  // delete[] in0;
  // delete[] in4;
  //Print Phi
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(Phi, dimt * sizeof(uint64_t));
  //   Iot[1]->send_data(Sig, dimt * sizeof(uint64_t));
  //   Iot[2]->send_data(Dlt, dimt * sizeof(uint64_t));
  //   // Iot[0]->send_data(&Tmax, sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *PhiT =  new uint64_t[dimt]; 
  //   uint64_t *SigT =  new uint64_t[dimt]; 
  //   uint64_t *DltT =  new uint64_t[dimt]; 
  //   // uint64_t TmaxT =  0; 
  //   Iot[0]->recv_data(PhiT, dimt * sizeof(uint64_t));
  //   Iot[1]->recv_data(SigT, dimt * sizeof(uint64_t));
  //   Iot[2]->recv_data(DltT, dimt * sizeof(uint64_t));
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((PhiT[i] + Phi[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((SigT[i] + Sig[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   for (int i = 0; i < dimt; i++) {
  //     cout<< ((DltT[i] + Dlt[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] PhiT;
  //   delete[] SigT;
  //   delete[] DltT;
  //   // Iot[0]->recv_data(&TmaxT, sizeof(uint64_t));
  //   // cout<< ((Tmax + TmaxT) & mask) << endl;
  // }
  Prod_H(dimt, T, Phi, res, Prodt[0]);
  for (int j = 0; j < dimt; j++) {
    rS[j] = (rS[j] + res[j]) & mask;
  }
  delete[] in1;
  delete[] in2;
  delete[] in3;
  delete[] Sig;
  delete[] Dlt;
  delete[] Phi;
  delete[] T;
  delete[] res;
}

uint64_t *SkylineResbyQ0(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 3;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  uint64_t SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  // uint64_t *tmp0 = new uint64_t[len];
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   for (int k1 = 0; k1 < len1; k1++)
  //   {
  //     int slen = SS_L[(k1 * len2) + k2];
  //     if (len == slen)
  //       continue;
  //     memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
  //     for (int k = slen; k < len; k++)
  //     {
  //       // if (party == ALICE)
  //       // {
  //       //   prg.random_data(&SS_zero, sizeof(uint64_t));
  //       //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
  //       //   tmp0[k] = SS_zero;
  //       // }
  //       // else
  //       // {
  //       //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
  //       //   tmp0[k] = (0 - SS_zero) & mask;
  //       // }
  //       tmp0[k] = SS_zero;
  //     }
  //     delete[] SS_Skyline[(k1 * len2) + k2];
  //     SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
  //     memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
  //   }
  // }
  // delete[] tmp0;
  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel num_threads(16)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // construct the same skyline
              // int slen = SS_L[(k1<<MAX)+k2];
              // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
              // uint64_t offset = 0;
              // Poly(polylen, a[k1], k2, offset);
              uint64_t offset = a[k1 * polylen];
              uint64_t t = k2;
              for (int i = 1; i < len; i++)
              {
                offset = (offset + a[k1 * polylen + i] * t) & mask;
                t = (t * k2) & mask;
              } 
              int lent = SS_L[(k1 * len2) + k2];
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
              }
            }
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[lent];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, lent * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], lent * sizeof(uint64_t));
              }
              for (int k = 0; k < lent; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[len1 * polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[len1 * polylen];
  uint64_t *outD = new uint64_t[polylen];

  uint64_t *qt = new uint64_t[m];
  // qt[0] = SS_G[1][k1 - 1];
  // qt[1] = SS_G[2][k2 - 1];
  for (int j = 0; j < m; j++) {
    qt[j] =  Q[j] & mask; 
  }
  int lt = len;
  uint64_t *S = new uint64_t[lt];
  uint64_t **St = new uint64_t*[lt];
  uint64_t *in1 = new uint64_t[lt*m];
  uint64_t *out1 = new uint64_t[lt*m];
  uint64_t **pt = new uint64_t*[lt];
  uint64_t *Bt = new uint64_t[lt];
  uint64_t *Rt = new uint64_t[lt*m];
  unordered_map<uint32_t, uint64_t*> Result;
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  double endscmp = omp_get_wtime();
  cout << "select element cmp:" << (endscmp-starts) << endl;
  // JT[0] = SS_one;
  // JT[1] = pos[1];
  for (int j = 0; j < len1; j++)
  {
    JT[j * polylen + 0] = SS_one;
    JT[j * polylen + 1] = pos[1];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    // JT[i] = outC[0];
    IT[i] = outC[1];
    for (int j = 0; j < len1; j++)
    {
      JT[j * polylen + i] = outC[0];
    } 
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  double startpoly = omp_get_wtime();
  Prod_H(len1 * polylen, a, JT, outD0, Prodt[0]);
  double endspoly = omp_get_wtime();
  cout << "select element poly:" << (endspoly-startpoly) << endl;
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      for (int k = 0; k < SS_L[(k1 * len2) + k2]; k++)
      {
        tmp2[k1 * len + k] = (tmp2[k1 * len + k] + SS_Skyline[(k1 * len2) + k2][k] * posindex[1][k2]) & mask;
      }
    }
    //polynomial
    uint64_t mk1 = 0;
    for (int k = 0; k < polylen; k++)
    {
      mk1 = (mk1 + outD0[k1 * polylen + k]) & mask;
    }
    // uint64_t offset = b[0] + b[1] * k1;  // a2*i+b2
    // uint64_t offset = 0;
    // Poly(polylen, b, k1, offset);
    uint64_t offset = (b[0] - mk1) & mask;
    uint64_t t = k1;
    for (int i = 1; i < polylen; i++)
    {
      offset = (offset + b[i] * t) & mask;
      t = (t * k1) & mask;
    } 
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
  }
  double endsaggr = omp_get_wtime();
  cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[1]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[1]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  for (int k = 0; k < polylen; k++)
  {
      mk2 = (mk2 + outD[k]) & mask;
  }
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  double endqua = omp_get_wtime();
  cout << "select element quasky:" << (endqua - starts) << endl;
  // select dynamic skyline == res->result
  int th = 0;
    // dynamic points
  // DP(len, res, qt, lenD, resD);
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t));   
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;
  // }
  // else
  // {
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t)); 
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;   
  // }
  double xx1 = omp_get_wtime();
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  //separate each dimension from "<<MAX"
  if (party == ALICE)
  {
    uint64_t masktt = (1ULL << (MAX-2)) - 1;
    for (int j = 0; j < lt*m; j++)
    {
      Rt[j] = Rt[j] & masktt;
    }
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (res[i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
    }
    Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
    uint64_t *Rt0 = new uint64_t[lt*m];
    Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      } 
      // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
      // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
    }
    delete[] Rt0;
  }
  else
  {
    Iot[th]->recv_data(Bt, lt * sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (Bt[i] + res[i]) & mask;
    }
    uint64_t *s = new uint64_t[m];
    uint64_t masktt = (1ULL << MAX) - 1;
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      s[0] = Bt[i] & masktt;
      for (int j = 1; j < m; j++)
      {
        Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
        s[j] = Bt[i] & masktt;
      }
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
        // x[j*lt+i] = pt[i][j] & mask;
      } 
    }
    delete[] s;
    Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));   
  }
  //Euclid product
  for (int i = 0; i < lt; i++) {
    for (int j = 0; j < m; j++) {
      in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
    }
  }
  Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  double xx2 = omp_get_wtime();
  cout << "dy element euclid:" << (xx2 - xx1) << endl;
  for (int i = 0; i < lt; i++) {
    S[i] = 0;
    St[i] = new uint64_t[m];
    for (int j = 0; j < m; j++) {
      St[i][j] = out1[i*m+j] & mask;
      S[i] = (S[i] + out1[i*m+j]) & mask;  
    }
  }
  uint64_t TMAX = 1ULL << (2 * MAX + 1);
  if (party == ALICE)
  {
    uint64_t T0 = 0;
    prg.random_data(&T0, sizeof(uint64_t));
    TMAX = (TMAX - T0) & mask;
    Iot[th]->send_data(&T0, sizeof(uint64_t));
  }
  else
  {
    Iot[th]->recv_data(&TMAX, sizeof(uint64_t)); 
  }      
  uint64_t lam = 1;  
  double xx3 = omp_get_wtime();
  cout << "dy element St:" << (xx3 - xx2) << endl; 
  uint64_t *r = new uint64_t[1];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *out2 = new uint64_t[1];
  while(lam!=0){
    uint64_t STMin = 0;
    double tt1 = omp_get_wtime();
    SMIN(lt, S, th, STMin);
    double tt2 = omp_get_wtime();
    cout << "loop element Smin:" << (tt2 - tt1) << endl; 
    prg.random_data(r, sizeof(uint64_t));
    in2[0] = (STMin - TMAX) & mask;
    Prod_H(1, in2, r, out2, Prodt[th]);
    if (party == ALICE)
    {
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      Iot[th]->send_data(out2, sizeof(uint64_t));
      lam = (lam + out2[0]) & mask;
    }
    else
    {
      Iot[th]->send_data(out2, sizeof(uint64_t));
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      lam = (lam + out2[0]) & mask;
    }
    cout<<"lam:"<<lam<<endl;
    double tt3 = omp_get_wtime();
    cout << "loop element break:" << (tt3 - tt2) << endl;
    if(lam!=0){
      uint64_t *A = new uint64_t[lt];
      for (int i = 0; i < lt; i++) {
          A[i] = (S[i] - STMin) & mask;
      }
      uint64_t *r3 = new uint64_t[lt];
      prg.random_data(r3, lt * sizeof(uint64_t));
      uint64_t *out3 = new uint64_t[lt];
      Prod_H(lt, A, r3, out3, Prodt[th]);
      uint32_t *Pi = new uint32_t[lt];
      uint32_t *Pi_Inv = new uint32_t[lt];
      for (int i = 0; i < lt; i++) {
        Pi[i] = i;
      }
      for (int i = 0; i < lt; i++) {
        uint32_t randomPosition = 0;
        prg.random_data(&randomPosition, sizeof(uint32_t));
        randomPosition = randomPosition & (lt-1);
        uint32_t temp = Pi[i];
        Pi[i] = Pi[randomPosition];
        Pi[randomPosition] = temp;
      }
      for (int i = 0; i < lt; i++) {
        Pi_Inv[Pi[i]] = i;
      }
      if (party == ALICE)
      {
        Iot[th]->send_data(Pi, lt * sizeof(uint32_t));
        Iot[th]->send_data(Pi_Inv, lt * sizeof(uint32_t));
      }
      else
      {
        Iot[th]->recv_data(Pi, lt * sizeof(uint32_t));
        Iot[th]->recv_data(Pi_Inv, lt * sizeof(uint32_t));
      }
      uint64_t *B = new uint64_t[lt];
      for (int i = 0; i < lt; i++) {
        B[Pi[i]] = out3[i];
      }       
      // B
      uint64_t *b = new uint64_t[lt];
      uint64_t *U = new uint64_t[lt];
      if (party == ALICE)
      {
        Iot[th]->recv_data(b, lt * sizeof(uint64_t));
        Iot[th]->send_data(B, lt * sizeof(uint64_t));
        for (int i = 0; i < lt; i++) {
          b[i] = (b[i] + B[i]) & mask;
          U[i] = SS_zero;
          if (b[i]==0){
              U[i] = SS_one;
          }
        }
      }
      else
      {
        Iot[th]->send_data(B, lt * sizeof(uint64_t));
        Iot[th]->recv_data(b, lt * sizeof(uint64_t));
        for (int i = 0; i < lt; i++) {
          b[i] = (b[i] + B[i]) & mask;
          U[i] = SS_zero;
          if (b[i]==0){
              U[i] = SS_one;
          }
        }
      }
      //select the first one
      uint64_t phi2 = SS_zero;
      uint64_t *U1 = new uint64_t[1];
      uint64_t *U2 = new uint64_t[1];
      uint64_t *U3 = new uint64_t[1];
      for (int i = 0; i < lt; i++) {
        phi2 = (U[i] + phi2) & mask;
        // 1<=phi2
        if (party == ALICE)
        {
          U1[0] = (phi2 - 1) & mask;
        }
        else
        {
          U1[0] = (0 - phi2) & mask;
        }
        comparison(1, U1, U2, Auxt[th]);
        U1[0] = U[i];
        U2[0] = (SS_one - U2[0]) & mask;
        Prod_H(1, U1, U2, U3, Prodt[th]);
        U[i] = U3[0];
      }
      
      // A
      uint64_t *V = new uint64_t[lt];
      for(int i = 0; i < lt; i++) {
          V[Pi_Inv[i]] = U[i];
      }
      uint64_t *Pmin = new uint64_t[m];
      uint64_t *Tmin = new uint64_t[m];
      uint64_t *in4 = new uint64_t[1];
      uint64_t *in5 = new uint64_t[1];
      uint64_t *out4 = new uint64_t[1];         
      for (int j = 0; j < m; j++) {
        Pmin[j] = SS_zero;
        Tmin[j] = SS_zero;
        for (int i = 0; i < lt; i++) {
          in4[0] = V[i];
          in5[0] = pt[i][j];
          Prod_H(1, in4, in5, out4, Prodt[th]); 
          Pmin[j] = (Pmin[j] + out4[0]) & mask;
          in5[0] = St[i][j];
          Prod_H(1, in4, in5, out4, Prodt[th]);
          Tmin[j] = (Tmin[j] + out4[0]) & mask;
        }
      }
      // A
      int pos = Result.size();
      // cout<<pos<<endl;
      Result[pos] = Pmin;
      //eliminate
      for (int i = 0; i < lt; i++) {
        in4[0] = V[i];
        in5[0] = (TMAX - S[i]) & mask;
        Prod_H(1, in4, in5, out4, Prodt[th]);
        S[i] = (out4[0] + S[i]) & mask;
      }
      for(int i = 0; i < lt; i++) {
          SDOMbyMin(m, Tmin, St[i], th, V[i]);
      }
      for (int i = 0; i < lt; i++) {
        in4[0] = V[i];
        in5[0] = (TMAX - S[i]) & mask;
        Prod_H(1, in4, in5, out4, Prodt[th]);
        S[i] = (out4[0] + S[i]) & mask;
      }
      delete[] A;
      delete[] B;
      delete[] b;
      delete[] r3;
      delete[] out3;
      delete[] in4;
      delete[] in5;
      delete[] out4;
      delete[] U;
      delete[] U1;
      delete[] U2;
      delete[] U3;
      delete[] V;
      // delete[] Pmin;
      delete[] Tmin;
      delete[] Pi;
      delete[] Pi_Inv;
    }
    double tt4 = omp_get_wtime();
    cout << "loop element filter:" << (tt4 - tt3) << endl;
  }
  double xx4 = omp_get_wtime();
  cout << "dy element loop:" << (xx4 - xx3) << endl;
  // Return
  // A
  int lenD = Result.size();
  // cout<<k<<endl;
  uint64_t *resD = new uint64_t[lenD*m];
  for (int i = 0; i < lenD; i++)
  {
    for (int j = 0; j < m; j++)
    {
      resD[i*m+j] = Result[i][j] & mask;
    }
  }
  reslen = lenD;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  delete[] r;
  delete[] in2;
  delete[] out2;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  delete[] Bt;
  delete[] Rt;
  for(int j = 0; j< lt; j++)
  {
    delete[] St[j];
    delete[] pt[j];
  }
  delete[] St;
  delete[] pt;
  delete[] qt;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  return resD;
}

uint64_t *SkylineResbyQ_old(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 10;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  for (int k1 = 0; k1 <= len1 - 1; k1++)
  {
    for (int k2 = len2 - 1; k2 >= 0; k2--)
    {
      SS_L[k1 * len2 + k2] = SS_L_itr[k1 * len2t + k2];
      SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_L_itr[k1 * len2t + k2]];
      memcpy(SS_Skyline[k1 * len2 + k2], SS_Skyline_itr[k1 * len2t + k2], SS_L_itr[k1 * len2t + k2] * sizeof(uint64_t));
    }
  }
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  uint64_t TMAX = (10000 << MAX) + 10000;
  // uint64_t STMAX = 1ULL << (2 * MAX + 1);
  uint64_t STMAX = 2 * 10000 * 10000;
  if (party == ALICE)
  {
    uint64_t T0 = 0;
    prg.random_data(&T0, sizeof(uint64_t));
    TMAX = (TMAX - T0) & mask;
    Iot[0]->send_data(&T0, sizeof(uint64_t));
    prg.random_data(&T0, sizeof(uint64_t));
    STMAX = (STMAX - T0) & mask;
    Iot[1]->send_data(&T0, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&TMAX, sizeof(uint64_t));
    Iot[1]->recv_data(&STMAX, sizeof(uint64_t)); 
  }    

  // int skylineLen = SS_L[(4000 * SS_G[0][1]) + 2366];
  // uint64_t *ssss = new uint64_t[skylineLen];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + skylineLen, ssss);
  // SS_Print(skylineLen, ssss);

  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = TMAX;
        // tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }
  
  // uint64_t *ss = new uint64_t[len];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + len, ss);
  // SS_Print(len, ss);

  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset = (offset + a[k1 * polylen + i] * t) & mask;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[len1 * polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[len1 * polylen];
  uint64_t *outD = new uint64_t[polylen];

  uint64_t *qt = new uint64_t[m];
  // qt[0] = SS_G[1][k1 - 1];
  // qt[1] = SS_G[2][k2 - 1];
  for (int j = 0; j < m; j++) {
    qt[j] =  Q[j] & mask; 
  }
  int lt = len;
  uint64_t *S = new uint64_t[lt];
  uint64_t **St = new uint64_t*[lt];
  uint64_t *in1 = new uint64_t[lt*m];
  uint64_t *out1 = new uint64_t[lt*m];
  uint64_t **pt = new uint64_t*[lt];
  uint64_t *Bt = new uint64_t[lt];
  uint64_t *Rt = new uint64_t[lt*m];
  uint64_t *Pmin = new uint64_t[m];
  uint64_t *Tmin = new uint64_t[m];
  unordered_map<uint32_t, uint64_t *> Result;
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // double endscmp = omp_get_wtime();
  // cout << "select element cmp:" << (endscmp-starts) << endl;
  // JT[0] = SS_one;
  // JT[1] = pos[1];
  double startpoly = omp_get_wtime();
  for (int j = 0; j < len1; j++)
  {
    JT[j * polylen + 0] = SS_one;
    JT[j * polylen + 1] = pos[1];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    // JT[i] = outC[0];
    IT[i] = outC[1];
    for (int j = 0; j < len1; j++)
    {
      JT[j * polylen + i] = outC[0];
    } 
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  Prod_H(len1 * polylen, a, JT, outD0, Prodt[0]);
  double endspoly = omp_get_wtime();
  cout << "select element poly:" << (endspoly-startpoly) << endl;
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
    //polynomial
    // uint64_t mk1 = 0;
    // for (int k = 0; k < polylen; k++)
    // {
    //   mk1 = (mk1 + outD0[k1 * polylen + k]) & mask;
    // }
    // uint64_t offset = (b[0] - mk1) & mask;
    // uint64_t t = k1;
    // for (int i = 1; i < polylen; i++)
    // {
    //   offset = (offset + b[i] * t) & mask;
    //   t = (t * k1) & mask;
    // } 
    uint64_t offset = 0;
    uint64_t t = 1;
    for (int k = 0; k < polylen; k++)
    {
      offset += b[k] * t - outD0[k1 * polylen + k];
      t = (t * k1) & mask;
    } 
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
  }
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[1]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[1]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  for (int k = 0; k < polylen; k++)
  {
    // mk2 = (mk2 + outD[k]) & mask;
      mk2 += outD[k];
  }
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  double endqua = omp_get_wtime();
  // SS_Print(len, res);
  cout << "select element quasky:" << (endqua - starts) << endl;
  // select dynamic skyline == res->result
  int th = 0;
  // dynamic points
  // DP(len, res, qt, lenD, resD);
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t));   
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;
  // }
  // else
  // {
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t)); 
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;   
  // }
  // double xx1 = omp_get_wtime();
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  //separate each dimension from "<<MAX"
  if (party == ALICE)
  {
    uint64_t masktt = (1ULL << (MAX-2)) - 1;
    for (int j = 0; j < lt*m; j++)
    {
      Rt[j] = Rt[j] & masktt;
    }
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (res[i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
    }
    Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
    uint64_t *Rt0 = new uint64_t[lt*m];
    Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      } 
      // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
      // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
    }
    delete[] Rt0;
  }
  else
  {
    Iot[th]->recv_data(Bt, lt * sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (Bt[i] + res[i]) & mask;
    }
    uint64_t *s = new uint64_t[m];
    uint64_t masktt = (1ULL << MAX) - 1;
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      s[0] = Bt[i] & masktt;
      for (int j = 1; j < m; j++)
      {
        Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
        s[j] = Bt[i] & masktt;
      }
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
        // x[j*lt+i] = pt[i][j] & mask;
      } 
    }
    delete[] s;
    Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));   
  }
  //Euclid product
  for (int i = 0; i < lt; i++) {
    for (int j = 0; j < m; j++) {
      in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
    }
  }
  Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  // double xx2 = omp_get_wtime();
  // cout << "dy element euclid:" << (xx2 - xx1) << endl;
  for (int i = 0; i < lt; i++) {
    S[i] = 0;
    St[i] = new uint64_t[m];
    for (int j = 0; j < m; j++) {
      St[i][j] = out1[i*m+j] & mask;
      S[i] = (S[i] + out1[i*m+j]) & mask;  
    }
  }
   //  Print S
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[lt]; 
  //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
  //   for (int i = 0; i < lt; i++) {
  //     cout<< ((STTT[i] + S[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }    

  uint64_t lam = 1;  
  double xx3 = omp_get_wtime();
  // cout << "dy element St:" << (xx3 - xx2) << endl; 
  uint64_t *r = new uint64_t[1];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *out2 = new uint64_t[1];
  while(lam!=0){
    uint64_t STMin = 0;
    // double tt1 = omp_get_wtime();
    SMIN(0, lt, S, pt, St, STMin, Pmin, Tmin);
    // double tt2 = omp_get_wtime();
    // cout << "loop element Smin:" << (tt2 - tt1) << endl; 
    prg.random_data(r, sizeof(uint64_t));
    in2[0] = (STMAX - STMin) & mask;
    Prod_H(1, in2, r, out2, Prodt[th]);
    if (party == ALICE)
    {
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      Iot[th+1]->send_data(out2, sizeof(uint64_t));
      lam = (lam + out2[0]) & mask;
    }
    else
    {
      Iot[th]->send_data(out2, sizeof(uint64_t));
      Iot[th+1]->recv_data(&lam, sizeof(uint64_t)); 
      lam = (lam + out2[0]) & mask;
    }
    // cout<<"lam:"<<lam<<endl;
    // if (party == ALICE)
    // {
    //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
    //   Iot[th+1]->send_data(&STMin, sizeof(uint64_t));
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    // else
    // {
    //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
    //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    
    // double tt3 = omp_get_wtime();
    // cout << "loop element break:" << (tt3 - tt2) << endl;
    if(lam!=0){
      int pos = Result.size();
      // cout<<pos<<endl;
      Result[pos] = new uint64_t[m];
      // for (int i = 0; i < m; i++)
      // {
      //   Result[pos][i] = Pmin[i];
      // }
      memcpy(Result[pos], Pmin, m * sizeof(uint64_t));
      // Result[pos] = Pmin;
      // cout<<Pmin[0]<<","<<Result[pos][0]<<endl;
      // if (party == ALICE)
      // {
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[0], sizeof(uint64_t));
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[1], sizeof(uint64_t));
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      // else
      // {
      //   Iot[th]->send_data(&Pmin[0], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->send_data(&Pmin[1], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      //eliminate
      SDOMbyMin(lt, STMAX, Tmin, St, STMin, S);
      //Print S
      // if (party == ALICE)
      // {
      //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
      // }
      // else
      // {
      //   uint64_t *STTT =  new uint64_t[lt]; 
      //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
      //   for (int i = 0; i < lt; i++) {
      //     cout<< ((STTT[i] + S[i]) & mask) << ",";
      //   }
      //   cout<<endl;
      //   delete[] STTT;
      // }  
    }
    // double tt4 = omp_get_wtime();
    // cout << "loop element filter:" << (tt4 - tt3) << endl;
  }
  double xx4 = omp_get_wtime();
  cout << "dy element loop:" << (xx4 - xx3) << endl;
  // Return
  // A
  int lenD = Result.size();
  uint64_t *resD = new uint64_t[lenD*m];
  for (int i = 0; i < lenD; i++)
  {
    for (int j = 0; j < m; j++)
    {
      resD[i*m+j] = Result[i][j];
    }
    // cout<<i<<":"<<Result[i][0]<<","<<resD[i*m+0]<<endl;
  }
  reslen = lenD;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  cout<<lenD<<endl;
  if (party == ALICE)
  {
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  else
  {
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  delete[] r;
  delete[] in2;
  delete[] out2;
  delete[] Tmin;
  delete[] Pmin;
  delete[] Bt;
  delete[] Rt;
  for(int j = 0; j< lt; j++)
  {
    delete[] St[j];
    delete[] pt[j];
  }
  delete[] St;
  delete[] pt;
  delete[] in1;
  delete[] out1;
  delete[] qt;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  for (int k = 0; k < lenD; k++)
  {
    delete[] Result[k];
  }
  Result.clear();
  return resD;
}

uint64_t *SkylineResbyQ(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 2;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  for (int k1 = 0; k1 <= len1 - 1; k1++)
  {
    for (int k2 = len2 - 1; k2 >= 0; k2--)
    {
      SS_L[k1 * len2 + k2] = SS_L_itr[k1 * len2t + k2];
      SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_L_itr[k1 * len2t + k2]];
      memcpy(SS_Skyline[k1 * len2 + k2], SS_Skyline_itr[k1 * len2t + k2], SS_L_itr[k1 * len2t + k2] * sizeof(uint64_t));
    }
  }
  uint64_t *rp2 = new uint64_t[len1];
  prg.random_data(rp2, len1 * sizeof(uint64_t));
  // memset(rp2, 0, len1 * sizeof(uint64_t));
  uint64_t rp1 = 0;
  prg.random_data(&rp1, sizeof(uint64_t));
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  uint64_t TMAX = (10000 << MAX) + 10000;
  // uint64_t STMAX = 1ULL << (2 * MAX + 1);
  uint64_t STMAX = 2 * 10000 * 10000;
  if (party == ALICE)
  {
    uint64_t T0 = 0;
    prg.random_data(&T0, sizeof(uint64_t));
    TMAX = (TMAX - T0) & mask;
    Iot[0]->send_data(&T0, sizeof(uint64_t));
    prg.random_data(&T0, sizeof(uint64_t));
    STMAX = (STMAX - T0) & mask;
    Iot[1]->send_data(&T0, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&TMAX, sizeof(uint64_t));
    Iot[1]->recv_data(&STMAX, sizeof(uint64_t)); 
  }    

  // int skylineLen = SS_L[(4000 * SS_G[0][1]) + 2366];
  // uint64_t *ssss = new uint64_t[skylineLen];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + skylineLen, ssss);
  // SS_Print(skylineLen, ssss);

  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = TMAX;
        // tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }
  
  // uint64_t *ss = new uint64_t[len];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + len, ss);
  // SS_Print(len, ss);

  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    // uint64_t mac = rp2[k1];
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset += a[k1 * polylen + i] * t;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      offset = offset & mask;
      // offset = mac + HashP(offset);
      offset = rp2[k1] + HashP(offset);
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  // exchange the copies
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }
  
  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[1];
  uint64_t *outD = new uint64_t[1];

  uint64_t *qt = new uint64_t[m];
  // qt[0] = SS_G[1][k1 - 1];
  // qt[1] = SS_G[2][k2 - 1];
  for (int j = 0; j < m; j++) {
    qt[j] =  Q[j] & mask; 
  }
  int lt = len;
  uint64_t *S = new uint64_t[lt];
  uint64_t **St = new uint64_t*[lt];
  uint64_t *in1 = new uint64_t[lt*m];
  uint64_t *out1 = new uint64_t[lt*m];
  uint64_t **pt = new uint64_t*[lt];
  uint64_t *Bt = new uint64_t[lt];
  uint64_t *Rt = new uint64_t[lt*m];
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  uint64_t *Pmin = new uint64_t[m];
  uint64_t *Tmin = new uint64_t[m];
  unordered_map<uint32_t, uint64_t *> Result;
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // double endscmp = omp_get_wtime();
  // cout << "select element cmp:" << (endscmp-starts) << endl;
  // JT[0] = SS_one;
  // JT[1] = pos[1];
  // double startpoly = omp_get_wtime();
  IT[0] = SS_one;
  IT[1] = pos[0];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
    IT[i] = outC[1];
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  // double endspoly = omp_get_wtime();
  // cout << "select element poly:" << (endspoly-startpoly) << endl;
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
  }
  MultMode mode1 = MultMode::Alice_has_A;
  MultMode mode2 = MultMode::Bob_has_B;
  uint64_t *coe_a = new uint64_t[polylen];
  uint64_t *outDA = new uint64_t[1];
  uint64_t *outDB = new uint64_t[1];
  for (int k1 = 0; k1 < len1; k1++)
  {
    uint64_t offset = b[0];
    uint64_t t = k1;
    for (int k = 1; k < polylen; k++)
    {
      offset += b[k] * t;
      t = (t * k1) & mask;
    } 
    offset = offset & mask;
    offset = rp1 + HashP(offset);
    memcpy(coe_a, a+k1*polylen, polylen * sizeof(uint64_t));
    uint64_t tmp = 0;
    Prod_M(1, polylen, 1, coe_a, JT, outDA, Prodt[0], mode1);
    Prod_M(1, polylen, 1, JT, coe_a, outDB, Prodt[0], mode2);
    if (party == ALICE)
    { // send outDA
      Iot[0]->send_data(outDA, sizeof(uint64_t));
      Iot[0]->recv_data(&tmp, sizeof(uint64_t));
      tmp = (outDB[0] + tmp) & mask;
    }
    else
    { //send outDB
      Iot[0]->recv_data(&tmp, sizeof(uint64_t));
      Iot[0]->send_data(outDB, sizeof(uint64_t));
      tmp = (outDA[0] + tmp) & mask;
    }
    tmp = HashP(tmp);
    offset = (offset - rp2[k1] - tmp) & mask;
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
  }
  delete[] coe_a;
  delete[] outDA;
  delete[] outDB;
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  uint64_t tmp = 0;
  Prod_M(1, polylen, 1, b, IT, outD0, Prodt[0], mode1);
  Prod_M(1, polylen, 1, IT, b, outD, Prodt[0], mode2);
  if (party == ALICE)
  {
    Iot[0]->send_data(outD0, sizeof(uint64_t));
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    tmp = (outD[0] + tmp) & mask;
  }
  else
  {
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    Iot[0]->send_data(outD, sizeof(uint64_t));
    tmp = (outD0[0] + tmp) & mask;
  }
  tmp = HashP(tmp);
  mk2 = (rp1 + tmp) & mask;
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  double endqua = omp_get_wtime();
  // SS_Print(len, res);
  cout << "select element quasky:" << (endqua - starts) << endl;
  ss_selectSky = endqua - starts;
  // select dynamic skyline == res->result
  int th = 0;
  // dynamic points
  // DP(len, res, qt, lenD, resD);
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t));   
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;
  // }
  // else
  // {
  //   uint32_t *pos0 = new uint32_t[m];
  //   Iot[th]->recv_data(pos0, m * sizeof(uint32_t));
  //   for (int i = 0; i < m; i++)
  //   {
  //     pos0[i] = (pos0[i] + pos[i]) & mask;
  //   }
  //   Iot[th]->send_data(pos, m * sizeof(uint32_t)); 
  //   cout<<pos0[0]<<"\t"<<pos0[1]<<"\t"<<lt<<endl;  
  //   delete[] pos0;   
  // }
  // double xx1 = omp_get_wtime();
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  //separate each dimension from "<<MAX"
  if (party == ALICE)
  {
    uint64_t masktt = (1ULL << (MAX-2)) - 1;
    for (int j = 0; j < lt*m; j++)
    {
      Rt[j] = Rt[j] & masktt;
    }
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (res[i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
    }
    Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
    uint64_t *Rt0 = new uint64_t[lt*m];
    Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      } 
      // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
      // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
    }
    delete[] Rt0;
  }
  else
  {
    Iot[th]->recv_data(Bt, lt * sizeof(uint64_t));
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (Bt[i] + res[i]) & mask;
    }
    uint64_t *s = new uint64_t[m];
    uint64_t masktt = (1ULL << MAX) - 1;
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      s[0] = Bt[i] & masktt;
      for (int j = 1; j < m; j++)
      {
        Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
        s[j] = Bt[i] & masktt;
      }
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
        // x[j*lt+i] = pt[i][j] & mask;
      } 
    }
    delete[] s;
    Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));   
  }
  //Euclid product
  for (int i = 0; i < lt; i++) {
    for (int j = 0; j < m; j++) {
      in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
    }
  }
  Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  // double xx2 = omp_get_wtime();
  // cout << "dy element euclid:" << (xx2 - xx1) << endl;
  for (int i = 0; i < lt; i++) {
    S[i] = 0;
    St[i] = new uint64_t[m];
    for (int j = 0; j < m; j++) {
      St[i][j] = out1[i*m+j] & mask;
      S[i] = (S[i] + out1[i*m+j]) & mask;  
    }
  }
   //  Print S
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[lt]; 
  //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
  //   for (int i = 0; i < lt; i++) {
  //     cout<< ((STTT[i] + S[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }    

  uint64_t lam = 1;  
  double xx3 = omp_get_wtime();
  // cout << "dy element St:" << (xx3 - xx2) << endl; 
  uint64_t *r = new uint64_t[1];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *out2 = new uint64_t[1];
  while(lam!=0){
    uint64_t STMin = 0;
    // double tt1 = omp_get_wtime();
    SMIN(0, lt, S, pt, St, STMin, Pmin, Tmin);
    // double tt2 = omp_get_wtime();
    // cout << "loop element Smin:" << (tt2 - tt1) << endl; 
    prg.random_data(r, sizeof(uint64_t));
    in2[0] = (STMAX - STMin) & mask;
    Prod_H(1, in2, r, out2, Prodt[th]);
    if (party == ALICE)
    {
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      Iot[th+1]->send_data(out2, sizeof(uint64_t));
      lam = (lam + out2[0]) & mask;
    }
    else
    {
      Iot[th]->send_data(out2, sizeof(uint64_t));
      Iot[th+1]->recv_data(&lam, sizeof(uint64_t)); 
      lam = (lam + out2[0]) & mask;
    }
    // cout<<"lam:"<<lam<<endl;
    // if (party == ALICE)
    // {
    //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
    //   Iot[th+1]->send_data(&STMin, sizeof(uint64_t));
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    // else
    // {
    //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
    //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    
    // double tt3 = omp_get_wtime();
    // cout << "loop element break:" << (tt3 - tt2) << endl;
    if(lam!=0){
      int pos = Result.size();
      // cout<<pos<<endl;
      Result[pos] = new uint64_t[m];
      // for (int i = 0; i < m; i++)
      // {
      //   Result[pos][i] = Pmin[i];
      // }
      memcpy(Result[pos], Pmin, m * sizeof(uint64_t));
      // Result[pos] = Pmin;
      // cout<<Pmin[0]<<","<<Result[pos][0]<<endl;
      // if (party == ALICE)
      // {
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[0], sizeof(uint64_t));
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[1], sizeof(uint64_t));
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      // else
      // {
      //   Iot[th]->send_data(&Pmin[0], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->send_data(&Pmin[1], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      //eliminate
      SDOMbyMin(lt, STMAX, Tmin, St, STMin, S);
      //Print S
      // if (party == ALICE)
      // {
      //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
      // }
      // else
      // {
      //   uint64_t *STTT =  new uint64_t[lt]; 
      //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
      //   for (int i = 0; i < lt; i++) {
      //     cout<< ((STTT[i] + S[i]) & mask) << ",";
      //   }
      //   cout<<endl;
      //   delete[] STTT;
      // }  
    }
    // double tt4 = omp_get_wtime();
    // cout << "loop element filter:" << (tt4 - tt3) << endl;
  }
  double xx4 = omp_get_wtime();
  cout << "dy element loop:" << (xx4 - xx3) << endl;
  // Return
  // A
  int lenD = Result.size();
  uint64_t *resD = new uint64_t[lenD*m];
  for (int i = 0; i < lenD; i++)
  {
    for (int j = 0; j < m; j++)
    {
      resD[i*m+j] = Result[i][j];
    }
    // cout<<i<<":"<<Result[i][0]<<","<<resD[i*m+0]<<endl;
  }
  reslen = lenD;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  cout<<lenD<<endl;
  if (party == ALICE)
  {
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  else
  {
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  delete[] r;
  delete[] in2;
  delete[] out2;
  delete[] Tmin;
  delete[] Pmin;
  delete[] Bt;
  delete[] Rt;
  for(int j = 0; j< lt; j++)
  {
    delete[] St[j];
    delete[] pt[j];
  }
  delete[] St;
  delete[] pt;
  delete[] in1;
  delete[] out1;
  delete[] qt;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] rp2;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  for (int k = 0; k < lenD; k++)
  {
    delete[] Result[k];
  }
  Result.clear();
  return resD;
}

uint64_t *SkylineResbyQ_T_old(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 10;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  for (int k1 = 0; k1 <= len1 - 1; k1++)
  {
    for (int k2 = len2 - 1; k2 >= 0; k2--)
    {
      SS_L[k1 * len2 + k2] = SS_L_itr[k1 * len2t + k2];
      SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_L_itr[k1 * len2t + k2]];
      memcpy(SS_Skyline[k1 * len2 + k2], SS_Skyline_itr[k1 * len2t + k2], SS_L_itr[k1 * len2t + k2] * sizeof(uint64_t));
    }
  }
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  uint64_t TMAX = (10000 << MAX) + 10000;
  // uint64_t STMAX = 1ULL << (2 * MAX + 1);
  uint64_t STMAX = 2 * 10000 * 10000;
  if (party == ALICE)
  {
    uint64_t T0 = 0;
    prg.random_data(&T0, sizeof(uint64_t));
    TMAX = (TMAX - T0) & mask;
    Iot[0]->send_data(&T0, sizeof(uint64_t));
    prg.random_data(&T0, sizeof(uint64_t));
    STMAX = (STMAX - T0) & mask;
    Iot[1]->send_data(&T0, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&TMAX, sizeof(uint64_t));
    Iot[1]->recv_data(&STMAX, sizeof(uint64_t)); 
  }    

  // int skylineLen = SS_L[(4000 * SS_G[0][1]) + 2366];
  // uint64_t *ssss = new uint64_t[skylineLen];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + skylineLen, ssss);
  // SS_Print(skylineLen, ssss);

  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = TMAX;
        // tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }
  
  // uint64_t *ss = new uint64_t[len];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + len, ss);
  // SS_Print(len, ss);

  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset = (offset + a[k1 * polylen + i] * t) & mask;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }

  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[len1 * polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[len1 * polylen];
  uint64_t *outD = new uint64_t[polylen];

  uint64_t *qt = new uint64_t[m];
  // qt[0] = SS_G[1][k1 - 1];
  // qt[1] = SS_G[2][k2 - 1];
  for (int j = 0; j < m; j++) {
    qt[j] =  Q[j] & mask; 
  }
  int lt = len;
  uint64_t *S = new uint64_t[lt];
  uint64_t **St = new uint64_t*[lt];
  uint64_t *in1 = new uint64_t[lt*m];
  uint64_t *out1 = new uint64_t[lt*m];
  uint64_t **pt = new uint64_t*[lt];
  uint64_t *Bt = new uint64_t[lt];
  uint64_t *Rt = new uint64_t[lt*m];
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  uint64_t *Pmin = new uint64_t[m];
  uint64_t *Tmin = new uint64_t[m];
  unordered_map<uint32_t, uint64_t *> Result;
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // SS_Print(len1, posindex[0]);
  // SS_Print(len2, posindex[1]);
  // double endscmp = omp_get_wtime();
  // cout << "select element cmp:" << (endscmp-starts) << endl;
  // JT[0] = SS_one;
  // JT[1] = pos[1];
  // double startpoly = omp_get_wtime();
  #pragma omp parallel for
  for (int j = 0; j < len1; j++)
  {
    JT[j * polylen + 0] = SS_one;
    JT[j * polylen + 1] = pos[1];
  }
  IT[0] = SS_one;
  IT[1] = pos[0];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    // JT[i] = outC[0];
    IT[i] = outC[1];
    #pragma omp parallel for
    for (int j = 0; j < len1; j++)
    {
      JT[j * polylen + i] = outC[0];
    } 
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  Prod_H(len1 * polylen, a, JT, outD0, Prodt[0]);
  // double endspoly = omp_get_wtime();
  // cout << "select element poly:" << (endspoly-startpoly) << endl;
  #pragma omp parallel for
  // for (int k1 = 0; k1 < len1; k1++)
  // {
  //   //polynomial
  //   // uint64_t offset = b[0] + b[1] * k1;  // a2*i+b2
  //   // Poly(polylen, b, k1, offset);
  //   uint64_t offset = 0;
  //   uint64_t t = 1;
  //   for (int k = 0; k < polylen; k++)
  //   {
  //     offset = (offset + b[k] * t - outD0[k1 * polylen + k]) & mask;
  //     t = (t * k1) & mask;
  //   } 
  //   for (int k = 0; k < len; k++)
  //   {
  //     // select
  //     // for (int k = 0; k < SS_L[(k1 * len2) + k2]; k++)
  //     for (int k2 = 0; k2 < len2; k2++)
  //     {
  //       tmp2[k1 * len + k] = (tmp2[k1 * len + k] + SS_Skyline[(k1 * len2) + k2][k] * posindex[1][k2]) & mask;
  //     }
  //     tmp2[k1 * len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
  //   }
  // }
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
  }
  // double endsaggr22 = omp_get_wtime();
  // cout << "select element aggr22:" << (endsaggr22-endspoly) << endl;
  for (int k1 = 0; k1 < len1; k1++)
  {
    //polynomial
    // uint64_t mk1 = 0;
    // for (int k = 0; k < polylen; k++)
    // {
    //   mk1 = (mk1 + outD0[k1 * polylen + k]) & mask;
    // }
    // uint64_t offset = (b[0] - mk1) & mask;
    // uint64_t t = k1;
    // for (int i = 1; i < polylen; i++)
    // {
    //   offset = (offset + b[i] * t) & mask;
    //   t = (t * k1) & mask;
    // } 
    uint64_t offset = 0;
    uint64_t t = 1;
    for (int k = 0; k < polylen; k++)
    {
      offset += b[k] * t - outD0[k1 * polylen + k];
      t = (t * k1) & mask;
    } 
    for (int k = 0; k < len; k++)
    {
      tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
    }
  }
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[1]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[1]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  #pragma omp parallel for /* reduction(+ : res) */
  // for (int k1 = 0; k1 < len1; k1++)
  // {
  //   uint64_t pindx = posindex[0][k1];
  //   for (int k = 0; k < len; k++)
  //   {
  //     res[k] += (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * pindx;
  //   }
  // }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  Prod_H(polylen, b, IT, outD, Prodt[0]);
  // #pragma omp parallel for
  for (int k = 0; k < polylen; k++)
  {
      // mk2 = (mk2 + outD[k]) & mask;
      mk2 += outD[k];
  }
  // #pragma omp parallel for
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  double endqua = omp_get_wtime();
  // SS_Print(len, res);
  cout << "select element quasky:" << (endqua - starts) << endl;
  // select dynamic skyline
  // select dynamic skyline == res->result
  int th = 0;
  // dynamic points
  // DP(len, res, qt, lenD, resD);
  // double xx1 = omp_get_wtime();
  //separate each dimension from "<<MAX"
  if (party == ALICE)
  {
    uint64_t masktt = (1ULL << (MAX-2)) - 1;
    #pragma omp parallel for
    for (int j = 0; j < lt*m; j++)
    {
      Rt[j] = Rt[j] & masktt;
    }
    #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (res[i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
    }
    Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
    // Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));
    uint64_t *Rt0 = new uint64_t[lt*m];
    Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
    #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      } 
      // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
      // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
    }
    delete[] Rt0;
  }
  else
  {
    Iot[th]->recv_data(Bt, lt * sizeof(uint64_t)); 
    // #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (Bt[i] + res[i]) & mask;
    }
    uint64_t *s = new uint64_t[m];
    uint64_t masktt = (1ULL << MAX) - 1;
    // uint64_t *Rt0 = new uint64_t[lt*m];
    // Iot[th]->recv_data(Rt0, lt *m * sizeof(uint64_t)); 
    // #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      s[0] = Bt[i] & masktt;
      for (int j = 1; j < m; j++)
      {
        Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
        s[j] = Bt[i] & masktt;
      }
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
        // x[j*lt+i] = pt[i][j] & mask;
      } 
      // for (int j = 0; j < m; j++)
      // {
      //   uint64_t ttt = (s[m-1-j] - Rt0[j*lt+i]) & mask;
      //   cout<< ttt <<",";
      // } 
      // cout<<";";
    }
    delete[] s;
    Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t)); 
    // delete[] Rt0; 
  }
  //Euclid product
  #pragma omp parallel for
  for (int i = 0; i < lt; i++) {
    for (int j = 0; j < m; j++) {
      in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
    }
  }
  Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  // double xx2 = omp_get_wtime();
  // cout << "dy element euclid:" << (xx2 - xx1) << endl;
  #pragma omp parallel for
  for (int i = 0; i < lt; i++) {
    S[i] = 0;
    St[i] = new uint64_t[m];
    for (int j = 0; j < m; j++) {
      St[i][j] = out1[i*m+j] & mask;
      S[i] = (S[i] + out1[i*m+j]) & mask;  
    }
  }
  // Print S
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[lt]; 
  //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
  //   for (int i = 0; i < lt; i++) {
  //     cout<< ((STTT[i] + S[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }
  
  uint64_t lam = 1;  
  double xx3 = omp_get_wtime();
  // cout << "dy element St:" << (xx3 - xx2) << endl; 
  uint64_t *r = new uint64_t[1];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *out2 = new uint64_t[1];
  while(lam!=0){
    uint64_t STMin = 0;
    // double tt1 = omp_get_wtime();
    SMIN_T(0, lt, S, pt, St, STMin, Pmin, Tmin);
    // double tt2 = omp_get_wtime();
    // cout << "loop element Smin:" << (tt2 - tt1) << endl; 
    prg.random_data(r, sizeof(uint64_t));
    in2[0] = (STMAX - STMin) & mask;
    Prod_H(1, in2, r, out2, Prodt[th]);
    if (party == ALICE)
    {
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      Iot[th+1]->send_data(out2, sizeof(uint64_t));
      lam = (lam + out2[0]) & mask;
    }
    else
    {
      Iot[th]->send_data(out2, sizeof(uint64_t));
      Iot[th+1]->recv_data(&lam, sizeof(uint64_t)); 
      lam = (lam + out2[0]) & mask;
    }
    // cout<<"lam:"<<lam<<endl;
    // if (party == ALICE)
    // {
    //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
    //   Iot[th+1]->send_data(&STMin, sizeof(uint64_t));
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    // else
    // {
    //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
    //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    
    // double tt3 = omp_get_wtime();
    // cout << "loop element break:" << (tt3 - tt2) << endl;
    if(lam!=0){
      int pos = Result.size();
      // cout<<pos<<endl;
      Result[pos] = new uint64_t[m];
      // for (int i = 0; i < m; i++)
      // {
      //   Result[pos][i] = Pmin[i];
      // }
      memcpy(Result[pos], Pmin, m * sizeof(uint64_t));
      // Result[pos] = Pmin;
      // cout<<Pmin[0]<<","<<Result[pos][0]<<endl;
      // if (party == ALICE)
      // {
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[0], sizeof(uint64_t));
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[1], sizeof(uint64_t));
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      // else
      // {
      //   Iot[th]->send_data(&Pmin[0], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->send_data(&Pmin[1], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
    //eliminate
      SDOMbyMin_T(lt, STMAX, Tmin, St, STMin, S);
      //Print S
      // if (party == ALICE)
      // {
      //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
      // }
      // else
      // {
      //   uint64_t *STTT =  new uint64_t[lt]; 
      //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
      //   for (int i = 0; i < lt; i++) {
      //     cout<< ((STTT[i] + S[i]) & mask) << ",";
      //   }
      //   cout<<endl;
      //   delete[] STTT;
      // }  
    }
    // double tt4 = omp_get_wtime();
    // cout << "loop element filter:" << (tt4 - tt3) << endl;
  }
  double xx4 = omp_get_wtime();
  cout << "dy element loop:" << (xx4 - xx3) << endl;
  // Return
  // A
  int lenD = Result.size();
  uint64_t *resD = new uint64_t[lenD*m];
  for (int i = 0; i < lenD; i++)
  {
    for (int j = 0; j < m; j++)
    {
      resD[i*m+j] = Result[i][j];
    }
  }
  reslen = lenD;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  cout<<lenD<<endl;
  if (party == ALICE)
  {
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  else
  {
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  delete[] r;
  delete[] in2;
  delete[] out2;
  delete[] Tmin;
  delete[] Pmin;
  delete[] Bt;
  delete[] Rt;
  for(int j = 0; j< lt; j++)
  {
    delete[] St[j];
    delete[] pt[j];
  }
  delete[] St;
  delete[] pt;
  delete[] in1;
  delete[] out1;
  delete[] qt;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  for (int k = 0; k < lenD; k++)
  {
    delete[] Result[k];
  }
  Result.clear();
  return resD;
}

uint64_t *SkylineResbyQ_T(vector<uint32_t> Q, uint32_t &reslen)
{
  int polylen = 2;
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  for (int k1 = 0; k1 <= len1 - 1; k1++)
  {
    for (int k2 = len2 - 1; k2 >= 0; k2--)
    {
      SS_L[k1 * len2 + k2] = SS_L_itr[k1 * len2t + k2];
      SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_L_itr[k1 * len2t + k2]];
      memcpy(SS_Skyline[k1 * len2 + k2], SS_Skyline_itr[k1 * len2t + k2], SS_L_itr[k1 * len2t + k2] * sizeof(uint64_t));
    }
  }
  uint64_t *rp2 = new uint64_t[len1];
  prg.random_data(rp2, len1 * sizeof(uint64_t));
  // memset(rp2, 0, len1 * sizeof(uint64_t));
  uint64_t rp1 = 0;
  prg.random_data(&rp1, sizeof(uint64_t));
  uint64_t *a = new uint64_t[len1 * polylen];
  prg.random_data(a, len1 * polylen * sizeof(uint64_t));
  // uint64_t *a = new uint64_t[len1];
  // uint64_t *b = new uint64_t[len1];
  // prg.random_data(a, len1 * sizeof(uint64_t));
  // prg.random_data(b, len1 * sizeof(uint64_t));
  uint64_t *b = new uint64_t[polylen];
  prg.random_data(b, polylen * sizeof(uint64_t));
  // uint64_t a2 = 0;
  // uint64_t b2 = 0;
  // prg.random_data(&a2, sizeof(uint64_t));
  // prg.random_data(&b2, sizeof(uint64_t));
  // dummy element
  double startd = omp_get_wtime();
  cout << "dummy element" << endl;
  int len = 0;
  #pragma omp parallel for reduction(max:len)
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      if (len < SS_L[(k1 * len2) + k2])
        len = SS_L[(k1 * len2) + k2];
    }
  }
  cout << "len:"<< len << endl;
  uint64_t SS_one = 0, SS_zero = 0;
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  uint64_t TMAX = (10000 << MAX) + 10000;
  // uint64_t STMAX = 1ULL << (2 * MAX + 1);
  uint64_t STMAX = 2 * 10000 * 10000;
  if (party == ALICE)
  {
    uint64_t T0 = 0;
    prg.random_data(&T0, sizeof(uint64_t));
    TMAX = (TMAX - T0) & mask;
    Iot[0]->send_data(&T0, sizeof(uint64_t));
    prg.random_data(&T0, sizeof(uint64_t));
    STMAX = (STMAX - T0) & mask;
    Iot[1]->send_data(&T0, sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(&TMAX, sizeof(uint64_t));
    Iot[1]->recv_data(&STMAX, sizeof(uint64_t)); 
  }    

  // int skylineLen = SS_L[(4000 * SS_G[0][1]) + 2366];
  // uint64_t *ssss = new uint64_t[skylineLen];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + skylineLen, ssss);
  // SS_Print(skylineLen, ssss);

  #pragma omp parallel for
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      uint64_t *tmp0 = new uint64_t[len];
      int slen = SS_L[(k1 * len2) + k2];
      if (len == slen)
        continue;
      memcpy(tmp0, SS_Skyline[(k1 * len2) + k2], slen * sizeof(uint64_t));
      for (int k = slen; k < len; k++)
      {
        // if (party == ALICE)
        // {
        //   prg.random_data(&SS_zero, sizeof(uint64_t));
        //   Iot[0]->send_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = SS_zero;
        // }
        // else
        // {
        //   Iot[0]->recv_data(&SS_zero, sizeof(uint64_t));
        //   tmp0[k] = (0 - SS_zero) & mask;
        // }
        tmp0[k] = TMAX;
        // tmp0[k] = SS_zero;
      }
      delete[] SS_Skyline[(k1 * len2) + k2];
      SS_Skyline[(k1 * len2) + k2] = new uint64_t[len];
      memcpy(SS_Skyline[(k1 * len2) + k2], tmp0, len * sizeof(uint64_t));
      delete[] tmp0;
    }
  }
  
  // uint64_t *ss = new uint64_t[len];
  // copy(SS_Skyline[(4000 * SS_G[0][1]) + 2366], SS_Skyline[(4000 * SS_G[0][1]) + 2366] + len, ss);
  // SS_Print(len, ss);

  double endd = omp_get_wtime();
  ss_dummy = endd - startd;
  cout << ss_dummy << " s" << endl;
  uint64_t comm_1 = 0;
  for(int j = 0; j< THs; j++){
    comm_1+=Iot[j]->counter;
  }
  double startm = omp_get_wtime();
  cout << "mask element" << endl;
  // m+a*i+b
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    // uint64_t mac = rp2[k1];
    for (int k2 = 0; k2 < len2; k2++)
    {
      // construct the same skyline
      // int slen = SS_L[(k1<<MAX)+k2];
      // uint64_t offset = a[k1][0] + a[k1][1]* k2 ; // a*j+b
      // uint64_t offset = 0;
      // Poly(polylen, a[k1], k2, offset);
      uint64_t offset = a[k1 * polylen];
      uint64_t t = k2;
      for (int i = 1; i < polylen; i++)
      {
        offset += a[k1 * polylen + i] * t;
        t = (t * k2) & mask;
      }
      // int lent = SS_L[(k1 * len2) + k2];
      // for (int k = 0; k < lent; k++)
      // {
      //   SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      // }
      offset = offset & mask;
      // offset = mac + HashP(offset);
      offset = rp2[k1] + HashP(offset);
      for (int k = 0; k < len; k++)
      {
        SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + offset) & mask;
      }
    }
  }
  // exchange the copies
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, SS_L)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            for (int k2 = 0; k2 < len2; k2++)
            {
              // int lent = SS_L[(k1 * len2) + k2];
              uint64_t *tmp1 = new uint64_t[len];
              if (party == ALICE)
              {
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
              }
              else
              {
                Iot[itr]->recv_data(tmp1, len * sizeof(uint64_t));
                Iot[itr]->send_data(SS_Skyline[(k1 * len2) + k2], len * sizeof(uint64_t));
              }
              for (int k = 0; k < len; k++)
              {
                SS_Skyline[(k1 * len2) + k2][k] = (SS_Skyline[(k1 * len2) + k2][k] + tmp1[k]) & mask;
              }
              delete[] tmp1;
            }
          }
        }
      }
      #pragma omp taskwait
    }  
  }

  double endm = omp_get_wtime();
  ss_mask = endm - startm;
  cout << ss_mask << " s" << endl;
  uint64_t comm_2 = 0;
  for(int j = 0; j< THs; j++){
    comm_2+=Iot[j]->counter;
  }
  com1 = comm_2-comm_1;
  cout << "select element" << endl;
  // select
  uint64_t *res = new uint64_t[len]();
  uint64_t *inA = new uint64_t[m];
  uint64_t *inB = new uint64_t[m];
  uint64_t *outC = new uint64_t[m];
  uint64_t *JT = new uint64_t[polylen];
  uint64_t *IT = new uint64_t[polylen];
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  uint64_t *tmp2 = new uint64_t[len1 * len];
  uint64_t *tmpt = new uint64_t[len1 * len];
  uint64_t *outD0 = new uint64_t[1];
  uint64_t *outD = new uint64_t[1];

  uint64_t *qt = new uint64_t[m];
  // qt[0] = SS_G[1][k1 - 1];
  // qt[1] = SS_G[2][k2 - 1];
  for (int j = 0; j < m; j++) {
    qt[j] =  Q[j] & mask; 
  }
  int lt = len;
  uint64_t *S = new uint64_t[lt];
  uint64_t **St = new uint64_t*[lt];
  uint64_t *in1 = new uint64_t[lt*m];
  uint64_t *out1 = new uint64_t[lt*m];
  uint64_t **pt = new uint64_t*[lt];
  uint64_t *Bt = new uint64_t[lt];
  uint64_t *Rt = new uint64_t[lt*m];
  prg.random_data(Rt, lt*m*sizeof(uint64_t));
  uint64_t *Pmin = new uint64_t[m];
  uint64_t *Tmin = new uint64_t[m];
  unordered_map<uint32_t, uint64_t *> Result;
  double starts = omp_get_wtime();
  PosIndex(Q, SS_G, pos, posindex);
  // SS_Print(len1, posindex[0]);
  // SS_Print(len2, posindex[1]);
  // double endscmp = omp_get_wtime();
  // cout << "select element cmp:" << (endscmp-starts) << endl;
  // JT[0] = SS_one;
  // JT[1] = pos[1];
  // double startpoly = omp_get_wtime();
  IT[0] = SS_one;
  IT[1] = pos[0];
  JT[0] = SS_one;
  JT[1] = pos[1];
  for (int i = 2; i < polylen; i++)
  {
    inA[0] = JT[i/2];
    inB[0] = JT[i - i/2];
    inA[1] = IT[i/2];
    inB[1] = IT[i - i/2];
    Prod_H(m, inA, inB, outC, Prodt[0]);
    JT[i] = outC[0];
    IT[i] = outC[1];
  }
  memset(tmp2, 0, (len1 * len) * sizeof(uint64_t));
  // double endspoly = omp_get_wtime();
  // cout << "select element poly:" << (endspoly-startpoly) << endl;
  #pragma omp parallel for
  for (int k1 = 0; k1 < len1; k1++)
  {
    for (int k2 = 0; k2 < len2; k2++)
    {
      // select
      uint64_t pindx = posindex[1][k2];
      uint64_t* skl_ = SS_Skyline[(k1 * len2) + k2];
      // #pragma omp simd
      for (int k = 0; k < len; k++)
      {
        tmp2[k1 * len + k] += (skl_[k] * pindx)/* & mask */;
      }
    }
  }
  // double endsaggr22 = omp_get_wtime();
  // cout << "select element aggr22:" << (endsaggr22-endspoly) << endl;
  MultMode mode1 = MultMode::Alice_has_A;
  MultMode mode2 = MultMode::Bob_has_B;
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(itr, THs, a, rp1, rp2)
        {
          int lendt = (((len1 - 1) * itr)/ THs) + 1;
          int lenut = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          uint64_t *coe_a = new uint64_t[polylen];
          uint64_t *outDA = new uint64_t[1];
          uint64_t *outDB = new uint64_t[1];
          for (int k1 = lendt; k1 <= lenut; k1++)
          {
            uint64_t offset = b[0];
            uint64_t t = k1;
            for (int k = 1; k < polylen; k++)
            {
              offset += b[k] * t;
              t = (t * k1) & mask;
            } 
            offset = offset & mask;
            offset = rp1 + HashP(offset);
            memcpy(coe_a, a+k1*polylen, polylen * sizeof(uint64_t));
            uint64_t tmp = 0;
            Prod_M(1, polylen, 1, coe_a, JT, outDA, Prodt[itr], mode1);
            Prod_M(1, polylen, 1, JT, coe_a, outDB, Prodt[itr], mode2);
            if (party == ALICE)
            { // send outDA
              Iot[itr]->send_data(outDA, sizeof(uint64_t));
              Iot[itr]->recv_data(&tmp, sizeof(uint64_t));
              tmp = (outDB[0] + tmp) & mask;
            }
            else
            { //send outDB
              Iot[itr]->recv_data(&tmp, sizeof(uint64_t));
              Iot[itr]->send_data(outDB, sizeof(uint64_t));
              tmp = (outDA[0] + tmp) & mask;
            }
            tmp = HashP(tmp);
            offset = (offset - rp2[k1] - tmp) & mask;
            for (int k = 0; k < len; k++)
            {
              tmp2[k1* len + k] = (tmp2[k1 * len + k] + offset) & mask; //(m+a[k1]*j+b[k1]) + a2*i+b2 - a[k1]*j-b[k1]
            }
          }
          delete[] coe_a;
          delete[] outDA;
          delete[] outDB;
        }
      }
      #pragma omp taskwait
    }  
  }
  // double endsaggr = omp_get_wtime();
  // cout << "select element aggr:" << (endsaggr-endspoly) << endl;
  if (party == ALICE)
  {
    Iot[0]->send_data(tmp2, len1 * len * sizeof(uint64_t));
    Iot[1]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
  }
  else
  {
    Iot[0]->recv_data(tmpt, len1 * len * sizeof(uint64_t));
    Iot[1]->send_data(tmp2, len1 * len * sizeof(uint64_t));
  }
  #pragma omp parallel for /* reduction(+ : res) */
  // for (int k1 = 0; k1 < len1; k1++)
  // {
  //   uint64_t pindx = posindex[0][k1];
  //   for (int k = 0; k < len; k++)
  //   {
  //     res[k] += (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * pindx;
  //   }
  // }
  for (int k = 0; k < len; k++)
  {
    for (int k1 = 0; k1 < len1; k1++)
    {
      res[k] = (res[k] + (tmpt[k1 * len + k] + tmp2[k1 * len + k]) * posindex[0][k1]) & mask;
    }
  }
  // double endsaggr2 = omp_get_wtime();
  // cout << "select element aggr2:" << (endsaggr2-endsaggr) << endl;
  //a2*pos[0]+b2
  uint64_t mk2 = 0;
  uint64_t tmp = 0;
  Prod_M(1, polylen, 1, b, IT, outD0, Prodt[0], mode1);
  Prod_M(1, polylen, 1, IT, b, outD, Prodt[0], mode2);
  if (party == ALICE)
  {
    Iot[0]->send_data(outD0, sizeof(uint64_t));
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    tmp = (outD[0] + tmp) & mask;
  }
  else
  {
    Iot[0]->recv_data(&tmp, sizeof(uint64_t));
    Iot[0]->send_data(outD, sizeof(uint64_t));
    tmp = (outD0[0] + tmp) & mask;
  }
  tmp = HashP(tmp);
  mk2 = (rp1 + tmp) & mask;
  // #pragma omp parallel for
  for (int k = 0; k < len; k++)
  {
    res[k] = (res[k] - mk2) & mask; //(m+a2*i+b2) - a2*i+b2
  }
  double endqua = omp_get_wtime();
  // SS_Print(len, res);
  cout << "select element quasky:" << (endqua - starts) << endl;
  ss_selectSky = endqua - starts;
  // select dynamic skyline
  // select dynamic skyline == res->result
  int th = 0;
  // dynamic points
  // DP(len, res, qt, lenD, resD);
  // double xx1 = omp_get_wtime();
  //separate each dimension from "<<MAX"
  if (party == ALICE)
  {
    uint64_t masktt = (1ULL << (MAX-2)) - 1;
    #pragma omp parallel for
    for (int j = 0; j < lt*m; j++)
    {
      Rt[j] = Rt[j] & masktt;
    }
    #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (res[i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
    }
    Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
    // Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));
    uint64_t *Rt0 = new uint64_t[lt*m];
    Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
    #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      } 
      // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
      // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
    }
    delete[] Rt0;
  }
  else
  {
    Iot[th]->recv_data(Bt, lt * sizeof(uint64_t)); 
    // #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      Bt[i] = (Bt[i] + res[i]) & mask;
    }
    uint64_t *s = new uint64_t[m];
    uint64_t masktt = (1ULL << MAX) - 1;
    // uint64_t *Rt0 = new uint64_t[lt*m];
    // Iot[th]->recv_data(Rt0, lt *m * sizeof(uint64_t)); 
    // #pragma omp parallel for
    for (int i = 0; i < lt; i++)
    {
      pt[i] = new uint64_t[m];
      s[0] = Bt[i] & masktt;
      for (int j = 1; j < m; j++)
      {
        Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
        s[j] = Bt[i] & masktt;
      }
      for (int j = 0; j < m; j++)
      {
        pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
        // x[j*lt+i] = pt[i][j] & mask;
      } 
      // for (int j = 0; j < m; j++)
      // {
      //   uint64_t ttt = (s[m-1-j] - Rt0[j*lt+i]) & mask;
      //   cout<< ttt <<",";
      // } 
      // cout<<";";
    }
    delete[] s;
    Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t)); 
    // delete[] Rt0; 
  }
  //Euclid product
  #pragma omp parallel for
  for (int i = 0; i < lt; i++) {
    for (int j = 0; j < m; j++) {
      in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
    }
  }
  Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  // double xx2 = omp_get_wtime();
  // cout << "dy element euclid:" << (xx2 - xx1) << endl;
  #pragma omp parallel for
  for (int i = 0; i < lt; i++) {
    S[i] = 0;
    St[i] = new uint64_t[m];
    for (int j = 0; j < m; j++) {
      St[i][j] = out1[i*m+j] & mask;
      S[i] = (S[i] + out1[i*m+j]) & mask;  
    }
  }
  // Print S
  // if (party == ALICE)
  // {
  //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  // }
  // else
  // {
  //   uint64_t *STTT =  new uint64_t[lt]; 
  //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
  //   for (int i = 0; i < lt; i++) {
  //     cout<< ((STTT[i] + S[i]) & mask) << ",";
  //   }
  //   cout<<endl;
  //   delete[] STTT;
  // }
  
  uint64_t lam = 1;  
  double xx3 = omp_get_wtime();
  // cout << "dy element St:" << (xx3 - xx2) << endl; 
  uint64_t *r = new uint64_t[1];
  uint64_t *in2 = new uint64_t[1];
  uint64_t *out2 = new uint64_t[1];
  while(lam!=0){
    uint64_t STMin = 0;
    // double tt1 = omp_get_wtime();
    SMIN_T(0, lt, S, pt, St, STMin, Pmin, Tmin);
    // double tt2 = omp_get_wtime();
    // cout << "loop element Smin:" << (tt2 - tt1) << endl; 
    prg.random_data(r, sizeof(uint64_t));
    in2[0] = (STMAX - STMin) & mask;
    Prod_H(1, in2, r, out2, Prodt[th]);
    if (party == ALICE)
    {
      Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
      Iot[th+1]->send_data(out2, sizeof(uint64_t));
      lam = (lam + out2[0]) & mask;
    }
    else
    {
      Iot[th]->send_data(out2, sizeof(uint64_t));
      Iot[th+1]->recv_data(&lam, sizeof(uint64_t)); 
      lam = (lam + out2[0]) & mask;
    }
    // cout<<"lam:"<<lam<<endl;
    // if (party == ALICE)
    // {
    //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
    //   Iot[th+1]->send_data(&STMin, sizeof(uint64_t));
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    // else
    // {
    //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
    //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
    //   cout<<"SMIN:"<<((STMin + out2[0]) & mask)<<endl;
    // }
    
    // double tt3 = omp_get_wtime();
    // cout << "loop element break:" << (tt3 - tt2) << endl;
    if(lam!=0){
      int pos = Result.size();
      // cout<<pos<<endl;
      Result[pos] = new uint64_t[m];
      // for (int i = 0; i < m; i++)
      // {
      //   Result[pos][i] = Pmin[i];
      // }
      memcpy(Result[pos], Pmin, m * sizeof(uint64_t));
      // Result[pos] = Pmin;
      // cout<<Pmin[0]<<","<<Result[pos][0]<<endl;
      // if (party == ALICE)
      // {
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[0], sizeof(uint64_t));
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->recv_data(out2, sizeof(uint64_t)); 
      //   Iot[th+1]->send_data(&Pmin[1], sizeof(uint64_t));
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
      // else
      // {
      //   Iot[th]->send_data(&Pmin[0], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<"PMIN:"<<((Pmin[0] + out2[0]) & mask);
      //   Iot[th]->send_data(&Pmin[1], sizeof(uint64_t));
      //   Iot[th+1]->recv_data(out2, sizeof(uint64_t)); 
      //   cout<<","<<((Pmin[1] + out2[0]) & mask)<<endl;
      // }
    //eliminate
      SDOMbyMin_T(lt, STMAX, Tmin, St, STMin, S);
      //Print S
      // if (party == ALICE)
      // {
      //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
      // }
      // else
      // {
      //   uint64_t *STTT =  new uint64_t[lt]; 
      //   Iot[th]->recv_data(STTT, lt * sizeof(uint64_t));
      //   for (int i = 0; i < lt; i++) {
      //     cout<< ((STTT[i] + S[i]) & mask) << ",";
      //   }
      //   cout<<endl;
      //   delete[] STTT;
      // }  
    }
    // double tt4 = omp_get_wtime();
    // cout << "loop element filter:" << (tt4 - tt3) << endl;
  }
  double xx4 = omp_get_wtime();
  cout << "dy element loop:" << (xx4 - xx3) << endl;
  // Return
  // A
  int lenD = Result.size();
  uint64_t *resD = new uint64_t[lenD*m];
  for (int i = 0; i < lenD; i++)
  {
    for (int j = 0; j < m; j++)
    {
      resD[i*m+j] = Result[i][j];
    }
  }
  reslen = lenD;
  double ends = omp_get_wtime();
  ss_select = ends - starts;
  cout << ss_select << " s" << endl;
  uint64_t comm_3 = 0;
  for(int j = 0; j< THs; j++){
    comm_3+=Iot[j]->counter;
  }
  com2 = comm_3-comm_2;
  cout<<lenD<<endl;
  if (party == ALICE)
  {
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  else
  {
    uint64_t *STTT =  new uint64_t[lenD*m]; 
    Iot[th]->recv_data(STTT, lenD*m * sizeof(uint64_t));
    Iot[th]->send_data(resD, lenD*m * sizeof(uint64_t));
    for (int i = 0; i < lenD; i++) {
      cout<< ((STTT[2*i] + resD[2*i]) & mask) << ","<< ((STTT[2*i+1] + resD[2*i+1]) & mask) << endl;
    }
    delete[] STTT;
  }
  delete[] r;
  delete[] in2;
  delete[] out2;
  delete[] Tmin;
  delete[] Pmin;
  delete[] Bt;
  delete[] Rt;
  for(int j = 0; j< lt; j++)
  {
    delete[] St[j];
    delete[] pt[j];
  }
  delete[] St;
  delete[] pt;
  delete[] in1;
  delete[] out1;
  delete[] qt;
  delete[] a;
  delete[] b;
  delete[] JT;
  delete[] IT;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete[] outD;
  delete[] outD0;
  delete[] rp2;
  delete[] tmpt;
  delete[] tmp2;
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  for (int k = 0; k < lenD; k++)
  {
    delete[] Result[k];
  }
  Result.clear();
  return resD;
}

void Test_poly()
{
  int *tt = new int[4];
  tt[0] = 10;
  tt[1] = 100;
  tt[2] = 1000;
  tt[3] = n;
  for (int kk = 0; kk < 4; kk++)
  {
    double sr = omp_get_wtime();
    for (int xx = 0; xx < 10; xx++)
    {
      int polylen = tt[kk];
      int len1 = SS_G[0][0];
      int len2 = SS_G[0][1];
      uint64_t *outD0 = new uint64_t[len1 * polylen];
      uint64_t *outD = new uint64_t[polylen];
      uint64_t *a = new uint64_t[len1 * polylen];
      prg.random_data(a, len1 * polylen * sizeof(uint64_t));
      uint64_t *b = new uint64_t[polylen];
      prg.random_data(b, polylen * sizeof(uint64_t));
      uint64_t SS_one = 0, SS_zero = 0;
      if (party == ALICE)
      {
        prg.random_data(&SS_one, sizeof(uint64_t));
        Iot[0]->send_data(&SS_one, sizeof(uint64_t));
        prg.random_data(&SS_zero, sizeof(uint64_t));
        Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
      }
      else
      {
        Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
        SS_one = (1 - SS_one) & mask;
        Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
        SS_zero = (0 - SS_zero) & mask;
      }
      uint32_t *pos = new uint32_t[2];
      prg.random_data(pos, 2 * sizeof(uint32_t));
      uint64_t *inA = new uint64_t[m];
      uint64_t *inB = new uint64_t[m];
      uint64_t *outC = new uint64_t[m];
      uint64_t *JT = new uint64_t[len1 * polylen];
      uint64_t *IT = new uint64_t[polylen];
      double startpoly = omp_get_wtime();
      for (int j = 0; j < len1; j++)
      {
        JT[j * polylen + 0] = SS_one;
        JT[j * polylen + 1] = pos[1];
      }
      IT[0] = SS_one;
      IT[1] = pos[0];
      for (int i = 2; i < polylen; i++)
      {
        inA[0] = JT[i/2];
        inB[0] = JT[i - i/2];
        inA[1] = IT[i/2];
        inB[1] = IT[i - i/2];
        Prod_H(m, inA, inB, outC, Prodt[0]);
        // JT[i] = outC[0];
        IT[i] = outC[1];
        for (int j = 0; j < len1; j++)
        {
          JT[j * polylen + i] = outC[0];
        } 
      }
      Prod_H(len1 * polylen, a, JT, outD0, Prodt[0]);
      Prod_H(polylen, b, IT, outD, Prodt[0]);
      double endspoly = omp_get_wtime();
      cout << "select element poly:" << (endspoly-startpoly) << endl;
      delete[] outD;
      delete[] outD0;
      delete[] a;
      delete[] b;
      delete[] inA;
      delete[] inB;
      delete[] outC;
      delete[] JT;
      delete[] IT;
      delete[] pos;
    }
    double er = omp_get_wtime();
    double sp = er - sr;
    cout<< "poly:" << (sp/10) << " s" << endl;
  }
  delete[] tt;
}

void Test_poly_T()
{
  int *tt = new int[4];
  tt[0] = 10;
  tt[1] = 100;
  tt[2] = 1000;
  tt[3] = n;
  for (int kk = 0; kk < 4; kk++)
  {
    int polylen = tt[kk];
    int len1 = SS_G[0][0];
    int len2 = SS_G[0][1];
    uint64_t *outD0 = new uint64_t[len1 * polylen];
    uint64_t *outD = new uint64_t[polylen];
    uint64_t *a = new uint64_t[len1 * polylen];
    prg.random_data(a, len1 * polylen * sizeof(uint64_t));
    uint64_t *b = new uint64_t[polylen];
    prg.random_data(b, polylen * sizeof(uint64_t));
    uint64_t SS_one = 0, SS_zero = 0;
    if (party == ALICE)
    {
      prg.random_data(&SS_one, sizeof(uint64_t));
      Iot[0]->send_data(&SS_one, sizeof(uint64_t));
      prg.random_data(&SS_zero, sizeof(uint64_t));
      Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    }
    else
    {
      Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
      SS_one = (1 - SS_one) & mask;
      Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
      SS_zero = (0 - SS_zero) & mask;
    }
    uint32_t *pos = new uint32_t[2];
    prg.random_data(pos, 2 * sizeof(uint32_t));
    uint64_t *inA = new uint64_t[m];
    uint64_t *inB = new uint64_t[m];
    uint64_t *outC = new uint64_t[m];
    uint64_t *JT = new uint64_t[len1 * polylen];
    uint64_t *IT = new uint64_t[polylen];
    double startpoly = omp_get_wtime();
    #pragma omp parallel for
    for (int j = 0; j < len1; j++)
    {
      JT[j * polylen + 0] = SS_one;
      JT[j * polylen + 1] = pos[1];
    }
    IT[0] = SS_one;
    IT[1] = pos[0];
    for (int i = 2; i < polylen; i++)
    {
      inA[0] = JT[i/2];
      inB[0] = JT[i - i/2];
      inA[1] = IT[i/2];
      inB[1] = IT[i - i/2];
      Prod_H(m, inA, inB, outC, Prodt[0]);
      // JT[i] = outC[0];
      IT[i] = outC[1];
      #pragma omp parallel for
      for (int j = 0; j < len1; j++)
      {
        JT[j * polylen + i] = outC[0];
      } 
    }
    Prod_H(len1 * polylen, a, JT, outD0, Prodt[0]);
    Prod_H(polylen, b, IT, outD, Prodt[0]);
    double endspoly = omp_get_wtime();
    cout << "select element poly:" << (endspoly-startpoly) << endl;
    delete[] outC;
    delete[] outD;
    delete[] outD0;
    delete[] a;
    delete[] b;
    delete[] inA;
    delete[] inB;
    delete[] outC;
    delete[] JT;
    delete[] IT;
  }
}

void StoreSKyline()
{
  if (party == ALICE)
  {
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + "_Skyline_A.csv", ios::app | ios::in);
    int len1 = SS_G[0][0];
    int len2 = SS_G[0][1];
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
  else
  {
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + "_Skyline_B.csv", ios::app | ios::in);
    int len1 = SS_G[0][0];
    int len2 = SS_G[0][1];
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
}

void ReadSKyline()
{
  string line;
  int k1 = 0;
  int k2 = 0;
  uint64_t *lineArray = new uint64_t[500];
  if (party == ALICE)
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_A.csv");
    while (getline(inf0, line0))
    {     
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_A.csv");
    while (getline(inf, line))
    {
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
  }
  else
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_B.csv");
    while (getline(inf0, line0))
    {
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_B.csv");
    while (getline(inf, line))
    {
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
  }
  delete[] lineArray;

  // cout<<k1<<","<<k2<<endl;
  // for (int j = 0; j<= k1-1; j++) {
  //   for (int i = 0; i <=k2-1; i++) {
  //     vector<uint32_t> xx(SS_L[(j * len2)+i]);
  //     copy(SS_Skyline[(j * len2)+i],SS_Skyline[(j * len2)+i]+SS_L[(j * len2)+i],xx.begin());
  // 		SS_Print(xx);
  //     cout << " ";
  //   }
  //   cout << endl;
  // }
}

void StoreSKyline(string path)
{
  if (party == ALICE)
  {
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_A.csv", ios::app | ios::in);
    int len1 = SS_G[0][0];
    int len2 = SS_G[0][1];
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
  else
  {
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_B.csv", ios::app | ios::in);
    int len1 = SS_G[0][0];
    int len2 = SS_G[0][1];
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
}

void ReadSKyline(string path)
{
  string line;
  int k1 = 0;
  int k2 = 0;
  uint64_t *lineArray = new uint64_t[1000];
  if (party == ALICE)
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_A.csv");
    while (getline(inf0, line0))
    {     
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_A.csv");
    while (getline(inf, line))
    {
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
  }
  else
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_B.csv");
    while (getline(inf0, line0))
    {
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_B.csv");
    while (getline(inf, line))
    {
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
  }
  delete[] lineArray;

  // cout<<k1<<","<<k2<<endl;
  // for (int j = 0; j<= k1-1; j++) {
  //   for (int i = 0; i <=k2-1; i++) {
  //     vector<uint32_t> xx(SS_L[(j * len2)+i]);
  //     copy(SS_Skyline[(j * len2)+i],SS_Skyline[(j * len2)+i]+SS_L[(j * len2)+i],xx.begin());
  // 		SS_Print(xx);
  //     cout << " ";
  //   }
  //   cout << endl;
  // }
}

void Store(string path)
{
  string line;
  int k1 = 0;
  int k2 = 0;
  uint64_t *lineArray = new uint64_t[1000];
  // cout<<"../../tests/grid/" + dataname + to_string(n) + "_Skyline_A.csv"<<endl;
  // cout<<"../../tests/grid/" + dataname + to_string(n) + path + "_Skyline_A.csv"<<endl;
  if (party == ALICE)
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_A.csv");
    while (getline(inf0, line0))
    {     
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) +"_Skyline_A0.csv");
    while (getline(inf, line))
    {
      if(k1>=len1){break;}
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + "_Skyline_A.csv", ios::app | ios::in);
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
  else
  {
    int len1 = 0;
    int len2 = 0;
    string line0;
    ifstream inf0;
    inf0.open("../../tests/grid/" + dataname + to_string(n) + "-1_Skyline_B.csv");
    while (getline(inf0, line0))
    {
      if(len1 == 0)
      {
        istringstream sin0(line0);
        string field0;
        while (getline(sin0, field0, ','))
        {
          len2++;
        }
      }
      len1++;
    }
    inf0.close();
    cout << "K1:" << len1 << endl;
    cout << "K2:" << len2 << endl;
    SS_L = new int[len1 * len2];
    SS_Skyline = new uint64_t*[len1 * len2];
    ifstream inf;
    inf.open("../../tests/grid/" + dataname + to_string(n) + "_Skyline_B0.csv");
    while (getline(inf, line))
    {
      if(k1>=len1){break;}
      istringstream sin(line);
      string field;
      k2 = 0;
      while (getline(sin, field, ','))
      {
        istringstream sin2(field);
        string skyline;
        int j = 0;
        while (getline(sin2, skyline, ';'))
        {
          uint64_t x = atoi(skyline.c_str());
          lineArray[j] = x;
          j++;
        }
        SS_L[(k1 * len2) + k2] = (int) lineArray[0];
        SS_Skyline[(k1 * len2) + k2] = new uint64_t[SS_L[(k1 * len2) + k2]];
        for (int i = 0; i < SS_L[(k1 * len2) + k2]; i++)
        {
          SS_Skyline[(k1 * len2) + k2][i] = lineArray[i + 1];
        }
        k2++;
      }
      k1++;
    }
    inf.close();
    ofstream outfile;
    outfile.open("../../tests/grid/" + dataname + to_string(n) + "_Skyline_B.csv", ios::app | ios::in);
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      for (int k2 = 0; k2 <= len2 - 2; k2++)
      {
        outfile << SS_L[(k1 * len2) + k2] << ";";
        for (int i = 0; i < SS_L[(k1 * len2) + k2] - 1; i++)
        {
          outfile << SS_Skyline[(k1 * len2) + k2][i] << ";";
        }
        outfile << SS_Skyline[(k1 * len2) + k2][SS_L[(k1 * len2) + k2] - 1] << ",";
      }
      outfile << SS_L[(k1 * len2) + len2 - 1] << ";";
      for (int i = 0; i < SS_L[(k1 * len2) + len2 - 1] - 1; i++)
      {
        outfile << SS_Skyline[(k1 * len2) + len2 - 1][i] << ";";
      }
      outfile << SS_Skyline[(k1 * len2) + len2 - 1][SS_L[(k1 * len2) + len2 - 1] - 1];
      outfile << endl;
    }
    outfile.close();
  }
}

void SSub_SS_Two_1(int lenl, int lenr, int lend, int lenu, int th)
{
  int LENX = SS_G[0][1];  //len2
  int len_x = lenr - lenl + 1;
  int len_y = lenu - lend + 1;
  int *L = new int[len_x + len_y - 1];
  uint64_t ** Skylinet = new uint64_t*[len_x + len_y - 1];
  // Process
  double starts = omp_get_wtime();
  L[len_x - 1] = 1;
  // con * point + (1-con) * (10000))
  uint64_t *inA1 = new uint64_t[1];
  uint64_t *inB1 = new uint64_t[1];
  uint64_t *outC1 = new uint64_t[1];
  inA1[0] = ((SS_G[1][lenr] << MAX) + SS_G[2][lenu] - ((5000 << MAX) + 5000)) & mask;
  inB1[0] = SS_C[(lenr * LENX) + lenu] & mask;
  Prod_H(1, inA1, inB1, outC1, Prodt[th]);
  Skylinet[len_x - 1] = new uint64_t[1];
  Skylinet[len_x - 1][0] = ((5000 << MAX) + 5000 + outC1[0]) & mask;
  delete[] inA1;
  delete[] inB1;
  delete[] outC1;
  // double startk = omp_get_wtime();
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single 
    {
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, len_x, len_y, MAX, mask)
      {
        // double startkk = omp_get_wtime();
        // cout << "k1 " << omp_get_thread_num() << ": ";
        // first row
        uint64_t **Skyline1 = new uint64_t *[len_x];
        // len_x - 1 = lenr - lenl;
        Skyline1[len_x - 1] = new uint64_t[1];
        Skyline1[len_x - 1][0] = Skylinet[len_x - 1][0];
        uint64_t *inA2 = new uint64_t[1];
        uint64_t *inB2 = new uint64_t[1];
        uint64_t *outC2 = new uint64_t[1];
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          // con * point + (1-con) * Skyline[((k1+1)<<MAX)+lenu][0]
          // SS_Print(1, Skyline1[k1 - lenl + 1]);
          inA2[0] = ((SS_G[1][k1] << MAX) + SS_G[2][lenu] - Skyline1[k1 - lenl + 1][0]) & mask;
          inB2[0] = SS_C[(k1 * LENX) + lenu] & mask;
          outC2[0] = 0;
          // Prod->hadamard_product(1, inA2, inB2, outC2, lambda, lambda, lambda, false);
          Prod_H(1, inA2, inB2, outC2, Prodt[th + 1]);
          Skyline1[k1 - lenl] = new uint64_t[1];
          Skyline1[k1 - lenl][0] = (Skyline1[k1 - lenl + 1][0] + outC2[0]) & mask;
          // Skylinet[(k1 << MAX) + lenu] = Skyline1[(k1 << MAX) + lenu];
        }
        delete[] inA2;
        delete[] inB2;
        delete[] outC2;
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          L[k1 - lenl] = 1;
          Skylinet[k1 - lenl] = new uint64_t[1];
          Skylinet[k1 - lenl][0] = Skyline1[k1 - lenl][0];
        }
        for (int k1 = len_x - 1; k1 >= 0; k1--)
        {
          delete[] Skyline1[k1];
        }
        delete[] Skyline1;
        // delete Prod1;
        // delete Otpack1;
        // delete Io1;
        // double endkk = omp_get_wtime();
        // cout << i << "-1" << ((endkk - startkk) * 1000) << endl;
      }
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, len_x, len_y, MAX, mask)
      {
        // double startkkk = omp_get_wtime();
        // cout << "k2 " << omp_get_thread_num() << ": ";
        // first col
        uint64_t **Skyline2 = new uint64_t *[len_y];
        // len_y - 1 = lenu - lend;
        Skyline2[len_y - 1] = new uint64_t[1];
        Skyline2[len_y - 1][0] = Skylinet[len_x - 1][0];
        uint64_t *inA3 = new uint64_t[1];
        uint64_t *inB3 = new uint64_t[1];
        uint64_t *outC3 = new uint64_t[1];
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          // con * point + (1-con) * Skylinet[(lenr<<MAX)+k2+1][0]
          inA3[0] = ((SS_G[1][lenr] << MAX) + SS_G[2][k2] - Skyline2[k2 -lend + 1][0]) & mask;
          inB3[0] = SS_C[(lenr * LENX) + k2] & mask;
          outC3[0] = 0;
          // Prod2->hadamard_product(1, inA3, inB3, outC3, lambda, lambda, lambda, false);
          Prod_H(1, inA3, inB3, outC3, Prodt[th + 2]);
          Skyline2[k2 -lend] = new uint64_t[1];
          Skyline2[k2 -lend][0] = (Skyline2[k2 -lend + 1][0] + outC3[0]) & mask;
          // Skylinet[(lenr << MAX) + k2] = Skyline2[(lenr << MAX) + k2];
        }
        delete[] inA3;
        delete[] inB3;
        delete[] outC3;
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          L[len_x + lenu - 1 - k2] = 1;
          Skylinet[len_x + lenu - 1 - k2] = new uint64_t[1];
          Skylinet[len_x + lenu - 1 - k2][0] = Skyline2[k2 -lend][0];
        }
        for (int k2 = len_y - 1; k2 >= 0; k2--)
        {
          delete[] Skyline2[k2];
        }
        delete[] Skyline2;
        // delete Prod2;
        // delete Otpack2;
        // delete Io2;
        // double endkkk = omp_get_wtime();
        // cout << i << "-2" << ((endkkk - startkkk) * 1000) << endl;
      }
      #pragma omp taskwait 
    }
  } 
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  for (int k2 = lenu; k2 >= lend; k2--)
  {
    SS_L[(lenr * LENX) + k2] = L[len_x + lenu - 1 - k2];
    SS_Skyline[(lenr * LENX) + k2] = new uint64_t[1];
    SS_Skyline[(lenr * LENX) + k2][0] = Skylinet[len_x + lenu - 1 - k2][0];
  }
  for (int k1 = lenr - 1; k1 >= lenl; k1--)
  {
    SS_L[(k1 * LENX) + lenu] = L[k1 - lenl];
    SS_Skyline[(k1 * LENX) + lenu] = new uint64_t[1];
    SS_Skyline[(k1 * LENX) + lenu][0] = Skylinet[k1 - lenl][0];
  }
  for (int k1 = len_x + len_y - 2; k1 >= 0; k1--)
  {
    delete[] Skylinet[k1];
  }
  delete[] Skylinet;
  delete[] L;
}

void SSub_SS_Two_2(int lenl, int lenr, int lend, int lenu, int th)
{
  int len1 = lenr - lenl;
  int len2 = lenu - lend;
  int LENX = SS_G[0][1];
  int len_x = len1 + 1;
  int len_y = len2 + 1;
  int *L = new int[len_x * len_y];
  uint64_t ** Skylinet = new uint64_t*[len_x * len_y];
  // Process
  double starts = omp_get_wtime();
  PRG128 prg;
  int max1 = MAX;
  uint64_t mask1 = mask;
  uint64_t SS_one = 0;
  prg.random_data(&SS_one, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[th]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    uint64_t one = 0;
    Iot[th]->recv_data(&one, sizeof(uint64_t));
    SS_one = (1 - one) & mask1;
  }
  for (int k2 = lenu; k2 >= lend; k2--)
  {
    Skylinet[(len1 * len_y) + k2 - lend] = new uint64_t[1];
    Skylinet[(len1 * len_y) + k2 - lend][0] = SS_Skyline[(lenr * LENX) + k2][0];
    L[(len1 * len_y) + k2 - lend] = SS_L[(lenr * LENX) + k2];
  }
  for (int k1 = lenr - 1; k1 >= lenl; k1--)
  {
    Skylinet[((k1 - lenl) * len_y) + len2] = new uint64_t[1];
    Skylinet[((k1 - lenl) * len_y) + len2][0] = SS_Skyline[(k1 * LENX) + lenu][0];
    L[((k1 - lenl) * len_y) + len2] = SS_L[(k1 * LENX) + lenu];
  }
  int minLen = (len1 > len2) ? len2 : len1;
  // if (minLen > len2)
  //   minLen = len2;
  int len11 = 0;
  int len12 = 0;
  int len13 = 0;
  uint64_t *h;
  int HS_len = 0;
  uint64_t *inA;
  uint64_t *inB;
  uint64_t *outD;
  uint64_t inBt1 = 0;
  for (int i = 1; i <= minLen; i++)
  {
    // cout << endl << (minLen - i) << " " << th << " " << omp_get_thread_num() << " " << omp_get_num_threads() << ": ";
    // sleep(1);
    if((minLen - i)%100 == 0){
      cout << (minLen - i) << " " << th << " : " << endl;
    }
    uint64_t *set;
    HS_len = 0;
    len11 = ((len1 - i + 1) * len_y) + len2 - i;
    len12 = ((len1 - i) * len_y) + len2 - i + 1;
    len13 = ((len1 - i + 1) * len_y) + len2 - i + 1;
    // cout<<"0-----"<<len11<<" "<<len12<<" "<<len13<<endl;
    h = new uint64_t[L[len11] + L[len12] + L[len13]];
    memcpy(h, Skylinet[len11], L[len11] * sizeof(uint64_t));
    memcpy(h + L[len11], Skylinet[len12], L[len12] * sizeof(uint64_t));
    memcpy(h + L[len11] + L[len12], Skylinet[len13], L[len13] * sizeof(uint64_t));
    SkylineGen_1(h, L[len11], L[len12], L[len13], HS_len, set, Auxt[th], Prodt[th], Iot[th]);
    L[len12 - 1] = HS_len;    
    inA = new uint64_t[HS_len];
    inB = new uint64_t[HS_len];
    outD = new uint64_t[HS_len];
    Skylinet[len12 - 1] = new uint64_t[HS_len];
    inA[0] = ((SS_G[1][(lenr - i)] << max1) + SS_G[2][(lenu - i)] - set[0]) & mask1;
    inB[0] = SS_C[((lenr - i) * LENX) + (lenu - i)] & mask1;
    // if (HS_len > 1)
    // {
    //   // memcpy(inA + 1, set+ 1, (HS_len - 1) * sizeof(uint64_t));
    //   // inBt1 = (SS_one - inB[0]) & mask1;
    //   for (int j = 1; j < HS_len; j++)
    //   {
    //     inA[j] = (((5000 << MAX) + 5000) - set[j]) & mask1;
    //     inB[j] = SS_C[((lenr - i) * LENX) + (lenu - i)] & mask1;
    //   }
    // }
    // // Prod_H(HS_len, inA, inB, Skylinet[len12 - 1], Prod);
    // Prod_H(HS_len, inA, inB, outD, Prodt[th]);
    // // Skylinet[len12 - 1] = outD;
    // memcpy(Skylinet[len12 - 1], outD, HS_len * sizeof(uint64_t)); 
    // for (int j = 0; j < HS_len; j++)
    // {
    //   Skylinet[len12 - 1][j] = (Skylinet[len12 - 1][j] + set[j]) & mask1;
    // }
    if (HS_len > 1)
    {
      memcpy(inA + 1, set+ 1, (HS_len - 1) * sizeof(uint64_t));
      inBt1 = (SS_one - inB[0]) & mask1;
      for (int j = 1; j < HS_len; j++)
      {
        inB[j] = inBt1;
      }
    }
    // Prod_H(HS_len, inA, inB, Skylinet[len12 - 1], Prod);
    Prod_H(HS_len, inA, inB, outD, Prodt[th]);
    // Skylinet[len12 - 1] = outD;
    memcpy(Skylinet[len12 - 1], outD, HS_len * sizeof(uint64_t)); 
    Skylinet[len12 - 1][0] = (Skylinet[len12 - 1][0] + set[0]) & mask1;
    delete[] inA;
    delete[] inB;
    delete[] outD;
    delete[] set;
    delete []h;
    // delete Aux;
    // delete Prod;
    // delete Otpa;
    // delete Io;
    // cout << (minLen - i) << " 0-2 " << omp_get_thread_num() << " " << th << " : " << endl;
  // #pragma omp parallel sections
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single 
    {
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, i, th, MAX, mask, LENX, len_y)
      {
        // cout << " K1 " << omp_get_thread_num() << ",";
        int k2 = lenu - i;
        int T2 = lenr - i;
        int T1 = lenl;
        int max2 = MAX;
        int LENXt = LENX;
        uint64_t mask2 = mask;
        uint64_t one2 = SS_one;
	      int leny2 = T2 - T1 + 1;
        uint64_t **skylinet1 = new uint64_t*[2 * leny2];
        int * L1 = new int[2 * leny2];
        L1[T2 - T1] = L[((T2 - lenl) * len_y) + k2 - lend];
        skylinet1[T2 - T1] = new uint64_t[L1[T2 - T1]];
        memcpy(skylinet1[T2 - T1], Skylinet[((T2 - lenl) * len_y) + k2 - lend], L1[T2 - T1] * sizeof(uint64_t));
        for (int k1 = T2; k1 >= T1; k1--)
        {
          L1[k1 - T1 + leny2] = L[((k1 - lenl) * len_y) + k2 - lend + 1];
          skylinet1[k1 - T1 + leny2] = new uint64_t[L1[k1 - T1 + leny2]];
          memcpy(skylinet1[k1 - T1 + leny2], Skylinet[((k1 - lenl) * len_y) + k2 - lend + 1], L1[k1 - T1 +leny2] * sizeof(uint64_t));
        }
        // cout << (minLen - i) << " 1-1 " << omp_get_thread_num() << " " << th + 1 << " : " << endl;
        // int len21 = 0;
        // int len22 = 0;
        // int len23 = 0;
        // int HS_len2 = 0;
        // uint64_t inBt2 = 0;
        for (int k1 = T2 - 1; k1 >= T1; k1--)
        {
          int len21 = L1[k1 - T1 + 1];
          int len22 = L1[k1 - T1 + leny2];
          int len23 = L1[k1 - T1 + 1 + leny2];
          // cout<<"1-----"<<len21<<" "<<len22<<" "<<len23<<endl;
          uint64_t *h2 = new uint64_t[len21 + len22 + len23];
          memcpy(h2, skylinet1[k1 - T1 + 1], len21 * sizeof(uint64_t));
          memcpy(h2 + len21, skylinet1[k1 - T1 + leny2], len22 * sizeof(uint64_t));
          memcpy(h2 + len21 + len22, skylinet1[k1 - T1 + 1 + leny2], len23 * sizeof(uint64_t));
          uint64_t *set2 = nullptr;
          int HS_len2 = 0;
          SkylineGen_1(h2, len21, len22, len23, HS_len2, set2, Auxt[th + 1], Prodt[th + 1], Iot[th + 1]);
          L1[k1 - T1] = HS_len2;
          uint64_t *inA2 = new uint64_t[HS_len2];
          uint64_t *inB2 = new uint64_t[HS_len2];
          uint64_t *outC2 = new uint64_t[HS_len2];
          skylinet1[k1 - T1] = new uint64_t[HS_len2];
          inA2[0] = ((SS_G[1][k1] << max2) + SS_G[2][k2] - set2[0]) & mask2;
          inB2[0] = SS_C[(k1 * LENXt) + k2] & mask2;
          // if (HS_len2 > 1)
          // {
          //   // memcpy(inA2 + 1, set2 + 1, (HS_len2 - 1) * sizeof(uint64_t));
          //   // uint64_t inBt2 = (one2 - inB2[0]) & mask2;
          //   for (int j = 1; j < HS_len2; j++)
          //   {
          //     inA2[j] = (((5000 << MAX) + 5000) - set2[j]) & mask2;
          //     inB2[j] = SS_C[(k1 * LENXt) + k2] & mask2;
          //   }
          // }
          // // Prod_H(HS_len2, inA2, inB2, skylinet1[k1 - T1], Prodt1);
          // Prod_H(HS_len2, inA2, inB2, outC2, Prodt[th + 1]);
          // memcpy(skylinet1[k1 - T1], outC2, HS_len2 * sizeof(uint64_t));
          // for (int j = 0; j < HS_len2; j++)
          // {
          //   skylinet1[k1 - T1][j] = (skylinet1[k1 - T1][j] + set2[j]) & mask2;
          // }
          if (HS_len2 > 1)
          {
            memcpy(inA2 + 1, set2 + 1, (HS_len2 - 1) * sizeof(uint64_t));
            uint64_t inBt2 = (one2 - inB2[0]) & mask2;
            for (int j = 1; j < HS_len2; j++)
            {
              inB2[j] = inBt2;
            }
          }
          // Prod_H(HS_len2, inA2, inB2, skylinet1[k1 - T1], Prodt1);
          Prod_H(HS_len2, inA2, inB2, outC2, Prodt[th + 1]);
          memcpy(skylinet1[k1 - T1], outC2, HS_len2 * sizeof(uint64_t));
          skylinet1[k1 - T1][0] = (skylinet1[k1 - T1][0] + set2[0]) & mask2;
          delete[] inA2;
          delete[] inB2;
          delete[] outC2;
          delete[] set2;
          delete[] h2;
        }
        // #pragma omp critical
        // {
          for (int k1 = T2 - 1; k1 >= T1; k1--)
          {
            // L[(k1 << max2) + k2] = L1[(k1 << max2) + k2];
            L[((k1 - lenl) * len_y) + k2 - lend] = L1[k1 - T1];
            Skylinet[((k1 - lenl) * len_y) + k2 - lend] = new uint64_t[L1[k1 - T1]];
            memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], skylinet1[k1 - T1], L1[k1 - T1] * sizeof(uint64_t));
          }
          for (int k1 = 2 * leny2 - 1; k1 >= 0; k1--)
          {
            delete[] skylinet1[k1];
          }
        // }
          delete[] skylinet1;
          delete[] L1;
          // delete Auxt1;
          // delete Prodt1;
          // delete otpackt1;
          // delete Iot1;
          // delete[] ss_C1;
          // delete[] ss_G1;
        // cout << (minLen - i) << " 1-2 " << omp_get_thread_num() << " " << th + 1 << " : " << endl;
      }
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, i, th, MAX, mask, LENX, len_y)
      {
        // cout << " K2 " << omp_get_thread_num();
        int k1 = lenr - i;
        int T2 = lenu - i;
        int T1 = lend;
        int max3 = MAX;
        uint64_t mask3 = mask;
        uint64_t one3 = SS_one;
        int LENXt = LENX;
	      int leny3 = T2 - T1 + 1;
        uint64_t ** skylinet2 = new uint64_t *[2 * leny3];
        int * L2 = new int[2 * leny3];
        L2[T2 - T1] = L[((k1 - lenl) * len_y) + T2 - lend];
        skylinet2[T2 - T1] = new uint64_t[L2[T2 - T1]];
        memcpy(skylinet2[T2 - T1], Skylinet[((k1 - lenl) * len_y) + T2 - lend], L2[T2 - T1] * sizeof(uint64_t));
        // skylinet2[(k1 << max3) + T2] = Skylinet[(k1 << max3) + T2];
        // skylinet2[((lenr - i) << max3) + T2] = new uint32_t[L[((lenr - i) << max3) + T2]];
        // copy(Skylinet[((lenr - i) << max3) + T2], Skylinet[((lenr - i) << max3) + T2] + L[((lenr - i) << max3) + T2], skylinet2[((lenr - i) << max3) + T2]);
        for (int k3 = T2; k3 >= T1; k3--)
        {
          // L2[((k1 + 1) << max3) + k3] = L[((k1 + 1) << max3) + k3];
          L2[k3 - T1 + leny3] = L[((k1 - lenl + 1) * len_y) + k3 - lend];
          skylinet2[k3 - T1 + leny3] = new uint64_t[L2[k3 - T1 + leny3]];
          memcpy(skylinet2[k3 - T1 + leny3], Skylinet[((k1 - lenl + 1) * len_y) + k3 - lend], L2[k3 - T1 + leny3] * sizeof(uint64_t));
        }
        // cout << (minLen - i) <<" 2-1 "<<omp_get_thread_num()<<" " << th+1 << " : " << endl;
        // int len31 = 0;
        // int len32 = 0;
        // int len33 = 0;
        // int HS_len3 = 0;
        // uint64_t inBt3 = 0;
        for (int k3 = T2 - 1; k3 >= T1; k3--)
        {
          // int len31 = L2[((k1 + 1) << max3) + k3];
          // int len32 = L2[(k1 << max3) + k3 + 1];
          // int len33 = L2[((k1 + 1) << max3) + k3 + 1];
          int len31 = L2[k3 - T1 + leny3];
          int len32 = L2[k3 - T1 + 1];
          int len33 = L2[k3 - T1 + 1 + leny3];
          // cout<<"2-----"<<len31<<" "<<len32<<" "<<len33<<endl;
          uint64_t *h3 = new uint64_t[len31 + len32 + len33];
          memcpy(h3, skylinet2[k3 - T1 + leny3], len31 * sizeof(uint64_t));
          memcpy(h3 + len31, skylinet2[k3 - T1 + 1], len32 * sizeof(uint64_t));
          memcpy(h3 + len31 + len32, skylinet2[k3 - T1 + 1 + leny3], len33 * sizeof(uint64_t));
          uint64_t *set3 = nullptr;
          int HS_len3 = 0;
          SkylineGen_1(h3, len31, len32, len33, HS_len3, set3, Auxt[th + 2], Prodt[th + 2], Iot[th + 2]);
          L2[k3 - T1] = HS_len3;
          uint64_t *inA3 = new uint64_t[HS_len3];
          uint64_t *inB3 = new uint64_t[HS_len3];
          uint64_t *outC3 = new uint64_t[HS_len3];
          skylinet2[k3 - T1] = new uint64_t[HS_len3];
          inA3[0] = ((SS_G[1][k1] << max3) + SS_G[2][k3] - set3[0]) & mask3;
          inB3[0] = SS_C[(k1 * LENXt) + k3] & mask3; 
          // if (HS_len3 > 1)
          // {
          //   // memcpy(inA3 + 1, set3+ 1, (HS_len3 - 1) * sizeof(uint64_t));
          //   // uint64_t inBt3 = (one3 - inB3[0]) & mask3;
          //   for (int j = 1; j < HS_len3; j++)
          //   {
          //     inA3[j] = (((5000 << MAX) + 5000) - set3[j]) & mask3;
          //     inB3[j] = SS_C[(k1 * LENXt) + k3] & mask3; 
          //   }
          // }
          // // Prod_H(HS_len3, inA3, inB3, skylinet2[k3 - T1], Prodt2);
          // Prod_H(HS_len3, inA3, inB3, outC3, Prodt[th + 2]);
          // memcpy(skylinet2[k3 - T1], outC3, HS_len3 * sizeof(uint64_t));
          // for (int j = 0; j < HS_len3; j++)
          // {
          //   skylinet2[k3 - T1][j] = (skylinet2[k3 - T1][j] + set3[j]) & mask3;
          // }
          if (HS_len3 > 1)
          {
            memcpy(inA3 + 1, set3+ 1, (HS_len3 - 1) * sizeof(uint64_t));
            uint64_t inBt3 = (one3 - inB3[0]) & mask3;
            for (int j = 1; j < HS_len3; j++)
            {
              inB3[j] = inBt3;
            }
          }
          // Prod_H(HS_len3, inA3, inB3, skylinet2[k3 - T1], Prodt2);
          Prod_H(HS_len3, inA3, inB3, outC3, Prodt[th + 2]);
          memcpy(skylinet2[k3 - T1], outC3, HS_len3 * sizeof(uint64_t));
          skylinet2[k3 - T1][0] = (skylinet2[k3 - T1][0] + set3[0]) & mask3;
          delete[] inA3;
          delete[] inB3;
          delete[] outC3;
          delete[] set3;
          delete[] h3;
        }
        // #pragma omp critical
        // {
          for (int k3 = T2 - 1; k3 >= T1; k3--)
          {
            // L[(k1 << max3) + k3] = L2[(k1 << max3) + k3];
            L[((k1 - lenl) * len_y) + k3 - lend] = L2[k3 - T1];
            Skylinet[((k1 - lenl) * len_y) + k3 - lend] = new uint64_t[L2[k3 - T1]];
            memcpy(Skylinet[((k1 - lenl) * len_y) + k3 - lend], skylinet2[k3 - T1], L2[k3 - T1] * sizeof(uint64_t));
            // Skylinet[(k1 << max3) + k3] = skylinet2[(k1 << max3) + k3];
          }
          for (int k3 = 2 * leny3 - 1; k3 >= 0; k3--)
          {
            delete[] skylinet2[k3];
          }
        // }
          delete[] skylinet2;
          delete[] L2;
          // delete Auxt2;
          // delete Prodt2;
          // delete otpackt2;
          // delete Iot2;
          // delete[] ss_C2;
          // delete[] ss_G2;
        // cout << (minLen - i) <<" 2-2 "<<omp_get_thread_num()<<" " << th+2 << " : " << endl;
      }
      #pragma omp taskwait 
    }   
  }
  // cout << (minLen - i) <<" 3-2 "<<omp_get_thread_num()<<" " << th << " *******"<<endl;
  }
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  cout << "[(" << lenl << "," << lend << "),(" << lenr << "," << lenu << ")]" << ss_t << " s" << endl;
  // #pragma omp parallel for
  for (int k2 = lenu - 1; k2 >= lend; k2--)
  {
    for (int k1 = lenr - 1; k1 >= lenl; k1--)
    {
      SS_L[(k1 * LENX) + k2] = L[((k1 - lenl) * len_y) + k2 - lend];
      SS_Skyline[(k1 * LENX) + k2] = new uint64_t[L[((k1 - lenl) * len_y) + k2 - lend]];
      memcpy(SS_Skyline[(k1 * LENX) + k2], Skylinet[((k1 - lenl) * len_y) + k2 - lend], L[((k1 - lenl) * len_y) + k2 - lend] * sizeof(uint64_t));
    }
  }
  for (int k1 = len_x * len_y - 1; k1 >= 0; k1--)
  {
      delete[] Skylinet[k1];
  }
  delete[] Skylinet;
  delete[] L;
}

// Constrain Sub

void SSub_Con(int lenl, int lenr, int lendt, int lenut, int th)
{
  // int T2 = lenr;
  // int T1 = lenl;
  int LENXt = SS_G[0][1];
  uint64_t maskt = mask;
  int len_xt = lenr - lenl + 1;
  int len_yt = lenut - lendt + 1;
  uint32_t* Cont = new uint32_t[len_xt * len_yt];
  // Process
  double starts = omp_get_wtime();
  uint64_t *in1 = new uint64_t[len_xt];
  uint64_t *out1 = new uint64_t[len_xt];
  for (int k2 = lenut; k2 >= lendt; k2--)
  {
    if (party == ALICE)
    {
      for (int k1 = lenr; k1 >= lenl; k1--)
      {
        in1[k1 - lenl] = (0 - SS_C[(k1 * LENXt) + k2] - SS_C[((k1 + 1) * LENXt) + k2] - SS_C[(k1 * LENXt) + k2 + 1]) & maskt;
      }
    }
    else
    {
      for (int k1 = lenr; k1 >= lenl; k1--)
      {
        in1[k1 - lenl] = (SS_C[(k1 * LENXt) + k2] + SS_C[((k1 + 1) * LENXt) + k2] + SS_C[(k1 * LENXt) + k2 + 1]) & maskt;
      }
    } 
    equality(len_xt, in1, out1, Auxt[th]);
    for (int k1 = lenr; k1 >= lenl; k1--)
    {
      Cont[((k1 - lenl) * len_yt) + k2 - lendt] = out1[k1 - lenl] & maskt;
    }
  }
  delete[] in1;
  delete[] out1;
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  // cout << "[(" << lenl << "," << lendt << "),(" << lenr << "," << lenut << ")]" << ss_t << " s" << endl;
  // #pragma omp parallel for
  for (int k2 = lenut; k2 >= lendt; k2--)
  {
    for (int k1 = lenr; k1 >= lenl; k1--)
    {
      SS_Con[(k1 * LENXt) + k2] = Cont[((k1 - lenl) * len_yt) + k2 - lendt];
    }
  }
  delete[] Cont;
}

void CSub_SS_Two_1(int lenl, int lenr, int lend, int lenu)
{ 
  int len1 = lenr - lenl;
  int len2 = lenu - lend;
  int len_x = len1 + 1;
  int len_y = len2 + 1;
  uint64_t ** Skylinet = new uint64_t*[len_x * len_y];
  int LENX = SS_G[0][1];
  int LENY = SS_G[0][0];   
  SS_Con = new uint32_t[LENY * LENX];
  double starts = omp_get_wtime();
  // PRG128 prg;
  uint64_t SS_one = 0;
  prg.random_data(&SS_one, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    uint64_t one = 0;
    Iot[0]->recv_data(&one, sizeof(uint64_t));
    SS_one = (1 - one) & mask;
  }
  int tmpLen = SS_L[(lenr * LENX) + lenu];
  // #pragma omp parallel
  // for (int k1 = lenr; k1 >= lenl; k1--)
  // {
  //   for (int k2 = lenu; k2 >= lend; k2--)
  //   {
  //     Skyline[(k1 << MAX) + k2] = new uint32_t[tmpLen];
  //   }
  // }
  // vector<uint64_t> XX(tmpLen);
  // copy(SS_Skyline[(lenr << MAX) + lenu], SS_Skyline[(lenr << MAX) + lenu] + tmpLen, XX.begin());
  // Skyline_Print(XX);
  for (int k1 = len_x * len_y - 1; k1 >= 0; k1--)
  {
    Skylinet[k1] = new uint64_t[tmpLen];
  }
  memcpy(Skylinet[((lenr - lenl) * len_y) + lenu - lend], SS_Skyline[(lenr * LENX) + lenu], tmpLen * sizeof(uint64_t));
  // Skylinet[(lenr << MAX) + lenu] = SS_Skyline[(lenr * LENX) + lenu];
  // copy(SS_Skyline[(lenr << MAX) + lenu], SS_Skyline[(lenr << MAX) + lenu] + tmpLen, Skylinet[(lenr << MAX) + lenu]);
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single 
    {
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, len_y)
      {
        // first row
        int LENXt = LENX;
		    uint64_t mask2 = mask;
		    uint64_t one2 = SS_one;
        int tmpLent = tmpLen;
        uint64_t **Skyline1 = new uint64_t* [len_x];
        for (int k1 = len_x - 1; k1 >= 0; k1--)
        {
          Skyline1[k1] = new uint64_t[tmpLent];
        }
        memcpy(Skyline1[lenr - lenl], Skylinet[((lenr - lenl) * len_y) + lenu - lend], tmpLent * sizeof(uint64_t));
        // uint64_t *inA = new uint64_t[tmpLent];
        uint64_t *inB = new uint64_t[tmpLent];
        uint64_t *outD = new uint64_t[tmpLent];
        int k2 = lenu;
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          // copy(Skyline1[((k1 + 1) << MAX) + k2], Skyline1[((k1 + 1) << MAX) + k2] + tmpLent, inA);
          uint64_t inBt = (one2 - SS_C[(k1 * LENXt) + k2]) & mask2;
          for (int i = 0; i < tmpLent; i++)
          {
            inB[i] = inBt;
          }
          //(1-con) * SSone
          Prod_H(tmpLent, Skyline1[k1 - lenl + 1], inB, outD, Prodt[0]);
          memcpy(Skyline1[k1 - lenl], outD, tmpLent * sizeof(uint64_t));
        }
        // delete[] inA;
        delete[] inB;
        delete[] outD;
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skyline1[k1 - lenl], tmpLent * sizeof(uint64_t));
        }
        for (int k1 = len_x - 1; k1 >= 0; k1--)
        {
          delete[] Skyline1[k1];
        }
        delete[] Skyline1;
      }
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, len_y)
      {
        // first col
		    int LENXt = LENX;
		    uint64_t mask3 = mask;
		    uint64_t one3 = SS_one;
        int tmpLent = tmpLen;
        uint64_t **Skyline2 = new uint64_t* [len_y];
        for (int k2 = len_y - 1; k2 >= 0; k2--)
        {
          Skyline2[k2] = new uint64_t[tmpLent];
        }
        memcpy(Skyline2[lenu - lend], Skylinet[((lenr - lenl) * len_y) + lenu - lend], tmpLent * sizeof(uint64_t));
        // uint64_t *inA = new uint64_t[tmpLent];
        uint64_t *inB = new uint64_t[tmpLent];
        uint64_t *outD = new uint64_t[tmpLent];
        int k1 = lenr;
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          // copy(Skyline2[(k1 << MAX) + k2 + 1], Skyline2[(k1 << MAX) + k2 + 1] + tmpLent, inA);
          uint64_t inBt = (one3 - SS_C[(k1 * LENXt) + k2]) & mask3;
          for (int i = 0; i < tmpLent; i++)
          {
            inB[i] = inBt;
          }
          //(1-con) * SSone
          Prod_H(tmpLent, Skyline2[k2 -lend + 1], inB, outD, Prodt[1]);
          memcpy(Skyline2[k2 -lend], outD, tmpLent * sizeof(uint64_t));
        }
        // delete[] inA;
        delete[] inB;
        delete[] outD;
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skyline2[k2 -lend], tmpLent * sizeof(uint64_t));
        }
        for (int k2 = len_y - 1; k2 >= 0; k2--)
        {
          delete[] Skyline2[k2];
        }

        delete[] Skyline2;
      }
      #pragma omp taskwait
    }    
  }
  // #pragma omp parallel
  // {
  //   #pragma omp single 
  //   {
  //     #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_C)
  //     {
  //       SSub_Con((lenr + 1) / 2, lenr - 1, (lenu + 1) / 2, lenu - 1, 0);
  //     }
  //     #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_C)
  //     {
  //       SSub_Con((lenr + 1) / 2, lenr - 1, lend, (lenu - 1) / 2, 2);
  //     }
  //     #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_C)
  //     {
  //       SSub_Con(lenl, (lenr - 1) / 2, (lenu + 1) / 2, lenu - 1, 4);
  //     }
  //     #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_C)
  //     {
  //       SSub_Con(lenl, (lenr - 1) / 2, lend, (lenu - 1) / 2, 6);
  //     }
  //     #pragma omp taskwait
  //   }
  // }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(lenr, lenl, lenu, lend, THs)
        {
          int lendt = (((lenu - 1 - lend) * itr)/ THs) + 1 +lend;
          int lenut = ((lenu - 1 - lend) * (itr + 1))/ THs + lend;
          if(itr == 0){
            lendt = lend;
          }
          SSub_Con(lenl, lenr - 1, lendt, lenut, itr);
        }
      }
      #pragma omp taskwait
    }  
  }
  for (int k2 = lenu - 1 - lend; k2 >= 0; k2--)
  {
    cout<<G[2][k2]<<"\t";
    for (int k1 = 0; k1 <= LENX - 1; k1++)
    {
      uint64_t *x = new uint64_t[1];
      x[0] = SS_Con[(k1 * LENX) + k2];
      SS_Print(1, x);
      cout << "\t";
    }
    cout << endl;
  }
  cout<<"\t";
  for (int k1 = 0; k1 <= LENX - 1; k1++)
  {
    cout<<G[1][k1]<<"\t";
  }
  cout<<endl;
  int minLen = (len1 > len2) ? len2 : len1;
  for (int i = 1; i <= minLen; i++)
  {
    cout << (minLen - i) << " : " << endl;
    #pragma omp parallel num_threads(THs)
    {
      #pragma omp single 
      {
        for (int itr = 0; itr < THs; itr++)
        {
          #pragma omp task firstprivate(lenr, lenl, lenu, lend, THs, len_y)
          {
            int LENXt = LENX;
            uint64_t maskt = mask;
            int tmpLent = tmpLen;      
            uint64_t *inB = new uint64_t[tmpLent];
            uint64_t *outD = new uint64_t[tmpLent];
            if(itr < (THs / 2)){
              int k2 = lenu - i;
              int T1 = (((lenr - i - lenl) * itr)/ (THs / 2)) + 1 + lenl;
              int T2 = (((lenr - i - lenl) * (itr + 1))/ (THs / 2)) + lenl;
              if(itr == 0){
                T1 = lenl;
              }
              // int T2 = lenr - i;
              // int T1 = lenl;
              if(T2 - T1 + 1 > 0){
                uint64_t ** Skylinet1 = new uint64_t* [T2 - T1 + 1];
                for (int k1 = T2; k1 >= T1; k1--)
                {
                  Skylinet1[k1 - T1] = new uint64_t[tmpLent];
                  for (int j = 0; j < tmpLent; j++)
                  {
                    inB[j] = SS_Con[(k1 * LENXt) + k2] & maskt;
                  }
                  //Get value from up grid, then free copy about Skylinet1.
                  Prod_H(tmpLent, Skylinet[((k1 - lenl) * len_y) + k2 - lend + 1], inB, outD, Prodt[itr]);
                  memcpy(Skylinet1[k1 - T1], outD, tmpLent * sizeof(uint64_t));
                }
                for (int k1 = T2; k1 >= T1; k1--)
                {
                  memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skylinet1[k1 - T1], tmpLent * sizeof(uint64_t));
                  delete[] Skylinet1[k1 - T1];
                }
                delete[] Skylinet1;
              }             
            }
            else{
              int k1 = lenr - i;
              int T1 = (((lenu - i - 1 - lend) * (itr - (THs / 2)))/ (THs / 2)) + 1 + lend;
              int T2 = (((lenu - i - 1 - lend) * (itr + 1 - (THs / 2)))/ (THs / 2)) + lend;
              if(itr == (THs / 2)){
                T1 = lend;
              }
              // int T2 = lenu - i - 1;
              // int T1 = lend;
              if(T2 - T1 + 1 > 0){
                uint64_t ** Skylinet2 = new uint64_t* [T2 - T1 + 1];
                for (int k3 = T2; k3 >= T1; k3--)
                {
                  Skylinet2[k3 - T1] = new uint64_t[tmpLent];
                  for (int j = 0; j < tmpLent; j++)
                  {
                    inB[j] = SS_Con[(k1 * LENXt) + k3] & maskt;
                  }
                  //Get value from right grid, then free copy about Skylinet2.
                  Prod_H(tmpLent,  Skylinet[((k1 - lenl + 1) * len_y) + k3 - lend], inB, outD, Prodt[itr]);          
                  memcpy(Skylinet2[k3 - T1], outD, tmpLent * sizeof(uint64_t));
                }
                for (int k3 = T2; k3 >= T1; k3--)
                {
                  memcpy(Skylinet[((k1 - lenl) * len_y) + k3 - lend], Skylinet2[k3 - T1], tmpLent * sizeof(uint64_t));
                  delete[] Skylinet2[k3 - T1];
                }
                delete[] Skylinet2;
              }             
            }
            delete[] inB;
            delete[] outD;
          }
        }
        #pragma omp taskwait
      }    
    }
  }
    for (int k2 = lenu - 1 - lend; k2 >= 0; k2--)
  {
    cout<<G[2][k2]<<"\t";
    for (int k1 = 0; k1 <= LENX - 1; k1++)
    {
      SS_Print(tmpLen,  Skylinet[((k1 - lenl) * len_y) + k2 - lend]);
      cout << "\t";
    }
    cout << endl;
  }
  cout<<"\t";
  for (int k1 = 0; k1 <= LENX - 1; k1++)
  {
    cout<<G[1][k1]<<"\t";
  }
  cout<<endl;
  // #pragma omp parallel for
  for (int k2 = lenu - 1; k2 >= lend; k2--)
  {
    for (int k1 = lenr - 1; k1 >= lenl; k1--)
    {
      uint64_t *tmpSkyline = new uint64_t[SS_L[(k1 * LENX) + k2]];
      memcpy(tmpSkyline, SS_Skyline[(k1 * LENX) + k2], SS_L[(k1 * LENX) + k2] * sizeof(uint64_t));
      delete[] SS_Skyline[(k1 * LENX) + k2];
      SS_Skyline[(k1 * LENX) + k2] = new uint64_t[tmpLen + SS_L[(k1 * LENX) + k2]];
      memcpy(SS_Skyline[(k1 * LENX) + k2], Skylinet[((k1 - lenl) * len_y) + k2 - lend], tmpLen * sizeof(uint64_t));
      memcpy(SS_Skyline[(k1 * LENX) + k2] + tmpLen, tmpSkyline, SS_L[(k1 * LENX) + k2] * sizeof(uint64_t));
      SS_L[(k1 * LENX) + k2] = tmpLen + SS_L[(k1 * LENX) + k2];
      delete[] tmpSkyline;
      // delete[] Skylinet[((k1 - lenl) * len_y) + k2 - lend];
    }
  }
  for (int k1 = len_x * len_y - 1; k1 >= 0; k1--)
  {
    delete[] Skylinet[k1];
  }
  delete[] Skylinet;
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  cout << "[(" << lenl << "," << lend << "),(" << (lenr - 1) << "," << (lenu - 1) << ")]" << endl;
  cout << "ConstrainSub_SS:"<< ss_t << " s" << endl;
}

void CSub_SS_Two(int lenl, int lenr, int lend, int lenu)
{ 
  int len1 = lenr - lenl;
  int len2 = lenu - lend;
  int len_x = len1 + 1;
  int len_y = len2 + 1;
  uint64_t ** Skylinet = new uint64_t*[len_x * len_y];
  int LENX = SS_G[0][1];
  int LENY = SS_G[0][0];   
  SS_Con = new uint32_t[LENY * LENX];
  double starts = omp_get_wtime();
  // PRG128 prg;
  uint64_t SS_one = 0;
  prg.random_data(&SS_one, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
  }
  else
  {
    uint64_t one = 0;
    Iot[0]->recv_data(&one, sizeof(uint64_t));
    SS_one = (1 - one) & mask;
  }
  int tmpLen = SS_L[((lenr + 1) * LENX) + (lenu + 1)];
  for (int k1 = len_x * len_y - 1; k1 >= 0; k1--)
  {
    Skylinet[k1] = new uint64_t[tmpLen];
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {
        #pragma omp task firstprivate(lenr, lenl, lenu, lend, THs)
        {
          int lendt = (((lenu - lend) * itr)/ THs) + 1 +lend;
          int lenut = (((lenu - lend) * (itr + 1))/ THs) + lend;
          if(itr == 0){
            lendt = lend;
          }
          SSub_Con(lenl, lenr, lendt, lenut, itr);
        }
      }
      #pragma omp taskwait
    }  
  }
  //  uint32_t *SS_C0 = new uint32_t[LENX * LENY];
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(SS_C, LENX * LENY * sizeof(uint32_t));
  //   Iot[0]->recv_data(SS_C0, LENX * LENY * sizeof(uint32_t));
  // }
  // else
  // {
  //   Iot[0]->recv_data(SS_C0, LENX * LENY * sizeof(uint32_t));
  //   for (int i = 0; i < LENX * LENY; i++)
  //   {
  //     SS_C0[i] = (SS_C0[i] + SS_C[i]) & mask;
  //   }
  //   Iot[0]->send_data(SS_C0, LENX * LENY * sizeof(uint32_t));
  // } 
  // for (int k2 = LENY - 1; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= LENX - 1; k1++)
  //   {
  //     cout<<SS_C0[k1 * LENY + k2]<<"\t";
  //   }
  //   cout<<endl;
  // }
  // cout<<"\t";
  // for (int k1 = 0; k1 <= LENX - 1; k1++)
  // {
  //   cout<<G[1][k1]<<"\t";
  // }
  // cout<<endl;
  // for (int k2 = lenu -lend; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= lenr - lenl; k1++)
  //   {
  //     uint64_t *x = new uint64_t[1];
  //     x[0] = SS_Con[(k1 * LENX) + k2];
  //     SS_Print(1, x);
  //     cout << "\t";
  //   }
  //   cout << endl;
  // }
  // cout<<"\t";
  // for (int k1 = 0; k1 <= LENX - 1; k1++)
  // {
  //   cout<<G[1][k1]<<"\t";
  // }
  // cout<<endl;
  uint64_t *in1 = new uint64_t[tmpLen];
  uint64_t *in2 = new uint64_t[tmpLen];
  uint64_t *out1 = new uint64_t[tmpLen];
  for (int j = 0; j < tmpLen; j++)
  {
    in1[j] = SS_Con[((lenr - lenl) * LENX) + lenu - lend] & mask;
    in2[j] = (SS_Skyline[((lenr + 1) * LENX) + (lenu + 1)][j] - ((5000 << MAX) + 5000)) & mask;
  }
  //Get value from up right grid.
  Prod_H(tmpLen, in1, in2, out1, Prodt[0]);
  for (int j = 0; j < tmpLen; j++)
  {
    out1[j] = (out1[j] + ((5000 << MAX) + 5000)) & mask;
  }
  memcpy(Skylinet[((lenr - lenl) * len_y) + lenu - lend], out1, tmpLen * sizeof(uint64_t));
  delete[] in1;
  delete[] in2;
  delete[] out1;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single 
    {
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, len_y)
      {
        // first row
        int LENXt = LENX;
		    uint64_t mask2 = mask;
		    uint64_t one2 = SS_one;
        int tmpLent = tmpLen;
        uint64_t **Skyline1 = new uint64_t* [len_x];
        for (int k1 = len_x - 1; k1 >= 0; k1--)
        {
          Skyline1[k1] = new uint64_t[tmpLent];
        }
        memcpy(Skyline1[lenr - lenl], Skylinet[((lenr - lenl) * len_y) + lenu - lend], tmpLent * sizeof(uint64_t));
        uint64_t *inA = new uint64_t[tmpLent];
        uint64_t *inB = new uint64_t[tmpLent];
        uint64_t *outD = new uint64_t[tmpLent];
        int k2 = lenu;
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          // copy(Skyline1[((k1 + 1) << MAX) + k2], Skyline1[((k1 + 1) << MAX) + k2] + tmpLent, inA);
          uint64_t inBt = SS_C[(k1 * LENXt) + k2] & mask2;
          for (int i = 0; i < tmpLent; i++)
          {
            inA[i] = (((5000 << MAX) + 5000) - Skyline1[k1 - lenl + 1][i]) & mask2;
            inB[i] = inBt;
          }
          //(1-con) * SSone
          Prod_H(tmpLent, inA, inB, outD, Prodt[1]);
          memcpy(Skyline1[k1 - lenl], outD, tmpLent * sizeof(uint64_t));
          for (int i = 0; i < tmpLent; i++)
          {
            Skyline1[k1 - lenl][i] = (Skyline1[k1 - lenl][i] + Skyline1[k1 - lenl + 1][i]) & mask2;
          }
        }
        delete[] inA;
        delete[] inB;
        delete[] outD;
        for (int k1 = lenr - 1; k1 >= lenl; k1--)
        {
          memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skyline1[k1 - lenl], tmpLent * sizeof(uint64_t));
        }
        for (int k1 = len_x - 1; k1 >= 0; k1--)
        {
          delete[] Skyline1[k1];
        }
        delete[] Skyline1;
      }
      #pragma omp task firstprivate(lenr, lenl, lenu, lend, SS_one, len_y)
      {
        // first col
		    int LENXt = LENX;
		    uint64_t mask3 = mask;
		    uint64_t one3 = SS_one;
        int tmpLent = tmpLen;
        uint64_t **Skyline2 = new uint64_t* [len_y];
        for (int k2 = len_y - 1; k2 >= 0; k2--)
        {
          Skyline2[k2] = new uint64_t[tmpLent];
        }
        memcpy(Skyline2[lenu - lend], Skylinet[((lenr - lenl) * len_y) + lenu - lend], tmpLent * sizeof(uint64_t));
        uint64_t *inA = new uint64_t[tmpLent];
        uint64_t *inB = new uint64_t[tmpLent];
        uint64_t *outD = new uint64_t[tmpLent];
        int k1 = lenr;
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          // copy(Skyline2[(k1 << MAX) + k2 + 1], Skyline2[(k1 << MAX) + k2 + 1] + tmpLent, inA);
          uint64_t inBt = SS_C[(k1 * LENXt) + k2] & mask3;
          for (int i = 0; i < tmpLent; i++)
          {
            inA[i] = (((5000 << MAX) + 5000) - Skyline2[k2 -lend + 1][i]) & mask3;
            inB[i] = inBt;
          }
          //(1-con) * SSone
          Prod_H(tmpLent, inA, inB, outD, Prodt[2]);
          memcpy(Skyline2[k2 -lend], outD, tmpLent * sizeof(uint64_t));
          for (int i = 0; i < tmpLent; i++)
          {
            Skyline2[k2 -lend][i] = (Skyline2[k2 -lend][i] + Skyline2[k2 -lend + 1][i]) & mask3;
          }
        }
        delete[] inA;
        delete[] inB;
        delete[] outD;
        for (int k2 = lenu - 1; k2 >= lend; k2--)
        {
          memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skyline2[k2 -lend], tmpLent * sizeof(uint64_t));
        }
        for (int k2 = len_y - 1; k2 >= 0; k2--)
        {
          delete[] Skyline2[k2];
        }
        delete[] Skyline2;
      }
      #pragma omp taskwait
    }    
  }
    int minLen = (len1 > len2) ? len2 : len1;
  for (int i = 1; i <= minLen; i++)
  {
    if((minLen - i)%100 == 0){
      cout << (minLen - i) << " : " << endl;
    }
    #pragma omp parallel num_threads(THs)
    {
      #pragma omp single 
      {
        for (int itr = 0; itr < THs; itr++)
        {
          #pragma omp task firstprivate(lenr, lenl, lenu, lend, THs, len_y)
          {
            int LENXt = LENX;
            uint64_t maskt = mask;
            int tmpLent = tmpLen;      
            uint64_t *inA = new uint64_t[tmpLent];
            uint64_t *inB = new uint64_t[tmpLent];
            uint64_t *outD = new uint64_t[tmpLent];
            if(itr < (THs / 2)){
              int k2 = lenu - i;
              int T1 = (((lenr - i - lenl) * itr)/ (THs / 2)) + 1 + lenl;
              int T2 = (((lenr - i - lenl) * (itr + 1))/ (THs / 2)) + lenl;
              if(itr == 0){
                T1 = lenl;
              }
              // int T2 = lenr - i;
              // int T1 = lenl;
              if(T2 - T1 + 1 > 0){
                uint64_t ** Skylinet1 = new uint64_t* [T2 - T1 + 1];
                for (int k1 = T2; k1 >= T1; k1--)
                {
                  Skylinet1[k1 - T1] = new uint64_t[tmpLent];
                  for (int j = 0; j < tmpLent; j++)
                  {
                    inA[j] = SS_Con[(k1 * LENXt) + k2] & maskt;
                    inB[j] = (Skylinet[((k1 - lenl) * len_y) + k2 - lend + 1][j] - ((5000 << MAX) + 5000)) & maskt;
                  }
                  //Get value from up grid, then free copy about Skylinet1.
                  Prod_H(tmpLent, inA, inB, outD, Prodt[itr]);
                  for (int j = 0; j < tmpLent; j++)
                  {
                    outD[j] = (outD[j] + ((5000 << MAX) + 5000)) & maskt;
                  }
                  memcpy(Skylinet1[k1 - T1], outD, tmpLent * sizeof(uint64_t));
                }
                for (int k1 = T2; k1 >= T1; k1--)
                {
                  memcpy(Skylinet[((k1 - lenl) * len_y) + k2 - lend], Skylinet1[k1 - T1], tmpLent * sizeof(uint64_t));
                  delete[] Skylinet1[k1 - T1];
                }
                delete[] Skylinet1;
              }             
            }
            else{
              int k1 = lenr - i;
              int T1 = (((lenu - i - 1 - lend) * (itr - (THs / 2)))/ (THs / 2)) + 1 + lend;
              int T2 = (((lenu - i - 1 - lend) * (itr + 1 - (THs / 2)))/ (THs / 2)) + lend;
              if(itr == (THs / 2)){
                T1 = lend;
              }
              // int T2 = lenu - i - 1;
              // int T1 = lend;
              if(T2 - T1 + 1 > 0){
                uint64_t ** Skylinet2 = new uint64_t* [T2 - T1 + 1];
                for (int k3 = T2; k3 >= T1; k3--)
                {
                  Skylinet2[k3 - T1] = new uint64_t[tmpLent];
                  for (int j = 0; j < tmpLent; j++)
                  {
                    inA[j] = SS_Con[(k1 * LENXt) + k3] & maskt;
                    inB[j] = (Skylinet[((k1 - lenl + 1) * len_y) + k3 - lend][j] - ((5000 << MAX) + 5000)) & maskt;
                  }
                  //Get value from right grid, then free copy about Skylinet2.
                  Prod_H(tmpLent, inA, inB, outD, Prodt[itr]);          
                  for (int j = 0; j < tmpLent; j++)
                  {
                    outD[j] = (outD[j] + ((5000 << MAX) + 5000)) & maskt;
                  }
                  memcpy(Skylinet2[k3 - T1], outD, tmpLent * sizeof(uint64_t));
                }
                for (int k3 = T2; k3 >= T1; k3--)
                {
                  memcpy(Skylinet[((k1 - lenl) * len_y) + k3 - lend], Skylinet2[k3 - T1], tmpLent * sizeof(uint64_t));
                  delete[] Skylinet2[k3 - T1];
                }
                delete[] Skylinet2;
              }             
            }
            delete[] inA;
            delete[] inB;
            delete[] outD;
          }
        }
        #pragma omp taskwait
      }    
    }
  }
  // #pragma omp parallel for
  for (int k2 = lenu; k2 >= lend; k2--)
  {
    for (int k1 = lenr; k1 >= lenl; k1--)
    {
      uint64_t *tmpSkyline = new uint64_t[SS_L[(k1 * LENX) + k2]];
      memcpy(tmpSkyline, SS_Skyline[(k1 * LENX) + k2], SS_L[(k1 * LENX) + k2] * sizeof(uint64_t));
      delete[] SS_Skyline[(k1 * LENX) + k2];
      SS_Skyline[(k1 * LENX) + k2] = new uint64_t[tmpLen + SS_L[(k1 * LENX) + k2]];
      memcpy(SS_Skyline[(k1 * LENX) + k2], Skylinet[((k1 - lenl) * len_y) + k2 - lend], tmpLen * sizeof(uint64_t));
      memcpy(SS_Skyline[(k1 * LENX) + k2] + tmpLen, tmpSkyline, SS_L[(k1 * LENX) + k2] * sizeof(uint64_t));
      SS_L[(k1 * LENX) + k2] = tmpLen + SS_L[(k1 * LENX) + k2];
      delete[] tmpSkyline;
      // delete[] Skylinet[((k1 - lenl) * len_y) + k2 - lend];
    }
  }
  for (int k1 = len_x * len_y - 1; k1 >= 0; k1--)
  {
    delete[] Skylinet[k1];
  }
  delete[] Skylinet;
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  cout << "[(" << lenl << "," << lend << "),(" << (lenr) << "," << (lenu) << ")]" << endl;
  cout << "ConstrainSub_SS:"<< ss_t << " s" << endl;
}

// Inherit Sub
void ISub_SS_D1(int lenl, int lenr, int lend, int lenu, int i)
{
  double starts = omp_get_wtime();
  int LENX = SS_G[0][1];
  // PRG128 prg;
  uint32_t SS_one = 0;
  prg.random_data(&SS_one, sizeof(uint32_t));
  if (party == ALICE)
  {
    Iot[i]->send_data(&SS_one, sizeof(uint32_t));
  }
  else
  {
    Iot[i]->recv_data(&SS_one, sizeof(uint32_t));
    SS_one = (1 - SS_one) & mask;
  }
  // Dim1 Aggreate
  for (int k2 = lenu; k2 >= lend; k2--)
  {
    // cout << "D1:" << k2 << endl;
    int tmpLen = SS_L[((lenr + 1) * LENX) + k2];
    uint64_t *inA = new uint64_t[tmpLen];
    uint64_t *inB = new uint64_t[tmpLen];
    uint64_t *outD = new uint64_t[tmpLen];
    for (int k1 = lenr; k1 >= lenl; k1--)
    {
	    // memcpy(inA, SS_Skyline[((k1 + 1) * LENX) + k2], tmpLen * (sizeof(uint64_t)));
      // for (int j = 0; j < tmpLen; j++)
      // {
      //   inB[j] = (SS_one - SS_C[(k1 * LENX) + k2]) & mask;
      // }
      // //(1-con) * SS_Skyline[((k1 + 1) << MAX) + k2]
      // Prod_H(tmpLen, inA, inB, outD, Prodt[i]);
      for (int j = 0; j < tmpLen; j++)
      {
        inA[j] = (((5000 << MAX) + 5000) - SS_Skyline[((k1 + 1) * LENX) + k2][j]) & mask;
        inB[j] = SS_C[(k1 * LENX) + k2] & mask;
      }
      //(1-con) * SS_Skyline[((k1 + 1) << MAX) + k2]+con*1000
      Prod_H(tmpLen, inA, inB, outD, Prodt[i]);
      for (int j = 0; j < tmpLen; j++)
      {
        outD[j] = (outD[j] + SS_Skyline[((k1 + 1) * LENX) + k2][j]) & mask;
      }
      uint64_t *tmpSkyline = new uint64_t[SS_L[(k1 * LENX) + k2]];
      memcpy(tmpSkyline, SS_Skyline[(k1 * LENX) + k2], SS_L[(k1 * LENX) + k2] * (sizeof(uint64_t)));
      delete[] SS_Skyline[(k1 * LENX) + k2];
      SS_Skyline[(k1 * LENX) + k2] = new uint64_t[tmpLen + SS_L[(k1 * LENX) + k2]];
      memcpy(SS_Skyline[(k1 * LENX) + k2], outD, tmpLen * (sizeof(uint64_t)));
      memcpy(SS_Skyline[(k1 * LENX) + k2] + tmpLen, tmpSkyline, SS_L[(k1 * LENX) + k2] * (sizeof(uint64_t)));
      SS_L[(k1 * LENX) + k2] = tmpLen + SS_L[(k1 * LENX) + k2];      
      delete[] tmpSkyline;
    }
    delete[] inA;
    delete[] inB;
    delete[] outD;
  }
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  // cout << "[(" << lenl << "," << lend << "),(" << lenr << "," << lenu << ")]" << ss_t << " s" << endl;
}

void ISub_SS_D2(int lenl, int lenr, int lend, int lenu, int i)
{
  double starts = omp_get_wtime();
  // PRG128 prg;
  int LENX = SS_G[0][1];
  uint32_t SS_one = 0;
  prg.random_data(&SS_one, sizeof(uint32_t));
  if (party == ALICE)
  {
    Iot[i]->send_data(&SS_one, sizeof(uint32_t));
  }
  else
  {
    Iot[i]->recv_data(&SS_one, sizeof(uint32_t));
    SS_one = (1 - SS_one) & mask;
  }
  // Dim2 Aggreate
  for (int k1 = lenr; k1 >= lenl; k1--)
  {
    // cout << "D2:" << k1 << endl;
    int tmpLen = SS_L[(k1 * LENX) + lenu + 1];
    uint64_t *inA = new uint64_t[tmpLen];
    uint64_t *inB = new uint64_t[tmpLen];
    uint64_t *outD = new uint64_t[tmpLen];
    for (int k2 = lenu; k2 >= lend; k2--)
    { 
	    // memcpy(inA, SS_Skyline[(k1 * LENX) + k2 + 1], tmpLen * (sizeof(uint64_t)));
      // for (int j = 0; j < tmpLen; j++)
      // {
      //   inB[j] = (SS_one - SS_C[(k1 * LENX) + k2]) & mask;
      // }
      // //(1-con) * SSone
      // Prod_H(tmpLen, inA, inB, outD, Prodt[i]);
      for (int j = 0; j < tmpLen; j++)
      {
        inA[j] = (((5000 << MAX) + 5000) - SS_Skyline[(k1 * LENX) + k2 + 1][j]) & mask;
        inB[j] = SS_C[(k1 * LENX) + k2] & mask;
      }
      //(1-con) * SS_Skyline[((k1 + 1) << MAX) + k2]+con*1000
      Prod_H(tmpLen, inA, inB, outD, Prodt[i]);
      for (int j = 0; j < tmpLen; j++)
      {
        outD[j] = (outD[j] + SS_Skyline[(k1 * LENX) + k2 + 1][j]) & mask;
      }
      uint64_t *tmpSkyline = new uint64_t[SS_L[(k1 * LENX) + k2]];
      memcpy(tmpSkyline, SS_Skyline[(k1 * LENX) + k2], SS_L[(k1 * LENX) + k2] * (sizeof(uint64_t)));
      delete[] SS_Skyline[(k1 * LENX) + k2];
      SS_Skyline[(k1 * LENX) + k2] = new uint64_t[tmpLen + SS_L[(k1 * LENX) + k2]];
      memcpy(SS_Skyline[(k1 * LENX) + k2], outD, tmpLen * (sizeof(uint64_t)));
      memcpy(SS_Skyline[(k1 * LENX) + k2] + tmpLen, tmpSkyline, SS_L[(k1 * LENX) + k2] * (sizeof(uint64_t)));
      SS_L[(k1 * LENX) + k2] = tmpLen + SS_L[(k1 * LENX) + k2];     
      delete[] tmpSkyline;
    }
    delete[] inA;
    delete[] inB;
    delete[] outD;
  }
  double ends = omp_get_wtime();
  double ss_t = ends - starts;
  // cout << "[(" << lenl << "," << lend << "),(" << lenr << "," << lenu << ")]" << ss_t << " s" << endl;
}

vector<uint32_t> SS_Two(vector<uint32_t> q)
{
  vector<uint32_t> Q(q.size());
  // vector<uint32_t> SS_H(H.size());
  double startp = omp_get_wtime();
  // SS point whether select.
  SSQ(q, Q);
  // SSH(H, SS_H);
  // SS Grid
  SSG(G, SS_G);
  SSC(H, G, SS_C);
  double endp = omp_get_wtime();
  ss_upload = endp - startp;
  // Process
  double starts = omp_get_wtime();
  // PRG128 prg;
  double startSSub = omp_get_wtime();
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(SS_C, len1 * len2 * sizeof(uint32_t));
  //   Iot[0]->recv_data(SS_C, len1 * len2 * sizeof(uint32_t));
  // }
  // else
  // {
  //   uint32_t *SS_C0 = new uint32_t[len1 * len2];
  //   Iot[0]->recv_data(SS_C0, len1 * len2 * sizeof(uint32_t));
  //   for (int i = 0; i < len1 * len2; i++)
  //   {
  //     SS_C[i] = (SS_C0[i] + SS_C[i]) & mask;
  //   }
  //   delete[] SS_C0;
  //   Iot[0]->send_data(SS_C, len1 * len2 * sizeof(uint32_t));
  // } 
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= len1 - 1; k1++)
  //   {
  //     cout<<SS_C[k1 * len2 + k2]<<"\t";
  //   }
  //   cout<<endl;
  // }
  // cout<<"\t";
  // for (int k1 = 0; k1 <= len1 - 1; k1++)
  // {
  //   cout<<G[1][k1]<<"\t";
  // }
  // cout<<endl;

  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  cout << "K1:" << len1 << endl;
  cout << "K2:" << len2 << endl;
  omp_set_nested(20);
  // omp_set_dynamic(0);
  cout << "First Start" << endl;
  #pragma omp parallel num_threads(4)
  {
    // #pragma omp sections
    #pragma omp single
    // #pragma omp sections firstprivate(len1, len2, MAX, mask, lambda, SS_C, SS_G, SS_Skyline, SS_L)
    {
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "1 Start" << endl;
        // right up
        SSub_SS_Two_1((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
        // cout << "1 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "2 Start" << endl;
        // right down
        SSub_SS_Two_1((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 3);
        // cout << "2 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "3 Start" << endl;
        // left up
        SSub_SS_Two_1(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 6);
        // cout << "3 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "4 Start" << endl;
        // left down
        SSub_SS_Two_1(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 9);
        // cout << "4 Finish" << endl;
      }
      #pragma omp taskwait
    }
  }
  cout << "First Finish" << endl;
  #pragma omp parallel num_threads(4)
  {
    // #pragma omp sections
    #pragma omp single 
    {
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "1 Start" << endl;
        // right up
        SSub_SS_Two_2((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
        // SSub_SS_Two_T1((len1+1)/2,len1-1,(len2+1)/2,len2-1);
        cout << "1 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "2 Start" << endl;
        // right down
        SSub_SS_Two_2((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 3);
        // SSub_SS_Two_T2((len1+1)/2,len1-1,0,(len2-1)/2);
        cout << "2 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "3 Start" << endl;
        // left up
        SSub_SS_Two_2(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 6);
        // SSub_SS_Two_T3(0,(len1-1)/2,(len2+1)/2,len2-1);
        cout << "3 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "4 Start" << endl;
        // left down
        SSub_SS_Two_2(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 9);
        cout << "4 Finish" << endl;
      }
      #pragma omp taskwait
    }
  }
  // SSub_SS_Two((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
  // SSub_SS_Two((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 2);
  // SSub_SS_Two(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 4);
  // SSub_SS_Two(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 6);
  double endSSub = omp_get_wtime();
  cout << "SSub_SS_Two:" << (endSSub - startSSub) << " s" << endl;
  // SSub_SS_Two((len1+1)/2,len1-1,(len2+1)/2,len2-1); //right up
  // SSub_SS_Two((len1+1)/2,len1-1,0,(len2-1)/2); //right down
  // SSub_SS_Two(0,(len1-1)/2,(len2+1)/2,len2-1);  //left up
  // SSub_SS_Two(0,(len1-1)/2,0,(len2-1)/2);  //left down
  // SSub_SS_Two(0,(len1-1),0,(len2-1));
  // for (int k2 = SS_G[0][1] - 1; k2 >= 0; k2--)
  // {
  //   for (int k1 = 0; k1 <= SS_G[0][0] - 1; k1++)
  //   {
  //     SS_Print(SS_L[(k1 << MAX) + k2], SS_Skyline[(k1 << MAX) + k2]);
  //     cout << "\t";
  //   }
  //   cout << endl;
  // }
  // StoreSKyline();
  double startISub = omp_get_wtime();
  cout << " H-inherits Start" << endl;
  // 16 threads
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {       
        #pragma omp task firstprivate(len1, len2, THs)
        {
          int lendt = (((len2 - 1) * itr)/ THs) + 1;
          int lenut = ((len2 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          // Left inherits right
          ISub_SS_D1(0, (len1 - 1) / 2, lendt, lenut, itr);
        }        
      }
      #pragma omp taskwait
    }
  }
  cout << " H-inherits Finish" << endl;
  cout << " V-inherits Start" << endl;  
  #pragma omp parallel num_threads(THs)
  {
    // #pragma omp sections
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {             
        #pragma omp task firstprivate(len1, len2, THs)
        {
          int lenlt = (((len1 - 1) * itr)/ THs) + 1;
          int lenrt = ((len1 - 1) * (itr + 1))/ THs;
          if(itr == 0){
            lenlt = 0;
          }
          // Bottom inherits top
          ISub_SS_D2(lenlt, lenrt, 0, (len2 - 1) / 2, itr);
        }        
      }
    }
    #pragma omp taskwait
  }
  cout << " V-inherits Finish" << endl;  
  double endISub = omp_get_wtime();
  cout << "InheritSub_SS:" << (endISub - startISub) << " s" << endl;
  // unordered_map<uint32_t, uint64_t *> Skyline;
  // unordered_map<uint32_t, int> ss_L;
  // // #pragma omp parallel for
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   for (int k1 = len1 - 1; k1 >= 0; k1--)
  //   {
  //     Skyline[(k1 << MAX) + k2] = new uint64_t[SS_L[(k1 << MAX) + k2]];
  //     copy(SS_Skyline[(k1 << MAX) + k2], SS_Skyline[(k1 << MAX) + k2] + SS_L[(k1 << MAX) + k2], Skyline[(k1 << MAX) + k2]);
  //     ss_L[(k1 << MAX) + k2] = SS_L[(k1 << MAX) + k2];
  //   }
  // }
  cout << "last step" << endl;
  // StoreSKyline();
  CSub_SS_Two(0, (len1 + 1) / 2, 0, (len2 + 1) / 2); // create new skyline and length
  // SSub_SS_Two(0, len1 - 1, 0, len2 - 1);
  double ends = omp_get_wtime();
  ss_process = ends - starts;
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= len1 - 1; k1++)
  //   {
  //     SS_Print(SS_L[k1 * len2 + k2], SS_Skyline[k1 * len2 + k2]);
  //     cout << "\t";
  //   }
  //   cout << endl;
  // }
  return Q;
}

void SS_Two(unordered_map<uint32_t, vector<uint32_t>> G)
{
  // vector<uint32_t> SS_H(H.size());
  double startp = omp_get_wtime();
  // SS point whether select.
  // SSH(H, SS_H);
  // SS Grid
  SSG(G, SS_G);
  SSC(H, G, SS_C);
  double endp = omp_get_wtime();
  ss_upload = endp - startp;
  // Process
  double starts = omp_get_wtime();
  // PRG128 prg;
  double startSSub = omp_get_wtime();
  int len1 = SS_G[0][0];
  int len2 = SS_G[0][1];
  SS_L = new int[len1 * len2];
  SS_Skyline = new uint64_t*[len1 * len2];
  cout << "K1:" << len1 << endl;
  cout << "K2:" << len2 << endl;
  omp_set_nested(20);
  // omp_set_dynamic(0);
  cout << "First Start" << endl;
  #pragma omp parallel num_threads(4)
  {
    // #pragma omp sections
    #pragma omp single
    // #pragma omp sections firstprivate(len1, len2, MAX, mask, lambda, SS_C, SS_G, SS_Skyline, SS_L)
    {
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "1 Start" << endl;
        // right up
        SSub_SS_Two_1((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
        // cout << "1 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "2 Start" << endl;
        // right down
        SSub_SS_Two_1((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 3);
        // cout << "2 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "3 Start" << endl;
        // left up
        SSub_SS_Two_1(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 6);
        // cout << "3 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_C, SS_G)
      {
        // cout << "4 Start" << endl;
        // left down
        SSub_SS_Two_1(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 9);
        // cout << "4 Finish" << endl;
      }
      #pragma omp taskwait
    }
  }
  cout << "First Finish" << endl;
  #pragma omp parallel num_threads(4)
  {
    // #pragma omp sections
    #pragma omp single 
    {
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "1 Start" << endl;
        // right up
        SSub_SS_Two_2((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
        // SSub_SS_Two_T1((len1+1)/2,len1-1,(len2+1)/2,len2-1);
        cout << "1 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "2 Start" << endl;
        // right down
        SSub_SS_Two_2((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 3);
        // SSub_SS_Two_T2((len1+1)/2,len1-1,0,(len2-1)/2);
        cout << "2 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "3 Start" << endl;
        // left up
        SSub_SS_Two_2(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 6);
        // SSub_SS_Two_T3(0,(len1-1)/2,(len2+1)/2,len2-1);
        cout << "3 Finish" << endl;
      }
      #pragma omp task firstprivate(len1, len2, SS_G, SS_C)
      {
        cout << "4 Start" << endl;
        // left down
        SSub_SS_Two_2(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 9);
        cout << "4 Finish" << endl;
      }
      #pragma omp taskwait
    }
  }
  // SSub_SS_Two((len1 + 1) / 2, len1 - 1, (len2 + 1) / 2, len2 - 1, 0);
  // SSub_SS_Two((len1 + 1) / 2, len1 - 1, 0, (len2 - 1) / 2, 2);
  // SSub_SS_Two(0, (len1 - 1) / 2, (len2 + 1) / 2, len2 - 1, 4);
  // SSub_SS_Two(0, (len1 - 1) / 2, 0, (len2 - 1) / 2, 6);
  double endSSub = omp_get_wtime();
  cout << "SSub_SS_Two:" << (endSSub - startSSub) << " s" << endl;
  // SSub_SS_Two((len1+1)/2,len1-1,(len2+1)/2,len2-1); //right up
  // SSub_SS_Two((len1+1)/2,len1-1,0,(len2-1)/2); //right down
  // SSub_SS_Two(0,(len1-1)/2,(len2+1)/2,len2-1);  //left up
  // SSub_SS_Two(0,(len1-1)/2,0,(len2-1)/2);  //left down
  // SSub_SS_Two(0,(len1-1),0,(len2-1));
  // StoreSKyline();
  double startISub = omp_get_wtime();
  cout << " H-inherits Start" << endl;
  // 16 threads
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {       
        #pragma omp task firstprivate(len1, len2, THs)
        {
          int lendt = ((((len2 - 1)/ 2) * itr)/ THs) + 1;
          int lenut = (((len2 - 1)/ 2) * (itr + 1))/ THs;
          if(itr == 0){
            lendt = 0;
          }
          // Left inherits right
          ISub_SS_D1(0, (len1 - 1) / 2, lendt, lenut, itr);
        }        
      }
      #pragma omp taskwait
    }
  }
  cout << " V-inherits Start" << endl;  
  #pragma omp parallel num_threads(THs)
  {
    // #pragma omp sections
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {             
        #pragma omp task firstprivate(len1, len2, THs)
        {
          int lenlt = ((((len1 - 1)/ 2) * itr)/ THs) + 1;
          int lenrt = (((len1 - 1)/ 2) * (itr + 1))/ THs;
          if(itr == 0){
            lenlt = 0;
          }
          // Bottom inherits top
          ISub_SS_D2(lenlt, lenrt, 0, (len2 - 1) / 2, itr);
        }        
      }
    }
    #pragma omp taskwait
  }
  #pragma omp parallel num_threads(THs)
  {
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {       
        #pragma omp task firstprivate(len1, len2, THs)
        {
          //(len2 + 1) / 2 ~ len2 - 1; length (len2 - 1) / 2
          int lendt = (((len2 - 1 - (len2 + 1) / 2) * itr)/ THs) + 1 + (len2 + 1) / 2;
          int lenut = (((len2 - 1 - (len2 + 1) / 2) * (itr + 1))/ THs) + (len2 + 1) / 2;
          if(itr == 0){
            lendt = (len2 + 1) / 2;
          }
          // Left inherits right
          ISub_SS_D1(0, (len1 - 1) / 2, lendt, lenut, itr);
        }        
      }
      #pragma omp taskwait
    }
  }
  #pragma omp parallel num_threads(THs)
  {
    // #pragma omp sections
    #pragma omp single 
    {
      for (int itr = 0; itr < THs; itr++)
      {             
        #pragma omp task firstprivate(len1, len2, THs)
        {
          int lenlt = (((len1 - 1 - (len1 + 1) / 2) * itr)/ THs) + 1 + (len1 + 1) / 2;
          int lenrt = (((len1 - 1 - (len1 + 1) / 2) * (itr + 1))/ THs) + (len1 + 1) / 2;
          if(itr == 0){
            lenlt = (len1 + 1) / 2;
          }
          // Bottom inherits top
          ISub_SS_D2(lenlt, lenrt, 0, (len2 - 1) / 2, itr);
        }        
      }
    }
    #pragma omp taskwait
  }
  cout << " H-inherits Finish" << endl;
  cout << " V-inherits Finish" << endl;  
  double endISub = omp_get_wtime();
  cout << "InheritSub_SS:" << (endISub - startISub) << " s" << endl;
  // unordered_map<uint32_t, uint64_t *> Skyline;
  // unordered_map<uint32_t, int> ss_L;
  // // #pragma omp parallel for
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   for (int k1 = len1 - 1; k1 >= 0; k1--)
  //   {
  //     Skyline[(k1 << MAX) + k2] = new uint64_t[SS_L[(k1 << MAX) + k2]];
  //     copy(SS_Skyline[(k1 << MAX) + k2], SS_Skyline[(k1 << MAX) + k2] + SS_L[(k1 << MAX) + k2], Skyline[(k1 << MAX) + k2]);
  //     ss_L[(k1 << MAX) + k2] = SS_L[(k1 << MAX) + k2];
  //   }
  // }
  cout << "last step" << endl;
  // StoreSKyline();
  // CSub_SS_Two(0, (len1 + 1) / 2, 0, (len2 + 1) / 2); // create new skyline and length
  CSub_SS_Two(0, (len1 - 1) / 2, 0, (len2 - 1) / 2);
  // SSub_SS_Two(0, len1 - 1, 0, len2 - 1);
  double ends = omp_get_wtime();
  ss_process = ends - starts;  
  if (party == ALICE)
  {
    cout << "Alice Share Time\t" << ss_upload << " s" << endl;
    cout << "Alice Process Time\t" << ss_process << " s" << endl;
  }else
  {
    cout << "Bob Share Time\t"<< ss_upload << " s" << endl;
    cout << "Bob Process Time\t" << ss_process << " s" << endl;
  }
  
  // uint32_t *SS_C0 = new uint32_t[len1 * len2];
  // if (party == ALICE)
  // {
  //   Iot[0]->send_data(SS_C, len1 * len2 * sizeof(uint32_t));
  //   Iot[0]->recv_data(SS_C0, len1 * len2 * sizeof(uint32_t));
  // }
  // else
  // {
  //   Iot[0]->recv_data(SS_C0, len1 * len2 * sizeof(uint32_t));
  //   for (int i = 0; i < len1 * len2; i++)
  //   {
  //     SS_C0[i] = (SS_C0[i] + SS_C[i]) & mask;
  //   }
  //   Iot[0]->send_data(SS_C0, len1 * len2 * sizeof(uint32_t));
  // } 
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= len1 - 1; k1++)
  //   {
  //     cout<<SS_C0[k1 * len2 + k2]<<"\t";
  //   }
  //   cout<<endl;
  // }
  // cout<<"\t";
  // for (int k1 = 0; k1 <= len1 - 1; k1++)
  // {
  //   cout<<G[1][k1]<<"\t";
  // }
  // cout<<endl;
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   cout<<G[2][k2]<<"\t";
  //   for (int k1 = 0; k1 <= len1 - 1; k1++)
  //   {
  //     SS_Print(SS_L[k1 * len2 + k2], SS_Skyline[k1 * len2 + k2]);
  //     // cout<<SS_L[k1 * len2 + k2];
  //     cout << "\t";
  //   }
  //   cout << endl;
  // }
  // cout<<"\t";
  // for (int k1 = 0; k1 <= len1 - 1; k1++)
  // {
  //   cout<<G[1][k1]<<"\t";
  // }
  // cout<<endl;
}

vector<uint64_t> SS_Result(vector<uint32_t> q, vector<uint32_t> Q)
{
  // vector<uint32_t> pos(q.size());
  // Pos(Q, SS_G, pos);
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  PosIndex(Q, SS_G, pos, posindex);
  uint32_t *tmp = new uint32_t[m];
  if (party == ALICE)
  {
    // copy(pos.begin(), pos.end(), tmp);
    copy(pos, pos+m, tmp);
    Iot[0]->send_data(tmp, m * sizeof(uint32_t));
    Iot[0]->recv_data(tmp, m * sizeof(uint32_t));
  }
  else
  {
    uint32_t *pos0 = new uint32_t[m];
    Iot[0]->recv_data(pos0, m * sizeof(uint32_t));
    for (int i = 0; i < m; i++)
    {
      tmp[i] = ((pos[i] + pos0[i]) & mask);
    }
    delete[] pos0;
    Iot[0]->send_data(tmp, m * sizeof(uint32_t));
  }
  cout << tmp[0] << "," << tmp[1] << endl;
  int skylineLen = SS_L[(tmp[0] * SS_G[0][1]) + tmp[1]];
  uint64_t *s = new uint64_t[skylineLen];
  copy(SS_Skyline[(tmp[0] * SS_G[0][1]) + tmp[1]], SS_Skyline[(tmp[0] * SS_G[0][1]) + tmp[1]] + skylineLen, s);
  // s = SS_Skyline[(tmp[0]<<MAX)+tmp[1]];
  if (party == ALICE)
  {
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
    Iot[0]->recv_data(s, skylineLen * sizeof(uint64_t));
  }
  else
  {
    uint64_t *s0 = new uint64_t[skylineLen];
    Iot[0]->recv_data(s0, skylineLen * sizeof(uint64_t));
    for (int i = 0; i < skylineLen; i++)
    {
      s[i] = ((s[i] + s0[i]) & mask);
    }
    delete[] s0;
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
  }
  uint64_t *restmp = new uint64_t[skylineLen * m];
  int mm = 0;
  cout << skylineLen << endl;
  uint64_t *x = new uint64_t[m];
  for (int i = 0; i < skylineLen; i++)
  {
    x[0] = s[i] & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      s[i] = (s[i] - x[j - 1]) >> MAX;
      x[j] = s[i] & ((1ULL << MAX) - 1);
    }
    for (int j = m - 1; j >= 0; j--)
    {
      cout << x[j] << "\t";
    }
    cout << endl;
    copy(x, x + m, restmp + mm * m);
    mm++;
  }
  delete[] x;
  vector<uint64_t> res(skylineLen * m);
  copy(restmp, restmp + skylineLen * m, res.begin());
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  delete[] tmp;
  delete[] restmp;
  delete[] s;
  return res;
}

vector<uint64_t> SS_Result(vector<uint32_t> Q, uint32_t &respos)
{
  // vector<uint32_t> pos(q.size());
  // Pos(Q, SS_G, pos);
  uint32_t *pos = new uint32_t[m];
  unordered_map<uint32_t, uint32_t *> posindex;
  PosIndex(Q, SS_G, pos, posindex);
  uint32_t *tmp = new uint32_t[m];
  if (party == ALICE)
  {
    // copy(pos.begin(), pos.end(), tmp);
    copy(pos, pos+m, tmp);
    Iot[0]->send_data(tmp, m * sizeof(uint32_t));
    Iot[0]->recv_data(tmp, m * sizeof(uint32_t));
  }
  else
  {
    uint32_t *pos0 = new uint32_t[m];
    Iot[0]->recv_data(pos0, m * sizeof(uint32_t));
    for (int i = 0; i < m; i++)
    {
      tmp[i] = ((pos[i] + pos0[i]) & mask);
    }
    delete[] pos0;
    Iot[0]->send_data(tmp, m * sizeof(uint32_t));
  }
  cout << tmp[0] << "," << tmp[1] << endl;
  int skylineLen = SS_L[(tmp[0] * SS_G[0][1]) + tmp[1]];
  uint64_t *s = new uint64_t[skylineLen];
  copy(SS_Skyline[(tmp[0] * SS_G[0][1]) + tmp[1]], SS_Skyline[(tmp[0] * SS_G[0][1]) + tmp[1]] + skylineLen, s);
  respos = (tmp[0] * SS_G[0][1]) + tmp[1];
  // s = SS_Skyline[(tmp[0]<<MAX)+tmp[1]];
  if (party == ALICE)
  {
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
    Iot[0]->recv_data(s, skylineLen * sizeof(uint64_t));
  }
  else
  {
    uint64_t *s0 = new uint64_t[skylineLen];
    Iot[0]->recv_data(s0, skylineLen * sizeof(uint64_t));
    for (int i = 0; i < skylineLen; i++)
    {
      s[i] = ((s[i] + s0[i]) & mask);
    }
    delete[] s0;
    Iot[0]->send_data(s, skylineLen * sizeof(uint64_t));
  }
  uint64_t *restmp = new uint64_t[skylineLen * m];
  int mm = 0;
  cout << skylineLen << endl;
  uint64_t *x = new uint64_t[m];
  for (int i = 0; i < skylineLen; i++)
  {
    x[0] = s[i] & ((1ULL << MAX) - 1);
    for (int j = 1; j < m; j++)
    {
      s[i] = (s[i] - x[j - 1]) >> MAX;
      x[j] = s[i] & ((1ULL << MAX) - 1);
    }
    for (int j = m - 1; j >= 0; j--)
    {
      cout << x[j] << "\t";
    }
    cout << endl;
    copy(x, x + m, restmp + mm * m);
    mm++;
  }
  delete[] x;
  vector<uint64_t> res(skylineLen * m);
  copy(restmp, restmp + skylineLen * m, res.begin());
  delete[] pos;
  for (int k = 0; k < m; k++)
  {
    delete[] posindex[k];
  }
  posindex.clear();
  delete[] tmp;
  delete[] restmp;
  delete[] s;
  return res;
}

void dynamicSkyline(unordered_map<uint32_t, vector<uint32_t>> &G, vector<uint32_t> q){
  unordered_map<uint32_t, vector<uint32_t>> G1(G);
  unordered_map<uint32_t, vector<uint32_t>> G2(G);
  unordered_map<uint32_t, vector<uint32_t>> G4(G);
  int len1 = G[1].size();
  int len2 = G[2].size();
  unordered_map<uint32_t, vector<uint32_t>> G3;
  for (int j = 0; j < m; j++)
  {
    vector<uint32_t> tmp;
    int len = G[j+1].size();
    for (int k = 0; k < len; k++)
    {
      tmp.push_back(G[j+1][len-1-k]);
    }
    G3[j + 1] = tmp;
  }
  G2[1] = G3[1];
  G4[2] = G3[2];
  //dynamic borders add one row/column
  // Store
  for (int i = 0; i < 4; i++)
  {
    string path = "-";
    if(i == 0){
      G = G1;
      path = path + "1";
    }else if (i == 1){
      G = G2;
      path = path + "2";
    }else if (i == 2){
      G = G3;
      path = path + "3";
    }else{
      G = G4;
      path = path + "4";
    }
    SS_Two(G);
    StoreSKyline(path);
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline[j];
    }
    delete[] SS_Skyline;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
  }
  // // for (int i = 0; i < 4; i++)
  // // {
  // //   if(i == 0){
  // //     G = G1;
  // //   }else if (i == 1){
  // //     G = G2;
  // //   }else if (i == 2){
  // //     G = G3;
  // //   }else{
  // //     G = G4;
  // //   }
  // //   SS_Two(G);
  // //   if(i == 0){
  // //     for (int k2 = len2 - 1; k2 >= 0; k2--)
  // //     {
  // //       for (int k1 = 0; k1 <= len1 - 1; k1++)
  // //       {
  // //         SS_Lt[k1 * len2t + k2] = SS_L[k1 * len2 + k2];
  // //         SS_Skylinet[k1 * len2t + k2] = new uint64_t[SS_L[k1 * len2 + k2]];
  // //         memcpy(SS_Skylinet[k1 * len2t + k2], SS_Skyline[k1 * len2 + k2], SS_L[k1 * len2 + k2] * sizeof(uint64_t));
  // //       }
  // //     }
  // //     // SSG(G1, SS_G);
  // //     // vector<uint32_t> Q(q.size());
  // //     // SSQ(q, Q);
  // //     // SS_Result(q, Q);
  // //   }else if (i == 1){  //  right move one column
  // //     for (int k2 = len2 - 1; k2 >= 0; k2--)
  // //     {
  // //       SS_Skylinet[len1 * len2t + k2] = new uint64_t[SS_L[0 * len2 + k2]];
  // //       memcpy(SS_Skylinet[len1 * len2t + k2], SS_Skyline[0 * len2 + k2], SS_L[0 * len2 + k2] * sizeof(uint64_t));
  // //       SS_Lt[len1 * len2t + k2] = SS_L[0 * len2 + k2];
  // //     }
  // //     for (int k2 = len2 - 1; k2 >= 0; k2--)
  // //     {
  // //       for (int k1 = 0; k1 <= len1 - 2; k1++)
  // //       {
  // //         int k1t = len1 - 1 - k1;
  // //         uint64_t *tmpSkyline = new uint64_t[SS_Lt[(k1 + 1) * len2t + k2]];
  // //         memcpy(tmpSkyline, SS_Skylinet[(k1 + 1) * len2t + k2], SS_Lt[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
  // //         delete[] SS_Skylinet[(k1 + 1) * len2t + k2];
  // //         SS_Skylinet[(k1 + 1) * len2t + k2] = new uint64_t[SS_L[k1t * len2 + k2] + SS_Lt[(k1 + 1) * len2t + k2]];
  // //         memcpy(SS_Skylinet[(k1 + 1) * len2t + k2], tmpSkyline, SS_Lt[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
  // //         memcpy(SS_Skylinet[(k1 + 1) * len2t + k2] + SS_Lt[(k1 + 1) * len2t + k2], SS_Skyline[k1t * len2 + k2], SS_L[k1t * len2 + k2] * sizeof(uint64_t));
  // //         SS_Lt[(k1 + 1) * len2t + k2] = SS_L[k1t * len2 + k2] + SS_Lt[(k1 + 1) * len2t + k2];
  // //         delete[] tmpSkyline;
  // //       }
  // //     }
  // //   }else if (i == 2){  // right move one column, up move one row
  // //     for (int k1 = 0; k1 <= len1 - 1; k1++)
  // //     {
  // //       SS_Skylinet[(k1 + 1) * len2t + len2] = new uint64_t[SS_L[(len1 - 1 - k1) * len2 + 0]];
  // //       memcpy(SS_Skylinet[(k1 + 1) * len2t + len2], SS_Skyline[(len1 - 1 - k1) * len2 + 0], SS_L[(len1 - 1 - k1) * len2 + 0] * sizeof(uint64_t));
  // //       SS_Lt[(k1 + 1) * len2t + len2] = SS_L[(len1 - 1 - k1) * len2 + 0];
  // //     }
  // //     for (int k2 = len2 - 2; k2 >= 0; k2--)
  // //     {
  // //       for (int k1 = 0; k1 <= len1 - 1; k1++)
  // //       {
  // //         int k1t = len1 - 1 - k1;
  // //         int k2t = len2 - 1 - k2;
  // //         uint64_t *tmpSkyline = new uint64_t[SS_Lt[(k1 + 1) * len2t + (k2 + 1)]];
  // //         memcpy(tmpSkyline, SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)], SS_Lt[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
  // //         delete[] SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)];
  // //         SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)] = new uint64_t[SS_L[k1t * len2 + k2t] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)]];
  // //         memcpy(SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)], tmpSkyline, SS_Lt[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
  // //         memcpy(SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)], SS_Skyline[k1t * len2 + k2t], SS_L[k1t * len2 + k2t] * sizeof(uint64_t));
  // //         SS_Lt[(k1 + 1) * len2t + (k2 + 1)] = SS_L[k1t * len2 + k2t] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)];
  // //         delete[] tmpSkyline;
  // //       }
  // //     }
  // //   }else{  //  up move one row
  // //     SS_Skylinet[0 * len2t + len2] = new uint64_t[SS_L[0 * len2 + 0]];
  // //     memcpy(SS_Skylinet[0 * len2t + len2], SS_Skyline[0 * len2 + 0], SS_L[0 * len2 + 0] * sizeof(uint64_t));
  // //     SS_Lt[0 * len2t + len2] = SS_L[0 * len2 + 0];
  // //     for (int k2 = len2 - 1; k2 >= 0; k2--)
  // //     {
  // //       for (int k1 = 0; k1 <= len1 - 1; k1++)
  // //       {
  // //         if((k2 == len2 - 1)&&(k1 == 0)){
  // //           continue;
  // //         }
  // //         int k2t = len2 - 1 - k2;
  // //         uint64_t *tmpSkyline = new uint64_t[SS_Lt[k1 * len2t + (k2 + 1)]];
  // //         memcpy(tmpSkyline, SS_Skylinet[k1 * len2t + (k2 + 1)], SS_Lt[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
  // //         delete[] SS_Skylinet[k1 * len2t + (k2 + 1)];
  // //         SS_Skylinet[k1 * len2t + (k2 + 1)] = new uint64_t[SS_L[k1 * len2 + k2t] + SS_Lt[k1 * len2t + (k2 + 1)]];
  // //         memcpy(SS_Skylinet[k1 * len2t + (k2 + 1)], tmpSkyline, SS_Lt[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
  // //         memcpy(SS_Skylinet[k1 * len2t + (k2 + 1)] + SS_Lt[k1 * len2t + (k2 + 1)], SS_Skyline[k1 * len2 + k2t], SS_L[k1 * len2 + k2t] * sizeof(uint64_t));
  // //         SS_Lt[k1 * len2t + (k2 + 1)] = SS_L[k1 * len2 + k2t] + SS_Lt[k1 * len2t + (k2 + 1)];
  // //         delete[] tmpSkyline;
  // //       }
  // //     }
  // //   }
  // //   delete[] SS_Con;
  // //   delete[] SS_C;
  // //   delete[] SS_L;
  // //   for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  // //   {
  // //     delete[] SS_Skyline[j];
  // //   }
  // //   delete[] SS_Skyline;
  // //   for(int j = 0; j< m; j++)
  // //   {
  // //     delete[] SS_G[j];
  // //   }
  // //   delete[] SS_G;
  // // }
  // for (int k2 = len2 - 1; k2 >= 0; k2--)
  // {
  //   for (int k1 = 0; k1 <= len1 - 1; k1++)
  //   {
  //     SS_L[k1 * len2 + k2] = SS_Lt[k1 * len2t + k2];
  //     SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_Lt[k1 * len2t + k2]];
  //     memcpy(SS_Skyline[k1 * len2 + k2], SS_Skylinet[k1 * len2t + k2], SS_Lt[k1 * len2t + k2] * sizeof(uint64_t));
  //   }
  // }
  // SSG(G1, SS_G);
  // vector<uint32_t> Q(q.size());
  // SSQ(q, Q);
  // SS_Result(q, Q);
  unordered_set<uint32_t>().swap(H);
  H.clear();
  unordered_map<uint32_t, vector<uint32_t>>().swap(G);
  G.clear();
  unordered_map<uint32_t, unordered_set<uint32_t>>().swap(Skyline);
  Skyline.clear();
}

void dynamicSkylineR(unordered_map<uint32_t, vector<uint32_t>> &G, vector<uint32_t> q){
  unordered_map<uint32_t, vector<uint32_t>> G1(G);
  unordered_map<uint32_t, vector<uint32_t>> G2(G);
  unordered_map<uint32_t, vector<uint32_t>> G4(G);
  int len1 = G[1].size();
  int len2 = G[2].size();
  unordered_map<uint32_t, vector<uint32_t>> G3;
  for (int j = 0; j < m; j++)
  {
    vector<uint32_t> tmp;
    int len = G[j+1].size();
    for (int k = 0; k < len; k++)
    {
      tmp.push_back(G[j+1][len-1-k]);
    }
    G3[j + 1] = tmp;
  }
  G2[1] = G3[1];
  G4[2] = G3[2];
  //dynamic borders add one row/column
  // Read
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  int *SS_Lt = new int[len1t * len2t];
  uint64_t **SS_Skylinet = new uint64_t*[len1t * len2t];
  double sr = omp_get_wtime();
  for (int i = 0; i < 4; i++)
  {
    string path = "-";
    if(i == 0){
      G = G1;
      path = path + "1";
    }else if (i == 1){
      G = G2;
      path = path + "2";
    }else if (i == 2){
       G = G3;
      path = path + "3";
    }else{
       G = G4;
      path = path + "4";
    }
    ReadSKyline(path);
    SSG(G, SS_G);
    if(i == 0){
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          SS_Lt[k1 * len2t + k2] = SS_L[k1 * len2 + k2];
          SS_Skylinet[k1 * len2t + k2] = new uint64_t[SS_L[k1 * len2 + k2]];
          memcpy(SS_Skylinet[k1 * len2t + k2], SS_Skyline[k1 * len2 + k2], SS_L[k1 * len2 + k2] * sizeof(uint64_t));
        }
      }
      // SSG(G1, SS_G);
      // vector<uint32_t> Q(q.size());
      // SSQ(q, Q);
      // SS_Result(q, Q);
    }else if (i == 1){  //  right move one column
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        SS_Skylinet[len1 * len2t + k2] = new uint64_t[SS_L[0 * len2 + k2]];
        memcpy(SS_Skylinet[len1 * len2t + k2], SS_Skyline[0 * len2 + k2], SS_L[0 * len2 + k2] * sizeof(uint64_t));
        SS_Lt[len1 * len2t + k2] = SS_L[0 * len2 + k2];
      }
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 2; k1++)
        {
          int k1t = len1 - 1 - k1;
          uint64_t *tmpSkyline = new uint64_t[SS_Lt[(k1 + 1) * len2t + k2]];
          memcpy(tmpSkyline, SS_Skylinet[(k1 + 1) * len2t + k2], SS_Lt[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
          delete[] SS_Skylinet[(k1 + 1) * len2t + k2];
          SS_Skylinet[(k1 + 1) * len2t + k2] = new uint64_t[SS_L[k1t * len2 + k2] + SS_Lt[(k1 + 1) * len2t + k2]];
          memcpy(SS_Skylinet[(k1 + 1) * len2t + k2], tmpSkyline, SS_Lt[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
          memcpy(SS_Skylinet[(k1 + 1) * len2t + k2] + SS_Lt[(k1 + 1) * len2t + k2], SS_Skyline[k1t * len2 + k2], SS_L[k1t * len2 + k2] * sizeof(uint64_t));
          SS_Lt[(k1 + 1) * len2t + k2] = SS_L[k1t * len2 + k2] + SS_Lt[(k1 + 1) * len2t + k2];
          delete[] tmpSkyline;
        }
      }
    }else if (i == 2){  // right move one column, up move one row
      for (int k1 = 0; k1 <= len1 - 1; k1++)
      {
        SS_Skylinet[(k1 + 1) * len2t + len2] = new uint64_t[SS_L[(len1 - 1 - k1) * len2 + 0]];
        memcpy(SS_Skylinet[(k1 + 1) * len2t + len2], SS_Skyline[(len1 - 1 - k1) * len2 + 0], SS_L[(len1 - 1 - k1) * len2 + 0] * sizeof(uint64_t));
        SS_Lt[(k1 + 1) * len2t + len2] = SS_L[(len1 - 1 - k1) * len2 + 0];
      }
      for (int k2 = len2 - 2; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          int k1t = len1 - 1 - k1;
          int k2t = len2 - 1 - k2;
          uint64_t *tmpSkyline = new uint64_t[SS_Lt[(k1 + 1) * len2t + (k2 + 1)]];
          memcpy(tmpSkyline, SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)], SS_Lt[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
          delete[] SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)];
          SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)] = new uint64_t[SS_L[k1t * len2 + k2t] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)]];
          memcpy(SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)], tmpSkyline, SS_Lt[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
          memcpy(SS_Skylinet[(k1 + 1) * len2t + (k2 + 1)] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)], SS_Skyline[k1t * len2 + k2t], SS_L[k1t * len2 + k2t] * sizeof(uint64_t));
          SS_Lt[(k1 + 1) * len2t + (k2 + 1)] = SS_L[k1t * len2 + k2t] + SS_Lt[(k1 + 1) * len2t + (k2 + 1)];
          delete[] tmpSkyline;
        }
      }
    }else{  //  up move one row
      SS_Skylinet[0 * len2t + len2] = new uint64_t[SS_L[0 * len2 + 0]];
      memcpy(SS_Skylinet[0 * len2t + len2], SS_Skyline[0 * len2 + 0], SS_L[0 * len2 + 0] * sizeof(uint64_t));
      SS_Lt[0 * len2t + len2] = SS_L[0 * len2 + 0];
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          if((k2 == len2 - 1)&&(k1 == 0)){
            continue;
          }
          int k2t = len2 - 1 - k2;
          uint64_t *tmpSkyline = new uint64_t[SS_Lt[k1 * len2t + (k2 + 1)]];
          memcpy(tmpSkyline, SS_Skylinet[k1 * len2t + (k2 + 1)], SS_Lt[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
          delete[] SS_Skylinet[k1 * len2t + (k2 + 1)];
          SS_Skylinet[k1 * len2t + (k2 + 1)] = new uint64_t[SS_L[k1 * len2 + k2t] + SS_Lt[k1 * len2t + (k2 + 1)]];
          memcpy(SS_Skylinet[k1 * len2t + (k2 + 1)], tmpSkyline, SS_Lt[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
          memcpy(SS_Skylinet[k1 * len2t + (k2 + 1)] + SS_Lt[k1 * len2t + (k2 + 1)], SS_Skyline[k1 * len2 + k2t], SS_L[k1 * len2 + k2t] * sizeof(uint64_t));
          SS_Lt[k1 * len2t + (k2 + 1)] = SS_L[k1 * len2 + k2t] + SS_Lt[k1 * len2t + (k2 + 1)];
          delete[] tmpSkyline;
        }
      }
    }
  }
  double er = omp_get_wtime();
  ss_read = er - sr;
  SSG(G1, SS_G);
  for (int k2 = len2 - 1; k2 >= 0; k2--)
  {
    for (int k1 = 0; k1 <= len1 - 1; k1++)
    {
      //   // dynamic points
      // // DP(SS_Lt[k1 * len2t + k2], SS_Skylinet[k1 * len2t + k2], qt, SS_L[k1 * len2 + k2], SS_Skyline[k1 * len2 + k2]);
      // uint64_t *qt = new uint64_t[m];
      // if(k1 == 0){
      //   qt[0] = 0;
      // }else{
      //   qt[0] = SS_G[1][k1 - 1];
      // }
      // if(k2 == 0){
      //   qt[1] = 0;
      // }else{
      //   qt[1] = SS_G[2][k2 - 1];
      // }
      // int lt = SS_L[k1 * len2t + k2];
      // uint64_t **pt = new uint64_t[lt][m];
      // uint64_t *Bt = new uint64_t[lt];
      // uint64_t *Rt = new uint64_t[lt*m];
      // prg.random_data(Rt, lt*m*sizeof(uint64_t));
      // if (party == ALICE)
      // {
      //   for (int j = 0; j < lt*m; j++)
      //   {
      //     Rt[j] = Rt[j] & ((1ULL << (MAX-2)) - 1);
      //   }
      //   for (int i = 0; i < lt; i++)
      //   {
      //     Bt[i] = (SS_Skyline[k1 * len2t + k2][i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
      //   }
      //   Iot[0]->send_data(Bt, lt * sizeof(uint64_t));
      //   uint64_t *Rt0 = new uint64_t[lt*m];
      //   Iot[0]->recv_data(Rt0, lt*m*sizeof(uint64_t));
      //   for (int i = 0; i < lt; i++)
      //   {
      //     for (int j = 0; j < m; j++)
      //     {
      //       pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
      //     } 
      //   }
      //   delete[] Rt0;
      // }
      // else
      // {
      //   Iot[0]->recv_data(Bt, lt * sizeof(uint64_t));
      //   uint64_t *s = new uint64_t[m];
      //   for (int i = 0; i < lt; i++)
      //   {
      //     s[0] = Bt[i] & ((1ULL << MAX) - 1);
      //     for (int j = 1; j < m; j++)
      //     {
      //       Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
      //       s[j] = Bt[i] & ((1ULL << MAX) - 1);
      //     }
      //     for (int j = 0; j < m; j++)
      //     {
      //       pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
      //     } 
      //   }
      //   delete[] s;
      //   Iot[0]->send_data(Rt, lt*m*sizeof(uint64_t));
      // }
      // delete[] Bt;
      // delete[] Rt;
      SS_L[k1 * len2 + k2] = SS_Lt[k1 * len2t + k2];
      SS_Skyline[k1 * len2 + k2] = new uint64_t[SS_Lt[k1 * len2t + k2]];
      memcpy(SS_Skyline[k1 * len2 + k2], SS_Skylinet[k1 * len2t + k2], SS_Lt[k1 * len2t + k2] * sizeof(uint64_t));
    }
  }
  // vector<uint32_t> Q(q.size());
  // uint32_t respos = 0;
  // SSQ(q, Q);
  // SS_Result(Q, respos);
  //     int th = 0;
  //     unordered_map<uint32_t, uint64_t*> Result;
  //       // dynamic points
  //     // DP(SS_Lt[k1 * len2t + k2], SS_Skylinet[k1 * len2t + k2], qt, SS_L[k1 * len2 + k2], SS_Skyline[k1 * len2 + k2]);
  //     uint64_t *qt = new uint64_t[m];
  //     int k1 = (respos-(respos%SS_G[0][1]))/SS_G[0][1];
  //     int k2 = respos%SS_G[0][1];
  //     // qt[0] = SS_G[1][k1 - 1];
  //     // qt[1] = SS_G[2][k2 - 1];
  //     for (int j = 0; j < m; j++) {
  //       qt[j] =  Q[j] & mask; 
  //     }
  //     int lt = SS_L[respos];
  //     cout<<k1<<"\t"<<k2<<"\t"<<lt<<endl;
  //     //Cout q
  //     // if (party == ALICE)
  //     // {
  //     //   Iot[th]->send_data(qt, m * sizeof(uint64_t));
  //     //   uint64_t *qt0 = new uint64_t[m];
  //     //   Iot[th]->recv_data(qt0, m*sizeof(uint64_t));
  //     //   for (int i = 0; i < m; i++)
  //     //   {
  //     //     qt0[i] = (qt0[i] + qt[i]) & mask;
  //     //     cout<<qt0[i]<<"\t";
  //     //   }
  //     //   cout<<endl;
  //     //   delete[] qt0;
  //     // }
  //     // else{
  //     //   uint64_t *qt0 = new uint64_t[m];
  //     //   Iot[th]->recv_data(qt0, m*sizeof(uint64_t));
  //     //   Iot[th]->send_data(qt, m * sizeof(uint64_t));
  //     //   for (int i = 0; i < m; i++)
  //     //   {
  //     //     qt0[i] = (qt0[i] + qt[i]) & mask;
  //     //     cout<<qt0[i]<<"\t";
  //     //   }
  //     //   cout<<endl;
  //     //   delete[] qt0;
  //     // }
  //     uint64_t **pt = new uint64_t*[lt];
  //     uint64_t *Bt = new uint64_t[lt];
  //     uint64_t *Rt = new uint64_t[lt*m];
  //     prg.random_data(Rt, lt*m*sizeof(uint64_t));
  //     if (party == ALICE)
  //     {
  //       for (int j = 0; j < lt*m; j++)
  //       {
  //         Rt[j] = Rt[j] & ((1ULL << (MAX-2)) - 1);
  //       }
  //       for (int i = 0; i < lt; i++)
  //       {
  //         Bt[i] = (SS_Skyline[respos][i] + (Rt[i] << MAX) + Rt[i+lt]) & mask;
  //       }
  //       Iot[th]->send_data(Bt, lt * sizeof(uint64_t));
  //       uint64_t *Rt0 = new uint64_t[lt*m];
  //       Iot[th]->recv_data(Rt0, lt*m*sizeof(uint64_t));
  //       for (int i = 0; i < lt; i++)
  //       {
  //         pt[i] = new uint64_t[m];
  //         for (int j = 0; j < m; j++)
  //         {
  //           pt[i][j] = (Rt0[j*lt+i] - Rt[j*lt+i]) & mask;
  //         } 
  //         // pt[i][0] = (pt[i][0] - Rt[i]- (Rt[lt+i]-Rt[lt+i]%((1ULL << MAX) ))/(1ULL << MAX)) & mask;
  //         // pt[i][1] = (pt[i][1] - Rt[lt+i]%((1ULL << MAX) )) & mask;
  //       }
  //       delete[] Rt0;
  //     }
  //     else
  //     {
  //       Iot[th]->recv_data(Bt, lt * sizeof(uint64_t));
  //       for (int i = 0; i < lt; i++)
  //       {
  //         Bt[i] = (Bt[i] + SS_Skyline[respos][i]) & mask;
  //       }
  //       uint64_t *s = new uint64_t[m];
  //       for (int i = 0; i < lt; i++)
  //       {
  //         pt[i] = new uint64_t[m];
  //         s[0] = Bt[i] & ((1ULL << MAX) - 1);
  //         for (int j = 1; j < m; j++)
  //         {
  //           Bt[i] = (Bt[i] - s[j - 1]) >> MAX;
  //           s[j] = Bt[i] & ((1ULL << MAX) - 1);
  //         }
  //         for (int j = 0; j < m; j++)
  //         {
  //           pt[i][j] = (s[m-1-j] - Rt[j*lt+i]) & mask;
  //           // x[j*lt+i] = pt[i][j] & mask;
  //         } 
  //       }
  //       delete[] s;
  //       Iot[th]->send_data(Rt, lt*m*sizeof(uint64_t));   
  //     }
  //     delete[] Bt;
  //     delete[] Rt;
  //     //Euclid product
  //     uint64_t *S = new uint64_t[lt];
  //     uint64_t **St = new uint64_t*[lt];
  //     uint64_t *in1 = new uint64_t[lt*m];
  //     uint64_t *out1 = new uint64_t[lt*m];
  //     for (int i = 0; i < lt; i++) {
  //       for (int j = 0; j < m; j++) {
  //         in1[i*m+j] = (pt[i][j] - qt[j]) & mask; 
  //       }
  //     }
  //     Prod_H(lt*m, in1, in1, out1, Prodt[th]);
  //     for (int i = 0; i < lt; i++) {
  //       S[i] = 0;
  //       St[i] = new uint64_t[m];
  //       for (int j = 0; j < m; j++) {
  //         St[i][j] = out1[i*m+j];
  //         S[i] = (S[i] + out1[i*m+j]) & mask;  
  //       }
  //     }
  //     // if (party == ALICE)
  //     // {
  //     //   for (int i = 0; i < lt; i++) {
  //     //     Iot[th]->send_data(St[i], m * sizeof(uint64_t));
  //     //     uint64_t *S0 = new uint64_t[m];
  //     //     Iot[th]->recv_data(S0, m * sizeof(uint64_t));
  //     //     for (int j = 0; j < m; j++) {
  //     //       S0[j] = (S0[j] + St[i][j]) & mask; 
  //     //       cout<<S0[j]<<"\t";
  //     //     }
  //     //     cout<<endl;
  //     //     delete[] S0;
  //     //   }
  //     // }
  //     // else{
  //     //   for (int i = 0; i < lt; i++) {
  //     //     uint64_t *S0 = new uint64_t[m];
  //     //     Iot[th]->recv_data(S0, m * sizeof(uint64_t));
  //     //     Iot[th]->send_data(St[i], m * sizeof(uint64_t));
  //     //     for (int j = 0; j < m; j++) {
  //     //       S0[j] = (S0[j] + St[i][j]) & mask; 
  //     //       cout<<S0[j]<<"\t";
  //     //     }
  //     //     cout<<endl;
  //     //     delete[] S0;
  //     //   }
  //     // }
  //     // Cout S
  //     // if (party == ALICE)
  //     // {
  //     //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //     //   uint64_t *S0 = new uint64_t[lt];
  //     //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //     //   for (int i = 0; i < lt; i++)
  //     //   {
  //     //     S0[i] = (S0[i] + S[i]) & mask;
  //     //     cout<<S0[i]<<endl;
  //     //   }
  //     //   delete[] S0;
  //     // }
  //     // else{
  //     //   uint64_t *S0 = new uint64_t[lt];
  //     //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //     //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //     //   for (int i = 0; i < lt; i++)
  //     //   {
  //     //     S0[i] = (S0[i] + S[i]) & mask;
  //     //     cout<<S0[i]<<endl;
  //     //   }
  //     //   delete[] S0;
  //     // }
  //     // int logn = 0;
  //     // for (int i = 2; ; i++) {
  //     //   if((pow(2,i-1)<=lt)&&(pow(2,i)>lt)){
  //     //     logn = i;
  //     //     break;
  //     //   }
  //     // }
  //     // cout<<logn<<endl;
  //     // for (int i = 0; i < lt; i++) {
  //     //   if (party == ALICE)
  //     //   {
  //     //     S[i] = (S[i] << logn) + i;
  //     //   }
  //     //   else
  //     //   {
  //     //     S[i] = (S[i] << logn);
  //     //   }
  //     // }
  //     uint64_t TMAX = 1ULL << (2 * MAX + 1);
  //     if (party == ALICE)
  //     {
  //       uint64_t T0 = 0;
  //       prg.random_data(&T0, sizeof(uint64_t));
  //       TMAX = (TMAX - T0) & mask;
  //       Iot[th]->send_data(&T0, sizeof(uint64_t));
  //     }
  //     else
  //     {
  //       Iot[th]->recv_data(&TMAX, sizeof(uint64_t)); 
  //     }      
  //     uint64_t lam = 1;
  //     uint64_t SS_one = 0;
  //     uint64_t SS_zero = 0;
  //     prg.random_data(&SS_one, sizeof(uint64_t));
  //     prg.random_data(&SS_zero, sizeof(uint64_t));
  //     if (party == ALICE)
  //     {
  //       Iot[th]->send_data(&SS_one, sizeof(uint64_t));
  //       Iot[th]->send_data(&SS_zero, sizeof(uint64_t));
  //     }
  //     else
  //     {
  //       Iot[th]->recv_data(&SS_one, sizeof(uint64_t));
  //       SS_one = (1 - SS_one) & mask;
  //       Iot[th]->recv_data(&SS_zero, sizeof(uint64_t));
  //       SS_zero = (0 - SS_zero) & mask;
  //     }    
  //     while(lam!=0){
  //       uint64_t STMin = 0;
  //       SMIN(lt, S, th, STMin);
  //       //Cout STmin
  //       // if (party == ALICE)
  //       // {
  //       //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
  //       //   uint64_t STMin0 = 0;
  //       //   Iot[th]->recv_data(&STMin0, sizeof(uint64_t));
  //       //   cout<<((STMin0 + STMin) & mask)<<":Smin"<<endl;
  //       // }
  //       // else{
  //       //   uint64_t STMin0 = 0;
  //       //   Iot[th]->recv_data(&STMin0, sizeof(uint64_t));
  //       //   Iot[th]->send_data(&STMin, sizeof(uint64_t));
  //       //   cout<<((STMin0 + STMin) & mask)<<":Smin"<<endl;
  //       // }
  //       uint64_t *r = new uint64_t[1];
  //       prg.random_data(r, sizeof(uint64_t));
  //       uint64_t *in2 = new uint64_t[1];
  //       in2[0] = (STMin - TMAX) & mask;
  //       uint64_t *out2 = new uint64_t[1];
  //       Prod_H(1, in2, r, out2, Prodt[th]);
  //       if (party == ALICE)
  //       {
  //         Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
  //         Iot[th]->send_data(out2, sizeof(uint64_t));
  //         lam = (lam + out2[0]) & mask;
  //       }
  //       else
  //       {
  //         Iot[th]->send_data(out2, sizeof(uint64_t));
  //         Iot[th]->recv_data(&lam, sizeof(uint64_t)); 
  //         lam = (lam + out2[0]) & mask;
  //       }
  //       cout<<"lam:"<<lam<<endl;
  //       if(lam!=0){
  //         uint64_t *A = new uint64_t[lt];
  //         for (int i = 0; i < lt; i++) {
  //             A[i] = (S[i] - STMin) & mask;
  //         }
  //         uint64_t *r3 = new uint64_t[lt];
  //         prg.random_data(r3, lt * sizeof(uint64_t));
  //         uint64_t *out3 = new uint64_t[lt];
  //         Prod_H(lt, A, r3, out3, Prodt[th]);
  //         uint32_t *Pi = new uint32_t[lt];
  //         uint32_t *Pi_Inv = new uint32_t[lt];
  //         for (int i = 0; i < lt; i++) {
  //           Pi[i] = i;
  //         }
  //         for (int i = 0; i < lt; i++) {
  //           uint32_t randomPosition = 0;
  //           prg.random_data(&randomPosition, sizeof(uint32_t));
  //           randomPosition = randomPosition & (lt-1);
  //           uint32_t temp = Pi[i];
  //           Pi[i] = Pi[randomPosition];
  //           Pi[randomPosition] = temp;
  //         }
  //         for (int i = 0; i < lt; i++) {
  //           Pi_Inv[Pi[i]] = i;
  //         }
  //         if (party == ALICE)
  //         {
  //           Iot[th]->send_data(Pi, lt * sizeof(uint32_t));
  //           Iot[th]->send_data(Pi_Inv, lt * sizeof(uint32_t));
  //         }
  //         else
  //         {
  //           Iot[th]->recv_data(Pi, lt * sizeof(uint32_t));
  //           Iot[th]->recv_data(Pi_Inv, lt * sizeof(uint32_t));
  //         }
  //         uint64_t *B = new uint64_t[lt];
  //         for (int i = 0; i < lt; i++) {
  //           B[Pi[i]] = out3[i];
  //         }       
  //         // B
  //         uint64_t *b = new uint64_t[lt];
  //         uint64_t *U = new uint64_t[lt];
  //         if (party == ALICE)
  //         {
  //           Iot[th]->recv_data(b, lt * sizeof(uint64_t));
  //           Iot[th]->send_data(B, lt * sizeof(uint64_t));
  //           for (int i = 0; i < lt; i++) {
  //             b[i] = (b[i] + B[i]) & mask;
  //             U[i] = SS_zero;
  //             if (b[i]==0){
  //                 U[i] = SS_one;
  //             }
  //           }
  //         }
  //         else
  //         {
  //           Iot[th]->send_data(B, lt * sizeof(uint64_t));
  //           Iot[th]->recv_data(b, lt * sizeof(uint64_t));
  //           for (int i = 0; i < lt; i++) {
  //             b[i] = (b[i] + B[i]) & mask;
  //             U[i] = SS_zero;
  //             if (b[i]==0){
  //                 U[i] = SS_one;
  //             }
  //           }
  //         }
  //         //select the first one
  //         uint64_t phi2 = SS_zero;
  //         uint64_t *U1 = new uint64_t[1];
  //         uint64_t *U2 = new uint64_t[1];
  //         uint64_t *U3 = new uint64_t[1];
  //         for (int i = 0; i < lt; i++) {
  //           phi2 = (U[i] + phi2) & mask;
  //           // 1<=phi2
  //           if (party == ALICE)
  //           {
  //             U1[0] = (phi2 - 1) & mask;
  //           }
  //           else
  //           {
  //             U1[0] = (0 - phi2) & mask;
  //           }
  //           comparison(1, U1, U2, Auxt[th]);
  //           U1[0] = U[i];
  //           U2[0] = (SS_one - U2[0]) & mask;
  //           Prod_H(1, U1, U2, U3, Prodt[th]);
  //           U[i] = U3[0];
  //         }         
  //         // A
  //         uint64_t *V = new uint64_t[lt];
  //         for(int i = 0; i < lt; i++) {
  //             V[Pi_Inv[i]] = U[i];
  //         }
  //         // Cout V
  //         // if (party == ALICE)
  //         // {
  //         //   Iot[th]->send_data(V, lt * sizeof(uint64_t));
  //         //   uint64_t *V0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(V0, lt*sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     V0[i] = (V0[i] + V[i]) & mask;
  //         //     cout<<V0[i]<<"\t";
  //         //   }
  //         //   cout<<endl;
  //         //   delete[] V0;
  //         // }
  //         // else{
  //         //   uint64_t *V0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(V0, lt*sizeof(uint64_t));
  //         //   Iot[th]->send_data(V, lt * sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     V0[i] = (V0[i] + V[i]) & mask;
  //         //     cout<<V0[i]<<"\t";
  //         //   }
  //         //   cout<<endl;
  //         //   delete[] V0;
  //         // }
  //         uint64_t *Pmin = new uint64_t[m];
  //         uint64_t *Tmin = new uint64_t[m];
  //         uint64_t *in4 = new uint64_t[1];
  //         uint64_t *in5 = new uint64_t[1];
  //         uint64_t *out4 = new uint64_t[1];         
  //         for (int j = 0; j < m; j++) {
  //           Pmin[j] = SS_zero;
  //           Tmin[j] = SS_zero;
  //           for (int i = 0; i < lt; i++) {
  //             in4[0] = V[i];
  //             in5[0] = pt[i][j];
  //             Prod_H(1, in4, in5, out4, Prodt[th]); 
  //             Pmin[j] = (Pmin[j] + out4[0]) & mask;
  //             in5[0] = St[i][j];
  //             Prod_H(1, in4, in5, out4, Prodt[th]);
  //             Tmin[j] = (Tmin[j] + out4[0]) & mask;
  //           }
  //         }
  //         // A
  //         int pos = Result.size();
  //         // cout<<pos<<endl;
  //         Result[pos] = Pmin;
  //         //eliminate
  //         for (int i = 0; i < lt; i++) {
  //           in4[0] = V[i];
  //           in5[0] = (TMAX - S[i]) & mask;
  //           Prod_H(1, in4, in5, out4, Prodt[th]);
  //           S[i] = (out4[0] + S[i]) & mask;
  //         }
  //         // Cout S
  //         // if (party == ALICE)
  //         // {
  //         //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //         //   uint64_t *S0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     S0[i] = (S0[i] + S[i]) & mask;
  //         //     cout<<S0[i]<<endl;
  //         //   }
  //         //   delete[] S0;
  //         // }
  //         // else{
  //         //   uint64_t *S0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //         //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     S0[i] = (S0[i] + S[i]) & mask;
  //         //     cout<<S0[i]<<endl;
  //         //   }
  //         //   delete[] S0;
  //         // }
  //         for(int i = 0; i < lt; i++) {
  //             SDOMbyMin(m, Tmin, St[i], th, V[i]);
  //         }
  //         // Cout V
  //         // if (party == ALICE)
  //         // {
  //         //   Iot[th]->send_data(V, lt * sizeof(uint64_t));
  //         //   uint64_t *V0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(V0, lt*sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     V0[i] = (V0[i] + V[i]) & mask;
  //         //     cout<<V0[i]<<"\t";
  //         //   }
  //         //   cout<<endl;
  //         //   delete[] V0;
  //         // }
  //         // else{
  //         //   uint64_t *V0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(V0, lt*sizeof(uint64_t));
  //         //   Iot[th]->send_data(V, lt * sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     V0[i] = (V0[i] + V[i]) & mask;
  //         //     cout<<V0[i]<<"\t";
  //         //   }
  //         //   cout<<endl;
  //         //   delete[] V0;
  //         // }
  //         for (int i = 0; i < lt; i++) {
  //           in4[0] = V[i];
  //           in5[0] = (TMAX - S[i]) & mask;
  //           Prod_H(1, in4, in5, out4, Prodt[th]);
  //           S[i] = (out4[0] + S[i]) & mask;
  //         }
  //         // Cout S
  //         // if (party == ALICE)
  //         // {
  //         //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //         //   uint64_t *S0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     S0[i] = (S0[i] + S[i]) & mask;
  //         //     cout<<S0[i]<<"\t"<<S[i]<<endl;
  //         //   }
  //         //   delete[] S0;
  //         // }
  //         // else{
  //         //   uint64_t *S0 = new uint64_t[lt];
  //         //   Iot[th]->recv_data(S0, lt*sizeof(uint64_t));
  //         //   Iot[th]->send_data(S, lt * sizeof(uint64_t));
  //         //   for (int i = 0; i < lt; i++)
  //         //   {
  //         //     S0[i] = (S0[i] + S[i]) & mask;
  //         //     cout<<S0[i]<<"\t"<<S[i]<<endl;
  //         //   }
  //         //   delete[] S0;
  //         // }
  //         delete[] A;
  //         delete[] B;
  //         delete[] b;
  //         delete[] r3;
  //         delete[] out3;
  //         delete[] in4;
  //         delete[] in5;
  //         delete[] out4;
  //         delete[] U;
  //         delete[] U1;
  //         delete[] U2;
  //         delete[] U3;
  //         delete[] V;
  //         // delete[] Pmin;
  //         delete[] Tmin;
  //         delete[] Pi;
  //         delete[] Pi_Inv;
  //       }
  //       delete[] r;
  //       delete[] in2;
  //       delete[] out2;
  //     }
  //     // Return
  //     // A
  //     int k = Result.size();
  //     cout<<k<<endl;
  //     uint64_t *res = new uint64_t[k*m];
  //     for (int i = 0; i < k; i++)
  //     {
  //       for (int j = 0; j < m; j++)
  //       {
  //         res[i*m+j] = Result[i][j] & mask;
  //       }
  //       // memcpy(res)
  //     }
  //     if (party == ALICE)
  //     {
  //       Iot[th]->send_data(res, k * m * sizeof(uint64_t));
  //       uint64_t *Rt0 = new uint64_t[k*m];
  //       Iot[0]->recv_data(Rt0, k * m * sizeof(uint64_t));
  //       for (int i = 0; i < k; i++)
  //       {
  //         for (int j = 0; j < m; j++)
  //         {
  //           uint64_t tt = (Rt0[i*m+j] + res[i*m+j]) & mask;
  //           cout<<tt<<"\t";
  //         } 
  //         cout<<endl;
  //       }
  //       delete[] Rt0;
  //     }
  //     else
  //     {
  //       uint64_t *Rt0 = new uint64_t[k*m];
  //       Iot[0]->recv_data(Rt0, k * m * sizeof(uint64_t));
  //       Iot[th]->send_data(res, k * m * sizeof(uint64_t));
  //       for (int i = 0; i < k; i++)
  //       {
  //         for (int j = 0; j < m; j++)
  //         {
  //           uint64_t tt = (Rt0[i*m+j] + res[i*m+j]) & mask;
  //           cout<<tt<<"\t";
  //         } 
  //         cout<<endl;
  //       }
  //       delete[] Rt0;
  //     }
  //     delete[] res;
  //     for(int j = 0; j< lt; j++)
  //     {
  //       delete[] St[j];
  //       delete[] pt[j];
  //     }
  //     delete[] St;
  //     delete[] pt;
  // delete[] SS_Con;
  // delete[] SS_C;
  // delete[] SS_L;
  // for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  // {
  //   delete[] SS_Skyline[j];
  // }
  // delete[] SS_Skyline;
  // for(int j = 0; j< m; j++)
  // {
  //   delete[] SS_G[j];
  // }
  // delete[] SS_G;
  delete[] SS_Lt;
  for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  {
    delete[] SS_Skylinet[j];
  }
  delete[] SS_Skylinet;
  unordered_set<uint32_t>().swap(H);
  H.clear();
  unordered_map<uint32_t, vector<uint32_t>>().swap(G);
  G.clear();
  unordered_map<uint32_t, unordered_set<uint32_t>>().swap(Skyline);
  Skyline.clear();
}

void dynamicSkylineR(unordered_map<uint32_t, vector<uint32_t>> &G){
  unordered_map<uint32_t, vector<uint32_t>> G1(G);
  unordered_map<uint32_t, vector<uint32_t>> G2(G);
  unordered_map<uint32_t, vector<uint32_t>> G4(G);
  int len1 = G[1].size();
  int len2 = G[2].size();
  unordered_map<uint32_t, vector<uint32_t>> G3;
  for (int j = 0; j < m; j++)
  {
    vector<uint32_t> tmp;
    int len = G[j+1].size();
    for (int k = 0; k < len; k++)
    {
      tmp.push_back(G[j+1][len-1-k]);
    }
    G3[j + 1] = tmp;
  }
  G2[1] = G3[1];
  G4[2] = G3[2];
  //dynamic borders add one row/column
  // Read
  int len1t = len1 + 1;
  int len2t = len2 + 1;
  SS_L_itr = new int[len1t * len2t];
  SS_Skyline_itr = new uint64_t*[len1t * len2t];
  double sr = omp_get_wtime();
  for (int i = 0; i < 4; i++)
  {
    string path = "-";
    if(i == 0){
      G = G1;
      path = path + "1";
    }else if (i == 1){
      G = G2;
      path = path + "2";
    }else if (i == 2){
       G = G3;
      path = path + "3";
    }else{
       G = G4;
      path = path + "4";
    }
    ReadSKyline(path);
    SSG(G, SS_G);
    if(i == 0){
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          SS_L_itr[k1 * len2t + k2] = SS_L[k1 * len2 + k2];
          SS_Skyline_itr[k1 * len2t + k2] = new uint64_t[SS_L[k1 * len2 + k2]];
          memcpy(SS_Skyline_itr[k1 * len2t + k2], SS_Skyline[k1 * len2 + k2], SS_L[k1 * len2 + k2] * sizeof(uint64_t));
        }
      }
      // SSG(G1, SS_G);
      // vector<uint32_t> Q(q.size());
      // SSQ(q, Q);
      // SS_Result(q, Q);
    }else if (i == 1){  //  right move one column
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        SS_Skyline_itr[len1 * len2t + k2] = new uint64_t[SS_L[0 * len2 + k2]];
        memcpy(SS_Skyline_itr[len1 * len2t + k2], SS_Skyline[0 * len2 + k2], SS_L[0 * len2 + k2] * sizeof(uint64_t));
        SS_L_itr[len1 * len2t + k2] = SS_L[0 * len2 + k2];
      }
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 2; k1++)
        {
          int k1t = len1 - 1 - k1;
          uint64_t *tmpSkyline = new uint64_t[SS_L_itr[(k1 + 1) * len2t + k2]];
          memcpy(tmpSkyline, SS_Skyline_itr[(k1 + 1) * len2t + k2], SS_L_itr[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
          delete[] SS_Skyline_itr[(k1 + 1) * len2t + k2];
          SS_Skyline_itr[(k1 + 1) * len2t + k2] = new uint64_t[SS_L[k1t * len2 + k2] + SS_L_itr[(k1 + 1) * len2t + k2]];
          memcpy(SS_Skyline_itr[(k1 + 1) * len2t + k2], tmpSkyline, SS_L_itr[(k1 + 1) * len2t + k2] * sizeof(uint64_t));
          memcpy(SS_Skyline_itr[(k1 + 1) * len2t + k2] + SS_L_itr[(k1 + 1) * len2t + k2], SS_Skyline[k1t * len2 + k2], SS_L[k1t * len2 + k2] * sizeof(uint64_t));
          SS_L_itr[(k1 + 1) * len2t + k2] = SS_L[k1t * len2 + k2] + SS_L_itr[(k1 + 1) * len2t + k2];
          delete[] tmpSkyline;
        }
      }
    }else if (i == 2){  // right move one column, up move one row
      for (int k1 = 0; k1 <= len1 - 1; k1++)
      {
        SS_Skyline_itr[(k1 + 1) * len2t + len2] = new uint64_t[SS_L[(len1 - 1 - k1) * len2 + 0]];
        memcpy(SS_Skyline_itr[(k1 + 1) * len2t + len2], SS_Skyline[(len1 - 1 - k1) * len2 + 0], SS_L[(len1 - 1 - k1) * len2 + 0] * sizeof(uint64_t));
        SS_L_itr[(k1 + 1) * len2t + len2] = SS_L[(len1 - 1 - k1) * len2 + 0];
      }
      for (int k2 = len2 - 2; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          int k1t = len1 - 1 - k1;
          int k2t = len2 - 1 - k2;
          uint64_t *tmpSkyline = new uint64_t[SS_L_itr[(k1 + 1) * len2t + (k2 + 1)]];
          memcpy(tmpSkyline, SS_Skyline_itr[(k1 + 1) * len2t + (k2 + 1)], SS_L_itr[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
          delete[] SS_Skyline_itr[(k1 + 1) * len2t + (k2 + 1)];
          SS_Skyline_itr[(k1 + 1) * len2t + (k2 + 1)] = new uint64_t[SS_L[k1t * len2 + k2t] + SS_L_itr[(k1 + 1) * len2t + (k2 + 1)]];
          memcpy(SS_Skyline_itr[(k1 + 1) * len2t + (k2 + 1)], tmpSkyline, SS_L_itr[(k1 + 1) * len2t + (k2 + 1)] * sizeof(uint64_t));
          memcpy(SS_Skyline_itr[(k1 + 1) * len2t + (k2 + 1)] + SS_L_itr[(k1 + 1) * len2t + (k2 + 1)], SS_Skyline[k1t * len2 + k2t], SS_L[k1t * len2 + k2t] * sizeof(uint64_t));
          SS_L_itr[(k1 + 1) * len2t + (k2 + 1)] = SS_L[k1t * len2 + k2t] + SS_L_itr[(k1 + 1) * len2t + (k2 + 1)];
          delete[] tmpSkyline;
        }
      }
    }else{  //  up move one row
      SS_Skyline_itr[0 * len2t + len2] = new uint64_t[SS_L[0 * len2 + 0]];
      memcpy(SS_Skyline_itr[0 * len2t + len2], SS_Skyline[0 * len2 + 0], SS_L[0 * len2 + 0] * sizeof(uint64_t));
      SS_L_itr[0 * len2t + len2] = SS_L[0 * len2 + 0];
      for (int k2 = len2 - 1; k2 >= 0; k2--)
      {
        for (int k1 = 0; k1 <= len1 - 1; k1++)
        {
          if((k2 == len2 - 1)&&(k1 == 0)){
            continue;
          }
          int k2t = len2 - 1 - k2;
          uint64_t *tmpSkyline = new uint64_t[SS_L_itr[k1 * len2t + (k2 + 1)]];
          memcpy(tmpSkyline, SS_Skyline_itr[k1 * len2t + (k2 + 1)], SS_L_itr[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
          delete[] SS_Skyline_itr[k1 * len2t + (k2 + 1)];
          SS_Skyline_itr[k1 * len2t + (k2 + 1)] = new uint64_t[SS_L[k1 * len2 + k2t] + SS_L_itr[k1 * len2t + (k2 + 1)]];
          memcpy(SS_Skyline_itr[k1 * len2t + (k2 + 1)], tmpSkyline, SS_L_itr[k1 * len2t + (k2 + 1)] * sizeof(uint64_t));
          memcpy(SS_Skyline_itr[k1 * len2t + (k2 + 1)] + SS_L_itr[k1 * len2t + (k2 + 1)], SS_Skyline[k1 * len2 + k2t], SS_L[k1 * len2 + k2t] * sizeof(uint64_t));
          SS_L_itr[k1 * len2t + (k2 + 1)] = SS_L[k1 * len2 + k2t] + SS_L_itr[k1 * len2t + (k2 + 1)];
          delete[] tmpSkyline;
        }
      }
    }
  }
  double er = omp_get_wtime();
  ss_read = er - sr;
  SSG(G1, SS_G);
  unordered_set<uint32_t>().swap(H);
  H.clear();
  unordered_map<uint32_t, vector<uint32_t>>().swap(G);
  G.clear();
  unordered_map<uint32_t, unordered_set<uint32_t>>().swap(Skyline);
  Skyline.clear();
}

void test_Parrllel()
{
  double ttt1 = omp_get_wtime();
  omp_set_nested(8);
  for (int ee = 0; ee < 10; ee++)
  {
#pragma omp parallel
    {
#pragma omp sections
      {
#pragma omp section
        {
          cout << "T1 = " << omp_get_thread_num() << endl;
          string ss = "=1111=";
#pragma omp parallel
          {
#pragma omp sections
            {
#pragma omp section
              {
                double t1 = omp_get_wtime();
                cout << "11 = " << ss << omp_get_thread_num() << endl;
                for (int i = 0; i < 10; i++)
                {
                  string ss = "-1:" + to_string(i) + "-";
                  cout << ss;
                }

                double t2 = omp_get_wtime();
                cout << "test" << (t2 - t1) << endl;
              }
#pragma omp section
              {
                cout << "12 = " << omp_get_thread_num() << endl;
                for (int i = 0; i < 10; i++)
                {
                  string ss = "-2:" + to_string(i) + "-";
                  cout << ss;
                }
              }
            }
          }
        }
#pragma omp section
        {
          cout << "T2 = " << omp_get_thread_num() << endl;
          string ss = "=2222=";
          int a = 0;
          for (int j = 0; j < 3; j++)
          {
            int n = j + 5;
            cout << "a=" << a << "**" << endl;
#pragma omp parallel
            {
#pragma omp sections
              {
#pragma omp section
                {
                  cout << "21 = " << ss << omp_get_thread_num() << endl;
                  for (int i = 0; i < n; i++)
                  {
                    string ss = "-3:" + to_string(i) + "-";
                    cout << j << "*" << ss;
                  }
                  if (j == 0)
                  {
                    a = 1;
                    cout << "$$$$$$$$$$$$$$$$$";
                  }
                  if (j == 1)
                  {

                    cout << "@@@@@@@@@@@@@@@@@";
                  }
                  if (j == 2)
                  {
                    cout << "#################";
                  }
                }
#pragma omp section
                {
                  cout << "22 = " << omp_get_thread_num() << endl;
                  for (int i = 0; i < n; i++)
                  {
                    string ss = "-4:" + to_string(i) + "-";
                    cout << j << "*" << ss;
                  }
                  if (j == 0)
                  {
                    cout << "$$$$$$$$$$$$$$$$$";
                  }
                  if (j == 1)
                  {
                    a = 2;
                    cout << "@@@@@@@@@@@@@@@@@";
                  }
                  if (j == 2)
                  {

                    cout << "#################";
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  double ttt2 = omp_get_wtime();
  cout << "test" << (ttt2 - ttt1) << endl;
}

void test_SkylineGen_1(){
  uint64_t *tmp1 = new uint64_t[2]();
  uint64_t *tmp2 = new uint64_t[3]();
  uint64_t *tmp3 = new uint64_t[2]();
  prg.random_data(tmp1, 2 * sizeof(uint64_t));
  prg.random_data(tmp2, 3 * sizeof(uint64_t));
  tmp2[0] = 0;
  tmp2[1] = tmp1[1];
  tmp3[0] = tmp1[0];
  tmp3[1] = tmp2[1];
  int num = 0;
  uint64_t *res;
  SkylineGen_1(tmp1, tmp2, tmp3, 2,3,2,num,res,Auxt[0],Prodt[0],Iot[0]);
  SkylineGen_1(tmp1, tmp2, tmp3, 2,3,2,num,res,Auxt[1],Prodt[1],Iot[1]);
}

void test_Cmp(){
  int t = 1000;
  uint64_t yt = (1ULL << lambda);
  uint64_t *in1 = new uint64_t[t];
  uint8_t *fin1 = new uint8_t[t];
  uint8_t *fin2 = new uint8_t[t];
  uint8_t *fou1 = new uint8_t[t];
  uint8_t *fou2 = new uint8_t[t];
  prg.random_data(fin1, t * sizeof(uint8_t));
  prg.random_data(fin2, t * sizeof(uint8_t));
  for (int i = 0; i < t; i++) {
    fin1[i] = fin1[i] & 1;
    fin2[i] = fin2[i] & 1;
  }
  uint64_t *fout1 = new uint64_t[t];
  uint64_t *outT = new uint64_t[2*t];
  uint64_t *out = new uint64_t[2*t];
  uint8_t *outTT = new uint8_t[2*t];
  uint8_t *outTTT = new uint8_t[2*t];
  uint64_t *fout2 = new uint64_t[t];
  uint64_t *in2 = new uint64_t[t];
  uint64_t *in3 = new uint64_t[t];
  uint64_t *C = new uint64_t[t];
  uint64_t *C1 = new uint64_t[t];
  uint64_t *C2 = new uint64_t[t];
  uint64_t *C3 = new uint64_t[t];
  uint64_t *x = new uint64_t[t];
  uint64_t *x1 = new uint64_t[t];
  uint64_t *xx = new uint64_t[t];
  prg.random_data(xx, t * sizeof(uint64_t));
  uint64_t *y = new uint64_t[t];
  uint64_t *y1 = new uint64_t[t];
  uint64_t *yy = new uint64_t[t];
  prg.random_data(yy, t * sizeof(uint64_t));
  uint64_t mask = (1ULL << lambda) - 1;
  if (party == ALICE)
  {
    Iot[0]->recv_data(xx, t * sizeof(uint64_t));
    Iot[1]->recv_data(yy, t * sizeof(uint64_t));    
  }
  else
  {
    Iot[0]->send_data(xx, t * sizeof(uint64_t));
    Iot[1]->send_data(yy, t * sizeof(uint64_t));
  }  
  uint64_t tt = 2 * 10000 * 10000;
  // tt = 1ULL << 20; 
  for (int j = 0; j < t; j++) {
    xx[j] = xx[j] % tt;
    yy[j] = yy[j] % tt;
    if(xx[j]>=yy[j]){
      C[j] = 1;
    }else {
      C[j] = 0;
    }
    // cout<<xx[j]<< ":" <<yy[j]<<"="<<(C[j])<<",";
  }
  cout<<endl;
  for (int j = 0; j < t; j++) {
    // cout<<C[j]<<",";
  }
  cout<<endl;
  if (party == ALICE)
  {
    Iot[0]->recv_data(x, t * sizeof(uint64_t));
    Iot[1]->recv_data(y, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      x[j] = (x[j]) & mask;
      y[j] = (y[j]) & mask;
    }
    Iot[0]->recv_data(x1, t * sizeof(uint64_t));
    Iot[1]->recv_data(y1, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      x1[j] = (x1[j]) & mask;
      y1[j] = (y1[j]) & mask;
    }
    for (int j = 0; j < t; j++) {
      // cout<<((x[j] + x1[j])& mask)<<" "<<xx[j]<<endl;
      if(((x[j] + x1[j])& mask)!=xx[j]){
        cout<<"!!!!!!!"<<endl;
      }
    }
    for (int j = 0; j < t; j++) {
      // cout<<((y[j] + y1[j])& mask)<<" "<<yy[j]<<endl;
      if(((y[j] + y1[j])& mask)!=yy[j]){
        cout<<"@@@@@@"<<endl;
      }
    }
  }
  else
  {
    prg.random_data(x1, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      x1[j] = (x1[j]) & mask;
    }
    Iot[0]->send_data(x1, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      x[j] = (xx[j] - x1[j]) & mask;
    }
    prg.random_data(y1, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      y1[j] = (y1[j]) & mask;
    }
    Iot[1]->send_data(y1, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      y[j] = (yy[j] - y1[j]) & mask;
    }
    Iot[0]->send_data(x, t * sizeof(uint64_t));
    Iot[1]->send_data(y, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      // cout<<((x[j] + x1[j])& mask)<<" "<<xx[j]<<endl;
      if(((x[j] + x1[j])& mask)!=xx[j]){
        cout<<"!!!!!!!"<<endl;
      }
    }
    for (int j = 0; j < t; j++) {
      // cout<<((y[j] + y1[j])& mask)<<" "<<yy[j]<<endl;
      if(((y[j] + y1[j])& mask)!=yy[j]){
        cout<<"@@@@@@"<<endl;
      }
    }
  }  

  for (int j = 0; j < t; j++) {
    if (party == ALICE)
    {
      if (x[j] < y[j])
      {
        in1[j] = (x[j] - y[j]) & mask; //-2>-3==(mask-2)>(mask-3)
        if((y[j]) >(yt/2)){
          in1[j] += yt;
        }
      }
      else
      {
        in1[j] = yt + ((x[j] - y[j]) & mask); // add the same 1
      }
      if (x[j] < y[j]){
        in2[j] = (x[j] - y[j]) & mask;
      }else{
        in2[j] = yt + ((x[j] - y[j]) & mask);
      }
      // if (in2[j]<=(yt/2))
      // {
        // in2[j] = yt + in2[j];
      // }
      in3[j] = yt + x[j] - y[j];
    }
    else
    {
      if (y[j] < x[j])
      {
        in1[j] = (y[j] - x[j]) & mask;
        if((x[j]) >(yt/2)){
          in1[j] += yt;
        }
      }
      else
      {
        in1[j] = yt + ((y[j] - x[j]) & mask);
      }
      if (y[j] < x[j]){
        in2[j] = (y[j] - x[j]) & mask;
      }else{
        in2[j] = yt + ((y[j] - x[j]) & mask);
      }
      // if (in2[j]<=(yt/2))
      // {
        // in2[j] = yt + in2[j];
      // }
      in3[j] = yt + y[j] - x[j];
    }
  }
  cout<<yt<<endl;
  // comparison_with_eq(t, in1, C1, Auxt[1]);
  // comparison_with_eq(t, in2, C2, Auxt[2]);
  // comparison_with_eq(t, in3, C3, Auxt[3]);
  comparison_with_eq2N(t, x, y, C2, Auxt[2]);
  // comparison_with_eq2N(t, x, y, C2, out,outT, outTT, outTTT, Auxt[2]);
  if (party == ALICE)
  {
    uint64_t *tC1 =  new uint64_t[t]; 
    uint64_t *tC2 =  new uint64_t[t]; 
    uint64_t *tC3 =  new uint64_t[t]; 
    Iot[0]->recv_data(tC1, t * sizeof(uint64_t));
    Iot[1]->recv_data(tC2, t * sizeof(uint64_t));
    Iot[2]->recv_data(tC3, t * sizeof(uint64_t));
    Iot[0]->send_data(C1, t * sizeof(uint64_t));
    Iot[1]->send_data(C2, t * sizeof(uint64_t));
    Iot[2]->send_data(C3, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      C1[j] = (C1[j] + tC1[j]) & mask;
      // cout<< (C1[j])<<",";
    }
    cout<<endl;
    for (int j = 0; j < t; j++) {
      C2[j] = (C2[j] + tC2[j]) & mask;
      // cout<< (C2[j])<<",";
    }
    cout<<endl;
    for (int j = 0; j < t; j++) {
      C3[j] = (C3[j] + tC3[j]) & mask;
      // cout<< (C3[j])<<",";
    }
    cout<<endl;
    delete[] tC1;
    delete[] tC2;
    delete[] tC3;
  }
  else
  {
    uint64_t *tC1 =  new uint64_t[t]; 
    uint64_t *tC2 =  new uint64_t[t]; 
    uint64_t *tC3 =  new uint64_t[t]; 
    Iot[0]->send_data(C1, t * sizeof(uint64_t));
    Iot[1]->send_data(C2, t * sizeof(uint64_t));
    Iot[2]->send_data(C3, t * sizeof(uint64_t));
    Iot[0]->recv_data(tC1, t * sizeof(uint64_t));
    Iot[1]->recv_data(tC2, t * sizeof(uint64_t));
    Iot[2]->recv_data(tC3, t * sizeof(uint64_t));
    for (int j = 0; j < t; j++) {
      C1[j] = (C1[j] + tC1[j]) & mask;
      // cout<< (C1[j])<<",";
    }
    cout<<endl;
    for (int j = 0; j < t; j++) {
      C2[j] = (C2[j] + tC2[j]) & mask;
      cout<< (C2[j])<<",";
    }
    cout<<endl;
    for (int j = 0; j < t; j++) {
      C3[j] = (C3[j] + tC3[j]) & mask;
      // cout<< (C3[j])<<",";
    }
    cout<<endl;
    delete[] tC1;
    delete[] tC2;
    delete[] tC3;
  }  
  
  // cout<<"11"<<endl;
  // Auxt[1]->AND(fin1, fin2, fou1, t);
  // if (party == ALICE)
  // {
  //   uint8_t *tC1 =  new uint8_t[t]; 
  //   uint8_t *tC2 =  new uint8_t[t]; 
  //   uint8_t *tC3 =  new uint8_t[t];
  //   uint8_t *tC4 =  new uint8_t[t]; 
  //   uint8_t *tC5 =  new uint8_t[t]; 
  //   Iot[0]->recv_data(tC1, t * sizeof(uint8_t));
  //   Iot[1]->recv_data(tC2, t * sizeof(uint8_t));
  //   Iot[2]->recv_data(tC3, t * sizeof(uint8_t));
  //   Iot[0]->send_data(fin1, t * sizeof(uint8_t));
  //   Iot[1]->send_data(fin2, t * sizeof(uint8_t));
  //   Iot[2]->send_data(fou1, t * sizeof(uint8_t));
  //   for (int j = 0; j < t; j++) {
  //     tC1[j] = (fin1[j] ^ tC1[j]);
  //     // cout<< +tC1[j]<<",";
  //   }
  //   cout<<"a"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC2[j] = (fin2[j] ^ tC2[j]);
  //     // cout<< +(tC2[j])<<",";
  //   }
  //   cout<<"b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC3[j] = (fou1[j] ^ tC3[j]);
  //     // cout<< +(tC3[j])<<",";
  //   }
  //   cout<<"a^b"<<endl;
  //   int k1=0,k2=0;
  //   for (int j = 0; j < t; j++) {
  //     tC4[j] = (tC1[j] & tC2[j]);
  //     // cout<< +(tC4[j])<<",";
  //     if(tC4[j]!=tC3[j]){
  //       k1++;
  //     }
  //   }
  //   cout<<"a^b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC5[j] = (tC1[j] ^ tC2[j]);
  //     // cout<< +(tC5[j])<<",";
  //   }
  //   cout<<"a or b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     // cout<< +(tC1[j] + tC2[j] - 2 * tC3[j])<<",";
  //     if(tC5[j]!=(tC1[j] + tC2[j] - 2 * tC3[j])){
  //       k2++;
  //     }
  //   }
  //   cout<<"a or b"<<endl;
  //   cout<<k1<<","<<k2<<endl;
  //   delete[] tC1;
  //   delete[] tC2;
  //   delete[] tC3;
  // }
  // else
  // {
  //   uint8_t *tC1 =  new uint8_t[t]; 
  //   uint8_t *tC2 =  new uint8_t[t]; 
  //   uint8_t *tC3 =  new uint8_t[t]; 
  //   uint8_t *tC4 =  new uint8_t[t];
  //   uint8_t *tC5 =  new uint8_t[t];
  //   Iot[0]->send_data(fin1, t * sizeof(uint8_t));
  //   Iot[1]->send_data(fin2, t * sizeof(uint8_t));
  //   Iot[2]->send_data(fou1, t * sizeof(uint8_t));
  //   Iot[0]->recv_data(tC1, t * sizeof(uint8_t));
  //   Iot[1]->recv_data(tC2, t * sizeof(uint8_t));
  //   Iot[2]->recv_data(tC3, t * sizeof(uint8_t));
  //   for (int j = 0; j < t; j++) {
  //     tC1[j] = (fin1[j] ^ tC1[j]);
  //     // cout<< +tC1[j]<<",";
  //   }
  //   cout<<"a"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC2[j] = (fin2[j] ^ tC2[j]);
  //     // cout<< +(tC2[j])<<",";
  //   }
  //   cout<<"b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC3[j] = (fou1[j] ^ tC3[j]);
  //     // cout<< +(tC3[j])<<",";
  //   }
  //   cout<<"a^b"<<endl;
  //   int k1=0,k2=0;
  //   for (int j = 0; j < t; j++) {
  //     tC4[j] = (tC1[j] & tC2[j]);
  //     // cout<< +(tC4[j])<<",";
  //     if(tC4[j]!=tC3[j]){
  //       k1++;
  //     }
  //   }
  //   cout<<"a^b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     tC5[j] = (tC1[j] ^ tC2[j]);
  //     // cout<< +(tC5[j])<<",";
  //   }
  //   cout<<"a or b"<<endl;
  //   for (int j = 0; j < t; j++) {
  //     // cout<< +(tC1[j] + tC2[j] - 2 * tC3[j])<<",";
  //     if(tC5[j]!=(tC1[j] + tC2[j] - 2 * tC3[j])){
  //       k2++;
  //     }
  //   }
  //   cout<<"a or b"<<endl;
  //   cout<<k1<<","<<k2<<endl;
  //   delete[] tC1;
  //   delete[] tC2;
  //   delete[] tC3;
  // }  
 

  int k1=0,k2=0,k3=0;
  for (int j = 0; j < t; j++) {
    if (C[j] != C2[j]){
      cout<<j<<":"<<x[j]<<"-"<<y[j]<<":"<<y1[j]<<"-"<<x1[j]<<endl;
      cout<<"="<<(x[j])<<":"<<(mask + 1 - x1[j])<<"="<<(x[j]>=(mask + 1 - x1[j]));
      cout<<"="<<(y[j])<<":"<<(mask + 1 - y1[j])<<"="<<(y[j]>=(mask + 1 - y1[j]));
      cout<<"#"<<C[j]<<":"<<C2[j];
      cout<<endl;
      cout<<out[j]<<","<<out[j+t]<<endl;
      cout<<outT[j]<<","<<outT[j+t]<<endl;
      cout<<+outTT[j]<<","<<+outTT[j+t]<<endl;
      cout<<+outTTT[j]<<","<<+outTTT[j+t]<<endl;
      cout<<(x[j]>=(mask + 1 - x1[j]))*(mask+1)<<","<<(y[j]>=(mask + 1 - y1[j]))*(mask+1)<<endl;
      uint64_t tx = 4*(mask+1)+x[j]-y[j]-(x[j]>=(mask + 1 - x1[j]))*(mask+1);
      uint64_t ty = 4*(mask+1)+y1[j]-x1[j]-(y[j]>=(mask + 1 - y1[j]))*(mask+1);
      cout<<"new:"<<tx<<":"<<ty<<"="<<(tx>=ty)<<endl;
      cout<<"old:"<<((x[j] + x1[j]) & mask)<<":"<<((y1[j] + y[j]) & mask)<<endl;
      cout<<"old:"<<((xx[j]) & mask)<<":"<<((yy[j]) & mask)<<endl;
      k2++;
    }
  }
  if (party == ALICE)
  {
    cout<<"Alice:"<<endl;
  }
  else
  {
    cout<<"Bob:"<<endl;
  }
  cout<<k1<<endl;
  cout<<k2<<endl;
  cout<<k3<<endl;
  uint8_t dd = 1;
  cout<<((0-dd)&1)<<endl;
}

void test_Copy(){
  uint32_t *tmp1 = new uint32_t[100]();
  prg.random_data(tmp1, 100 * sizeof(uint32_t));
  double time_0 = omp_get_wtime();
  for (int i = 0; i < 81000000; i++)
  {
    uint32_t *tmp2 = new uint32_t[2]();
    copy(tmp1,tmp1+100,tmp2);
    delete []tmp2;
  }
  double time_1 = omp_get_wtime();
  for (int i = 0; i < 81000000; i++)
  {
    uint32_t *tmp3 = new uint32_t[2]();
    memcpy(tmp1,tmp3,100 * sizeof(uint32_t));
    delete []tmp3;
  }
  double time_2 = omp_get_wtime();
  cout<<(time_1 - time_0)<<endl;
  cout<<(time_2 - time_1)<<endl;
}

// whole "do" dataset
int mainq1(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 1000;
  t[1] = 1000;
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    double Ptime = 0, PCom = 0, Pupload = 0, Pprocess = 0;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = p[rand()%n][j];
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[i]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[i]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    } 
    cout<<"q:"<<q[0]<<"\t"<<q[1]<<endl;
    // plain(p, q);
    uint32_t len = 0;
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    double time_end = omp_get_wtime();
    Ptime += time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    PCom += comm_end-comm_start;
    uint64_t *X = SkylineRes(Q,len);
    double time_end2 = omp_get_wtime();
    Qtime+= time_end2 - time_end;
    uint64_t comm_end2 = 0;
    for(int j = 0; j< THs; j++){
      comm_end2+=Iot[j]->counter;
    }
    QCom += comm_end2-comm_end;
    QCom1 += com1;
    QCom2 += com2;
    vector<uint64_t> XX(len);
    copy(X,X+len,XX.begin());
    Pupload += ss_upload;
    Pprocess += ss_process;
    Qdummy += ss_dummy;
    Qmask += ss_mask;
    Qselect += ss_select;
    Qlen += len;
    // StoreSKyline();
    // vector<uint64_t> XX(len);
    // copy(X, X + len, XX.begin());
    // Skyline_Print(XX);
    p.clear();
    q.clear();
    Q.clear();
  }
  delete[] SS_Con;
  delete[] SS_C;
  delete[] SS_L;
  for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  {
    delete[] SS_Skyline[j];
  }
  delete[] SS_Skyline;
  for(int j = 0; j< m; j++)
  {
    delete[] SS_G[j];
  }
  delete[] SS_G;
  Ptime = Ptime/ itrs;
  PCom = PCom/ itrs;
  Ptime = Ptime/ itrs;
  Pupload = Pupload/ itrs;
  Pprocess = Pprocess/ itrs;
  QCom = QCom/ itrs;
  QCom1 = QCom1/ itrs;
  QCom2 = QCom2/ itrs;
  Qdummy = Qdummy/ itrs;
  Qmask = Qmask/ itrs;
  Qselect = Qselect/ itrs;
  Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      ofstream outfile;
      outfile.open("../../tests/out_A.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Alice Total Time:" << Ptime << " s" << endl;
      outfile << " Alice Share Time:" << Pupload << " s" << endl;
      outfile << " Alice Process Time:" << Pprocess << " s" << endl;
      outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      ofstream outfile;
      outfile.open("../../tests/out_B.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Bob Total Time:" << Ptime << " s" << endl;
      outfile << " Bob Share Time:" << Pupload << " s" << endl;
      outfile << " Bob Process Time:" << Pprocess << " s" << endl;
      outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  } 
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    double Ptime = 0, PCom = 0, Pupload = 0, Pprocess = 0;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = p[rand()%n][j];
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[i]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[i]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    } 
    cout<<"q:"<<q[0]<<"\t"<<q[1]<<endl;
    // plain(p, q);
    uint32_t len = 0;
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    double time_end = omp_get_wtime();
    Ptime += time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    PCom += comm_end-comm_start;
    uint64_t *X = SkylineRes_T(Q,len);
    double time_end2 = omp_get_wtime();
    Qtime+= time_end2 - time_end;
    uint64_t comm_end2 = 0;
    for(int j = 0; j< THs; j++){
      comm_end2+=Iot[j]->counter;
    }
    QCom += comm_end2-comm_end;
    QCom1 += com1;
    QCom2 += com2;
    vector<uint64_t> XX(len);
    copy(X,X+len,XX.begin());
    Pupload += ss_upload;
    Pprocess += ss_process;
    Qdummy += ss_dummy;
    Qmask += ss_mask;
    Qselect += ss_select;
    Qlen += len;
    // StoreSKyline();
    // vector<uint64_t> XX(len);
    // copy(X, X + len, XX.begin());
    // Skyline_Print(XX);
    p.clear();
    q.clear();
    Q.clear();
  }
  delete[] SS_Con;
  delete[] SS_C;
  delete[] SS_L;
  for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  {
    delete[] SS_Skyline[j];
  }
  delete[] SS_Skyline;
  for(int j = 0; j< m; j++)
  {
    delete[] SS_G[j];
  }
  delete[] SS_G;
  Ptime = Ptime/ itrs;
  PCom = PCom/ itrs;
  Ptime = Ptime/ itrs;
  Pupload = Pupload/ itrs;
  Pprocess = Pprocess/ itrs;
  QCom = QCom/ itrs;
  QCom1 = QCom1/ itrs;
  QCom2 = QCom2/ itrs;
  Qdummy = Qdummy/ itrs;
  Qmask = Qmask/ itrs;
  Qselect = Qselect/ itrs;
  Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      ofstream outfile;
      outfile.open("../../tests/out_A.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Alice Total Time:" << Ptime << " s" << endl;
      outfile << " Alice Share Time:" << Pupload << " s" << endl;
      outfile << " Alice Process Time:" << Pprocess << " s" << endl;
      outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      ofstream outfile;
      outfile.open("../../tests/out_B.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread"<< endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread"<< endl;  
      outfile << " Bob Total Time:" << Ptime << " s" << endl;
      outfile << " Bob Share Time:" << Pupload << " s" << endl;
      outfile << " Bob Process Time:" << Pprocess << " s" << endl;
      outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  } 
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int mainwq(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 1000;
  t[1] = 1000;
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    double Ptime = 0, PCom = 0, Pupload = 0, Pprocess = 0;
    double Qtime = 0, QCom = 0, Qdummy = 0, Qselect = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = p[rand()%n][j];
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[i]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[i]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    } 
    cout<<"q:"<<q[0]<<"\t"<<q[1]<<endl;
    plain(p, q);
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    double time_end = omp_get_wtime();
    Ptime += time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    PCom += comm_end-comm_start;
    uint32_t len = 0;
    uint64_t *X = SkylineRes(Q,len);
    double time_end2 = omp_get_wtime();
    Qtime+= time_end2 - time_end;
    uint64_t comm_end2 = 0;
    for(int j = 0; j< THs; j++){
      comm_end2+=Iot[j]->counter;
    }
    QCom += comm_end2-comm_end;
    vector<uint64_t> XX(len);
    copy(X,X+len,XX.begin());
    Pupload += ss_upload;
    Pprocess += ss_process;
    Qdummy += ss_dummy;
    Qselect += ss_select;
    // StoreSKyline();
    // vector<uint64_t> XX(len);
    // copy(X, X + len, XX.begin());
    // Skyline_Print(XX);
    p.clear();
    q.clear();
    Q.clear();
  }
  delete[] SS_Con;
  delete[] SS_C;
  delete[] SS_L;
  for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  {
    delete[] SS_Skyline[j];
  }
  delete[] SS_Skyline;
  for(int j = 0; j< m; j++)
  {
    delete[] SS_G[j];
  }
  delete[] SS_G;
  Ptime = Ptime/ itrs;
  PCom = PCom/ itrs;
  Ptime = Ptime/ itrs;
  Pupload = Pupload/ itrs;
  Pprocess = Pprocess/ itrs;
  QCom = QCom/ itrs;
  Qdummy = Qdummy/ itrs;
  Qselect = Qselect/ itrs;
    if (party == ALICE)
    {
      ofstream outfile;
      outfile.open("../../tests/out_A.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Alice Total Time:" << Ptime << " s" << endl;
      outfile << " Alice Share Time:" << Pupload << " s" << endl;
      outfile << " Alice Process Time:" << Pprocess << " s" << endl;
      outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      ofstream outfile;
      outfile.open("../../tests/out_B.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Bob Total Time:" << Ptime << " s" << endl;
      outfile << " Bob Share Time:" << Pupload << " s" << endl;
      outfile << " Bob Process Time:" << Pprocess << " s" << endl;
      outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  } 
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    double Ptime = 0, PCom = 0, Pupload = 0, Pprocess = 0;
    double Qtime = 0, QCom = 0, Qdummy = 0, Qselect = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = p[rand()%n][j];
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[i]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[i]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    } 
    cout<<"q:"<<q[0]<<"\t"<<q[1]<<endl;
    plain(p, q);
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    double time_end = omp_get_wtime();
    Ptime += time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    PCom += comm_end-comm_start;
    uint32_t len = 0;
    uint64_t *X = SkylineRes_T(Q,len);
    double time_end2 = omp_get_wtime();
    Qtime+= time_end2 - time_end;
    uint64_t comm_end2 = 0;
    for(int j = 0; j< THs; j++){
      comm_end2+=Iot[j]->counter;
    }
    QCom += comm_end2-comm_end;
    vector<uint64_t> XX(len);
    copy(X,X+len,XX.begin());
    Pupload += ss_upload;
    Pprocess += ss_process;
    Qdummy += ss_dummy;
    Qselect += ss_select;
    // StoreSKyline();
    // vector<uint64_t> XX(len);
    // copy(X, X + len, XX.begin());
    // Skyline_Print(XX);
    p.clear();
    q.clear();
    Q.clear();
  }
  delete[] SS_Con;
  delete[] SS_C;
  delete[] SS_L;
  for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
  {
    delete[] SS_Skyline[j];
  }
  delete[] SS_Skyline;
  for(int j = 0; j< m; j++)
  {
    delete[] SS_G[j];
  }
  delete[] SS_G;
  Ptime = Ptime/ itrs;
  PCom = PCom/ itrs;
  Ptime = Ptime/ itrs;
  Pupload = Pupload/ itrs;
  Pprocess = Pprocess/ itrs;
  QCom = QCom/ itrs;
  Qdummy = Qdummy/ itrs;
  Qselect = Qselect/ itrs;
    if (party == ALICE)
    {
      ofstream outfile;
      outfile.open("../../tests/out_A.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Alice Total Time:" << Ptime << " s" << endl;
      outfile << " Alice Share Time:" << Pupload << " s" << endl;
      outfile << " Alice Process Time:" << Pprocess << " s" << endl;
      outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      ofstream outfile;
      outfile.open("../../tests/out_B.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << Pupload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << Pprocess << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread"  << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;  
      outfile << " Bob Total Time:" << Ptime << " s" << endl;
      outfile << " Bob Share Time:" << Pupload << " s" << endl;
      outfile << " Bob Process Time:" << Pprocess << " s" << endl;
      outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  } 
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

// select quadrant
int main(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;
  string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  string datan[3] ={"corr-","unif-","anti-"};
  //  n = 5000;
  //   data_path = "./data/input=10000/size=" + to_string(n) + filename;
  //   vector<vector<uint32_t>> p;
  //   loadP(p, data_path);
  //   loadG(p, G);
  //   m = p[0].size();
  //   SSG(G, SS_G);
  //   Store("1-");
  for (int kk = 1; kk < 2; kk++)
  {
    filename = filen[kk];
    dataname = datan[kk];
    for (int i = 2; i <= 6; i++)
    {
      n = t[i];
      data_path = "./data/input=10000/size=" + to_string(n) + filename;
      vector<vector<uint32_t>> p;
      loadP(p, data_path);
      loadG(p, G);
      m = p[0].size();
      SSG(G, SS_G);
      // plain(p, q);
      double sr = omp_get_wtime();
      ReadSKyline();
      double er = omp_get_wtime();
      double ss_read = er - sr;
      double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
      int itrs = 20;
      for (int itr = 0; itr < itrs; itr++)
      {
        vector<uint32_t> q(m);
        for (int j = 0; j < m; j++)
        {
          q[j] = 1000 + rand()%7000;
        }
        if (party == ALICE)
        {
          uint32_t *q0 = new uint32_t[m];
          for (int j = 0; j < m; j++)
          {
            q0[j] = q[j];
          }
          Iot[i]->send_data(q0, m * sizeof(uint32_t));
          delete[] q0;
        }
        else
        {
          uint32_t *q0 = new uint32_t[m];
          Iot[i]->recv_data(q0, m * sizeof(uint32_t));
          for (int j = 0; j < m; j++)
          {
            q[j] = q0[j];
          }
          delete[] q0;
        } 
        cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
        vector<uint32_t> Q(q.size());
        SSQ(q, Q);
        uint32_t len = 0;
        uint64_t comm_start = 0;
        for(int j = 0; j< THs; j++){
          comm_start+=Iot[j]->counter;
        }
        double start = omp_get_wtime();
        // uint64_t *X = SkylineRes_T(Q, len);
        uint64_t *X = SkylineRes(Q, len);
        double end = omp_get_wtime();
        Qtime += end - start;
        uint64_t comm_end = 0;
        for(int j = 0; j< THs; j++){
          comm_end+=Iot[j]->counter;
        }
        QCom += comm_end-comm_start;
        QCom1 += com1;
        QCom2 += com2;
        Qread += ss_read;
        Qdummy += ss_dummy;
        Qmask += ss_mask;
        Qselect += ss_select;
        Qlen += len;
        // StoreSKyline();
        // vector<uint64_t> XX(len);
        // copy(X, X + len, XX.begin());
        // Skyline_Print(XX);
        p.clear();
        q.clear();
        Q.clear();
      }
      delete[] SS_Con;
      delete[] SS_C;
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
        delete[] SS_Skyline;
      for(int j = 0; j< m; j++)
      {
        delete[] SS_G[j];
      }
      delete[] SS_G;
      Qtime = Qtime/ itrs;
      QCom = QCom/ itrs;
      QCom1 = QCom1/ itrs;
      QCom2 = QCom2/ itrs;
      Qread = Qread/ itrs;
      Qdummy = Qdummy/ itrs;
      Qmask = Qmask/ itrs;
      Qselect = Qselect/ itrs;
      Qlen = Qlen/ itrs;
      if (party == ALICE)
      {
        cout << "n = " + to_string(n) + " " + dataname << endl;
        cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
        cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
        cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
        cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
        cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
        cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
        cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
        ofstream outfile;
        outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
        outfile << " -------------------------------------" << endl;
        outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant" << endl;
        // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;
        outfile << " Alice Total Time:" << Qtime << " s" << endl;
        outfile << " Alice Read Time:" << Qread << " s" << endl;
        outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
        outfile << " Alice Mask Time:" << Qmask << " s" << endl;
        outfile << " Alice Select Time:" << Qselect << " s" << endl;
        outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
        outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
        outfile << " Alice Communication:" << QCom << " Bytes" << endl;
        // for (uint64_t b : XX)
        // {
        //   outfile << b << "\t";
        // }
        outfile << endl;
        outfile << " -------------------------------------" << endl;
        outfile.close();
      }
      else
      {
        cout << "n = " + to_string(n) + " " + dataname << endl;
        cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
        cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
        cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
        cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
        cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
        cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
        cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
        ofstream outfile;
        outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
        outfile << " -------------------------------------" << endl;
        outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant"<< endl;
        // outfile << "n = " + to_string(n) + " " + dataname + "Thread"<< endl;
        outfile << " Bob Total Time:" << Qtime << " s" << endl;
        outfile << " Bob Read Time:" << Qread << " s" << endl;
        outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
        outfile << " Bob Mask Time:" << Qmask << " s" << endl;
        outfile << " Bob Select Time:" << Qselect << " s" << endl;
        outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
        outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
        outfile << " Bob Communication:" << QCom << " Bytes" << endl;
        // for (uint64_t b : XX)
        // {
        //   outfile << b << "\t";
        // }
        outfile << endl;
        outfile << " -------------------------------------" << endl;
        outfile.close();
      }
    }
  }
  for (int kk = 1; kk < 2; kk++)
  {
    filename = filen[kk];
    dataname = datan[kk];
    for (int i = 2; i <= 6; i++)
    {
      n = t[i];
      data_path = "./data/input=10000/size=" + to_string(n) + filename;
      vector<vector<uint32_t>> p;
      loadP(p, data_path);
      loadG(p, G);
      m = p[0].size();
      SSG(G, SS_G);
      // plain(p, q);
      double sr = omp_get_wtime();
      ReadSKyline();
      double er = omp_get_wtime();
      double ss_read = er - sr;
      double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
      int itrs = 20;
      for (int itr = 0; itr < itrs; itr++)
      {
        vector<uint32_t> q(m);
        for (int j = 0; j < m; j++)
        {
          q[j] = 1000 + rand()%7000;
        }
        if (party == ALICE)
        {
          uint32_t *q0 = new uint32_t[m];
          for (int j = 0; j < m; j++)
          {
            q0[j] = q[j];
          }
          Iot[i]->send_data(q0, m * sizeof(uint32_t));
          delete[] q0;
        }
        else
        {
          uint32_t *q0 = new uint32_t[m];
          Iot[i]->recv_data(q0, m * sizeof(uint32_t));
          for (int j = 0; j < m; j++)
          {
            q[j] = q0[j];
          }
          delete[] q0;
        } 
        cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
        vector<uint32_t> Q(q.size());
        SSQ(q, Q);
        uint32_t len = 0;
        uint64_t comm_start = 0;
        for(int j = 0; j< THs; j++){
          comm_start+=Iot[j]->counter;
        }
        double start = omp_get_wtime();
        uint64_t *X = SkylineRes_T(Q, len);
        // uint64_t *X = SkylineRes(Q, len);
        double end = omp_get_wtime();
        Qtime += end - start;
        uint64_t comm_end = 0;
        for(int j = 0; j< THs; j++){
          comm_end+=Iot[j]->counter;
        }
        QCom += comm_end-comm_start;
        QCom1 += com1;
        QCom2 += com2;
        Qread += ss_read;
        Qdummy += ss_dummy;
        Qmask += ss_mask;
        Qselect += ss_select;
        Qlen += len;
        // StoreSKyline();
        // vector<uint64_t> XX(len);
        // copy(X, X + len, XX.begin());
        // Skyline_Print(XX);
        p.clear();
        q.clear();
        Q.clear();
      }
      delete[] SS_Con;
      delete[] SS_C;
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
        delete[] SS_Skyline;
      for(int j = 0; j< m; j++)
      {
        delete[] SS_G[j];
      }
      delete[] SS_G;
      Qtime = Qtime/ itrs;
      QCom = QCom/ itrs;
      QCom1 = QCom1/ itrs;
      QCom2 = QCom2/ itrs;
      Qread = Qread/ itrs;
      Qdummy = Qdummy/ itrs;
      Qmask = Qmask/ itrs;
      Qselect = Qselect/ itrs;
      Qlen = Qlen/ itrs;
      if (party == ALICE)
      {
        cout << "n = " + to_string(n) + " " + dataname << endl;
        cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
        cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
        cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
        cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
        cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
        cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
        cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
        ofstream outfile;
        outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
        outfile << " -------------------------------------" << endl;
        // outfile << "n = " + to_string(n) + " " + dataname + "NonThread" << endl;
        outfile << "n = " + to_string(n) + " " + dataname + "Thread:Quadrant" << endl;
        outfile << " Alice Total Time:" << Qtime << " s" << endl;
        outfile << " Alice Read Time:" << Qread << " s" << endl;
        outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
        outfile << " Alice Mask Time:" << Qmask << " s" << endl;
        outfile << " Alice Select Time:" << Qselect << " s" << endl;
        outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
        outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
        outfile << " Alice Communication:" << QCom << " Bytes" << endl;
        // for (uint64_t b : XX)
        // {
        //   outfile << b << "\t";
        // }
        outfile << endl;
        outfile << " -------------------------------------" << endl;
        outfile.close();
      }
      else
      {
        cout << "n = " + to_string(n) + " " + dataname << endl;
        cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
        cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
        cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
        cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
        cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
        cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
        cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
        ofstream outfile;
        outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
        outfile << " -------------------------------------" << endl;
        // outfile << "n = " + to_string(n) + " " + dataname + "NonThread"<< endl;
        outfile << "n = " + to_string(n) + " " + dataname + "Thread:Quadrant"<< endl;
        outfile << " Bob Total Time:" << Qtime << " s" << endl;
        outfile << " Bob Read Time:" << Qread << " s" << endl;
        outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
        outfile << " Bob Mask Time:" << Qmask << " s" << endl;
        outfile << " Bob Select Time:" << Qselect << " s" << endl;
        outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
        outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
        outfile << " Bob Communication:" << QCom << " Bytes" << endl;
        // for (uint64_t b : XX)
        // {
        //   outfile << b << "\t";
        // }
        outfile << endl;
        outfile << " -------------------------------------" << endl;
        outfile.close();
      }
    }
  }  
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

// select "do" quadrant
int maindd(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 1000;
  t[1] = 1000;
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    ReadSKyline();
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = p[rand()%n][j];
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      // uint64_t *X = SkylineRes_T(Q, len);
      uint64_t *X = SkylineRes(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      Qselect += ss_select;
      Qlen += len;
      // StoreSKyline();
      // vector<uint64_t> XX(len);
      // copy(X, X + len, XX.begin());
      // Skyline_Print(XX);
      p.clear();
      q.clear();
      Q.clear();
    }
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline[j];
    }
      delete[] SS_Skyline;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant"<< endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread"<< endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }

  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = t[i];
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    ReadSKyline();
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = p[rand()%n][j];
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      uint64_t *X = SkylineRes_T(Q, len);
      // uint64_t *X = SkylineRes(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      Qselect += ss_select;
      Qlen += len;
      // StoreSKyline();
      // vector<uint64_t> XX(len);
      // copy(X, X + len, XX.begin());
      // Skyline_Print(XX);
      p.clear();
      q.clear();
      Q.clear();
    }
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline[j];
    }
      delete[] SS_Skyline;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread:Quadrant" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread:Quadrant"<< endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread:Quadrant"<< endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  } 
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

//generate all quadrant skyline diagram
int main0(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  // t[0] = 10;
  // t[1] = 100;
  // t[2] = 1000;
  // t[3] = 3000;
  // t[4] = 5000;
  // t[5] = 7000;
  // t[6] = 9000;
  // string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  // string datan[3] ={"corr-","unif-","anti-"};
  // for (int kk = 2; kk < 3; kk++)
  // {
  //   filename = filen[kk];
  //   dataname = datan[kk];
  // for (int i = 5; i <= 5; i++)
  // {
  //   n = t[i];
  //   data_path = "./data/input=10000/size=" + to_string(n) + filename;
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = 1000;
    data_path = "./data/" + filename;  
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    vector<uint32_t> q{7000, 5000};
    // vector<uint32_t> q{20, 20};
    // plain(p, q);
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    dynamicSkyline(G,q);
    // vector<uint32_t> Q = SS_Two(q);
    // SS_Result(q, Q);
    double time_end = omp_get_wtime();
    double Ptime = time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    double PCom = comm_end-comm_start;
    q.clear();
    // cout << "n = " + to_string(n) + " " + dataname << endl;
    // cout << "Total Time\t" << RED << Ptime << " s" << RESET << endl;
    // cout << "Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << ss_upload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << ss_process << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      // ofstream outfile;
      // outfile.open("../../tests/out_A_C94.txt", ios::app | ios::in);
      // outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname << endl;
      // outfile << " Alice Total Time:" << Ptime << " s" << endl;
      // outfile << " Alice Share Time:" << ss_upload << " s" << endl;
      // outfile << " Alice Process Time:" << ss_process << " s" << endl;
      // outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      // outfile << " -------------------------------------" << endl;
      // outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << ss_upload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << ss_process << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      // ofstream outfile;
      // outfile.open("../../tests/out_B_C94.txt", ios::app | ios::in);
      // outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname << endl;
      // outfile << " Bob Total Time:" << Ptime << " s" << endl;
      // outfile << " Bob Share Time:" << ss_upload << " s" << endl;
      // outfile << " Bob Process Time:" << ss_process << " s" << endl;
      // outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      // outfile << " -------------------------------------" << endl;
      // outfile.close();
    }
  }
  // }
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

//whole time
int mainw0(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;
  string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  string datan[3] ={"corr-","unif-","anti-"};
  for (int kk = 1; kk < 3; kk++)
  {
    filename = filen[kk];
    dataname = datan[kk];
  for (int i = 3; i <= 3; i++)
  {
    n = t[i];
    data_path = "./data/input=10000/size=" + to_string(n) + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    vector<uint32_t> q(m);
    q[0] = 1000 + rand()%7000;
    q[1] = 1000 + rand()%7000;
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[i]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[i]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    } 
    cout<<"q:"<<q[0]<<"\t"<<q[1]<<endl;
    plain(p, q);
    uint64_t comm_start = 0;
    for(int j = 0; j< THs; j++){
      comm_start+=Iot[j]->counter;
    }
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    double time_end = omp_get_wtime();
    double Ptime = time_end - time_start;
    uint64_t comm_end = 0;
    for(int j = 0; j< THs; j++){
      comm_end+=Iot[j]->counter;
    }
    double PCom = comm_end-comm_start;
    uint32_t len = 0;
    uint64_t *X = SkylineRes(Q,len);
    double time_end2 = omp_get_wtime();
    double Qtime = time_end2 - time_end;
    uint64_t comm_end2 = 0;
    for(int j = 0; j< THs; j++){
      comm_end2+=Iot[j]->counter;
    }
    double QCom = comm_end2-comm_end;
    vector<uint64_t> XX(len);
    copy(X,X+len,XX.begin());
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline[j];
    }
    delete[] SS_Skyline;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    q.clear();
    Q.clear();
    Skyline_Print(XX);
    if (party == ALICE)
    {
      ofstream outfile;
      outfile.open("../../tests/out_A.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Total Time\t" << RED << Ptime << " s" << RESET << endl;    
      cout << "Alice Share Time\t" << RED << ss_upload << " s" << RESET << endl;
      cout << "Alice Process Time\t" << RED << ss_process << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << ss_dummy << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << ss_select << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname << endl;
      outfile << " Alice Total Time:" << Ptime << " s" << endl;
      outfile << " Alice Share Time:" << ss_upload << " s" << endl;
      outfile << " Alice Process Time:" << ss_process << " s" << endl;
      outfile << " Alice Communication:" << PCom << " Bytes" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Dummy Time:" << ss_dummy << " s" << endl;
      outfile << " Alice Select Time:" << ss_select << " s" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      for (uint64_t b : XX)
      {
        outfile << b << "\t";
      }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      ofstream outfile;
      outfile.open("../../tests/out_B.txt", ios::app | ios::in);
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Total Time\t" << RED << Ptime << " s" << RESET << endl;
      cout << "Bob Share Time\t" << RED << ss_upload << " s" << RESET << endl;
      cout << "Bob Process Time\t" << RED << ss_process << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << PCom << " Bytes" << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << ss_dummy << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << ss_select << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname << endl;
      outfile << " Bob Total Time:" << Ptime << " s" << endl;
      outfile << " Bob Share Time:" << ss_upload << " s" << endl;
      outfile << " Bob Process Time:" << ss_process << " s" << endl;
      outfile << " Bob Communication:" << PCom << " Bytes" << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Dummy Time:" << ss_dummy << " s" << endl;
      outfile << " Bob Select Time:" << ss_select << " s" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      for (uint64_t b : XX)
      {
        outfile << b << "\t";
      }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }
  }
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

//generate single quadrant skyline diagram
int main01(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;  
  for (int i = 1; i <= 1; i++)
  {
    // unordered_map<uint32_t, uint32_t *> S1;
    // S1[0] = new uint32_t[1];
    // uint32_t *x = new uint32_t[1];
    // x[0] = 1;
    // // S1[0] = x;
    // copy(x,x+1,S1[0]);
    // cout<<S1[0]<<" "<<S1[0][0]<<endl;
    // unordered_map<uint32_t, uint32_t *> S2;
    // S2 = S1;
    // cout<<S2[0]<<" "<<S2[0][0]<<endl;
    // x[0] = 2;
    // cout<<S1[0]<<" "<<S1[0][0]<<endl;
    // cout<<S2[0]<<" "<<S2[0][0]<<endl;
    // S1.clear();
    // cout<<S2[0]<<" "<<(S2[0][0]+1)<<endl;
    n = t[i];
    data_path = "./data/input=10000/size=" + to_string(n) + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    vector<uint32_t> q{7000, 5000};
    // vector<uint32_t> Q(q.size());
    // SSQ(q, Q);
    // SSG(G, SS_G);
    // ReadSKyline();
    // uint32_t len = 0;
       plain(p,q);
    uint64_t comm_start = Iot[0]->counter;
    double time_start = omp_get_wtime();
    vector<uint32_t> Q = SS_Two(q);
    // uint32_t * X = SkylineRes(Q,&len);
    // vector<uint32_t> XX(len);
    // copy(X,X+len,XX.begin());
    double time_end = omp_get_wtime();
    double time = time_end - time_start;
    uint64_t comm_end = Iot[0]->counter;
    vector<uint64_t> XX = SS_Result(q, Q);
    StoreSKyline();
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline[j];
    }
    delete[] SS_Skyline;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    q.clear();
    Q.clear();
    // SS_Print(XX);
  //   if (party == ALICE)
  //   {
  //     ofstream outfile;
  //     outfile.open("../../tests/out_A.txt", ios::app | ios::in);
  //     cout << "n = " + to_string(n) + " " + dataname << endl;
  //     cout << "Alice Total Time\t" << RED << time << " s" << RESET << endl;    
  //     cout << "Alice Share Time\t" << RED << ss_upload << " s" << RESET << endl;
  //     cout << "Alice Process Time\t" << RED << ss_process << " s" << RESET << endl;
  //     cout << "Alice Communication\t" << BLUE << ((double)(comm_end - comm_start) * 8) << " bits" << RESET << endl;
  //     outfile << " -------------------------------------" << endl;
  //     outfile << "n = " + to_string(n) + " " + dataname << endl;
  //     outfile << " Alice Total Time:" << time << " s" << endl;
  //     outfile << " Alice Share Time:" << ss_upload << " s" << endl;
  //     outfile << " Alice Process Time:" << ss_process << " s" << endl;
  //     outfile << " Alice Communication:" << ((double)(comm_end - comm_start) * 8) << " bits" << endl;
  //     for (uint64_t b : XX)
  //     {
  //       outfile << b << "\t";
  //     }
  //     outfile << endl;
  //     outfile << " -------------------------------------" << endl;
  //     outfile.close();
  //   }
  //   else
  //   {
  //     ofstream outfile;
  //     outfile.open("../../tests/out_B.txt", ios::app | ios::in);
  //     cout << "n = " + to_string(n) + " " + dataname << endl;
  //     cout << "Bob Total Time\t" << RED << time << " s" << RESET << endl;
  //     cout << "Bob Share Time\t" << RED << ss_upload << " s" << RESET << endl;
  //     cout << "Bob Process Time\t" << RED << ss_process << " s" << RESET << endl;
  //     cout << "Bob Communication\t" << BLUE << ((double)(comm_end - comm_start) * 8) << " bits" << RESET << endl;
  //     outfile << " -------------------------------------" << endl;
  //     outfile << "n = " + to_string(n) + " " + dataname << endl;
  //     outfile << " Bob Total Time:" << time << " s" << endl;
  //     outfile << " Bob Share Time:" << ss_upload << " s" << endl;
  //     outfile << " Bob Process Time:" << ss_process << " s" << endl;
  //     outfile << " Bob Communication:" << ((double)(comm_end - comm_start) * 8) << " bits" << endl;
  //     for (uint32_t b : XX)
  //     {
  //       outfile << b << "\t";
  //     }
  //     outfile << endl;
  //     outfile << " -------------------------------------" << endl;
  //     outfile.close();
  //   }
  }
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  // delete[] Auxt;
  // delete[] Prodt;
  // delete[] Otpackt;
  // delete[] Iot;
  return 0;
}

int main000(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);

  test_Cmp();

  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

// test file
int main0000(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;
  uint64_t b = 0;
  prg.random_data(&b, sizeof(uint64_t));
  cout<<"o:"<<b<<endl;
  b = HashP(b);
  cout<<"n:"<<b<<endl;
  // string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  // string datan[3] ={"corr-","unif-","anti-"};
  // for (int kk = 0; kk < 3; kk++)
  // {
  //   filename = filen[kk];
  //   dataname = datan[kk];
  //   for (int i = 2; i <= 6; i++)
  //   {
  //     n = t[i];
  //     data_path = "./data/input=10000/size=" + to_string(n) + filename;
  //     vector<vector<uint32_t>> p;
  //     loadP(p, data_path);
  //     loadG(p, G);
  //     m = p[0].size();
  //     SSG(G, SS_G);
  //     // plain(p, q);
  //     dynamicSkylineR(G); 
  //     Test_poly();
  //     p.clear();
  //     for(int j = 0; j< m; j++)
  //     {
  //       delete[] SS_G[j];
  //     }
  //     delete[] SS_G;
  //   }
  // }
  // delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

//D
int main1(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;
  string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  string datan[3] ={"corr-","unif-","anti-"};
  for (int kk = 0; kk < 1; kk++)
  {
    filename = filen[kk];
    dataname = datan[kk];
  for (int i = 2; i <= 6; i++)
  {
    n = t[i];
    data_path = "./data/input=10000/size=" + to_string(n) + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    // ReadSKyline();
    dynamicSkylineR(G); 
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, QselectSky = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = 1000 + rand()%7000;
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      // q[0] = 7000;
      // q[1] = 5000;
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      // uint64_t *X = SkylineRes_T(Q, len);
      // uint64_t *X = SkylineRes(Q, len);
      uint64_t *X = SkylineResbyQ(Q, len);
        // uint64_t *X = SkylineResbyQ_T(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      QselectSky += ss_selectSky;
      Qselect += ss_select;
      Qlen += len;
      // StoreSKyline();
      // vector<uint64_t> XX(len);
      // copy(X, X + len, XX.begin());
      // Skyline_Print(XX);
      q.clear();
      Q.clear();
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
      delete[] SS_Skyline;
    }
    p.clear();
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L_itr;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline_itr[j];
    }
    delete[] SS_Skyline_itr;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    QselectSky = QselectSky/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread-D" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D"<< endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread-D"<< endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }
  }
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

//D-T
int main2(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  int *t = new int[7];
  t[0] = 10;
  t[1] = 100;
  t[2] = 1000;
  t[3] = 3000;
  t[4] = 5000;
  t[5] = 7000;
  t[6] = 9000;
  
  string filen[3] ={"/small-correlated.txt","/small-uniformly-distributed.txt","/small-anti-correlated.txt"};
  string datan[3] ={"corr-","unif-","anti-"};
  for (int kk = 1; kk < 2; kk++)
  {
    filename = filen[kk];
    dataname = datan[kk];
  for (int i = 6; i <= 6; i++)
  {
    n = t[i];
    data_path = "./data/input=10000/size=" + to_string(n) + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    // ReadSKyline("-1");
    dynamicSkylineR(G); 
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, QselectSky = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = 1000 + rand()%7000;
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      // q[0] = 7000;
      // q[1] = 5000;
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      // uint64_t *X = SkylineRes_T(Q, len);
      // uint64_t *X = SkylineRes(Q, len);
      // uint64_t *X = SkylineResbyQ(Q, len);
      uint64_t *X = SkylineResbyQ_T(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      QselectSky += ss_selectSky;
      Qselect += ss_select;
      Qlen += len;
      // StoreSKyline();
      // vector<uint64_t> XX(len);
      // copy(X, X + len, XX.begin());
      // Skyline_Print(XX);
      q.clear();
      Q.clear();
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
      delete[] SS_Skyline;
    }
    p.clear();
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L_itr;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline_itr[j];
    }
    delete[] SS_Skyline_itr;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    QselectSky = QselectSky/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread-D" << endl;
      outfile << " Alice Skyline Number:" << Qlen << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D"<< endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread-D"<< endl;
      outfile << " Bob Skyline Number:" << Qlen << endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }
  }
  delete[] t;
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

// other dataset: D and D-T
int main3(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = 1000;
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    // ReadSKyline();
    dynamicSkylineR(G); 
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, QselectSky = 0, Qselect = 0, Qlen = 0;
    int itrs = 20;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = p[rand()%n][j];
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      // uint64_t *X = SkylineRes_T(Q, len);
      // uint64_t *X = SkylineRes(Q, len);
      // uint64_t *X = SkylineResbyQ(Q, len);
      uint64_t *X = SkylineResbyQ_T(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      QselectSky += ss_selectSky;
      Qselect += ss_select;
      Qlen += len;
      q.clear();
      Q.clear();
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
      delete[] SS_Skyline;
    }
    p.clear();
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L_itr;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline_itr[j];
    }
    delete[] SS_Skyline_itr;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    QselectSky = QselectSky/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread-D" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Bob Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D"<< endl;
      outfile << "n = " + to_string(n) + " " + dataname + "Thread-D"<< endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }

  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int main4(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0)+party);
  string filen[2] ={"diabets.txt", "obesity.txt"};
  string datan[2] ={"diabets-", "obesity-"};
  for (int i = 0; i < 2; i++)
  {
    filename = filen[i];
    dataname = datan[i];
    n = 1000;
    data_path = "./data/" + filename;
    vector<vector<uint32_t>> p;
    loadP(p, data_path);
    loadG(p, G);
    m = p[0].size();
    SSG(G, SS_G);
    // plain(p, q);
    double sr = omp_get_wtime();
    // ReadSKyline();
    dynamicSkylineR(G); 
    double er = omp_get_wtime();
    double ss_read = er - sr;
    double Qtime = 0, QCom = 0, QCom1 = 0, QCom2 = 0, Qread = 0, Qdummy = 0, Qmask = 0, QselectSky = 0, Qselect = 0, Qlen = 0;
    int itrs = 30;
    for (int itr = 0; itr < itrs; itr++)
    {
      vector<uint32_t> q(m);
      for (int j = 0; j < m; j++)
      {
        q[j] = p[rand()%n][j];
      }
      if (party == ALICE)
      {
        uint32_t *q0 = new uint32_t[m];
        for (int j = 0; j < m; j++)
        {
          q0[j] = q[j];
        }
        Iot[i]->send_data(q0, m * sizeof(uint32_t));
        delete[] q0;
      }
      else
      {
        uint32_t *q0 = new uint32_t[m];
        Iot[i]->recv_data(q0, m * sizeof(uint32_t));
        for (int j = 0; j < m; j++)
        {
          q[j] = q0[j];
        }
        delete[] q0;
      } 
      cout<<"q-"<<itr<<":"<<q[0]<<"\t"<<q[1]<<endl;
      vector<uint32_t> Q(q.size());
      SSQ(q, Q);
      uint32_t len = 0;
      uint64_t comm_start = 0;
      for(int j = 0; j< THs; j++){
        comm_start+=Iot[j]->counter;
      }
      double start = omp_get_wtime();
      // uint64_t *X = SkylineRes_T(Q, len);
      // uint64_t *X = SkylineRes(Q, len);
      uint64_t *X = SkylineResbyQ(Q, len);
      // uint64_t *X = SkylineResbyQ_T(Q, len);
      double end = omp_get_wtime();
      Qtime += end - start;
      uint64_t comm_end = 0;
      for(int j = 0; j< THs; j++){
        comm_end+=Iot[j]->counter;
      }
      QCom += comm_end-comm_start;
      QCom1 += com1;
      QCom2 += com2;
      Qread += ss_read;
      Qdummy += ss_dummy;
      Qmask += ss_mask;
      QselectSky += ss_selectSky;
      Qselect += ss_select;
      Qlen += len;
      q.clear();
      Q.clear();
      delete[] SS_L;
      for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
      {
        delete[] SS_Skyline[j];
      }
      delete[] SS_Skyline;
    }
    p.clear();
    delete[] SS_Con;
    delete[] SS_C;
    delete[] SS_L_itr;
    for(int j = 0; j< SS_G[0][0] * SS_G[0][1]; j++)
    {
      delete[] SS_Skyline_itr[j];
    }
    delete[] SS_Skyline_itr;
    for(int j = 0; j< m; j++)
    {
      delete[] SS_G[j];
    }
    delete[] SS_G;
    Qtime = Qtime/ itrs;
    QCom = QCom/ itrs;
    QCom1 = QCom1/ itrs;
    QCom2 = QCom2/ itrs;
    Qread = Qread/ itrs;
    Qdummy = Qdummy/ itrs;
    Qmask = Qmask/ itrs;
    QselectSky = QselectSky/ itrs;
    Qselect = Qselect/ itrs;
    Qlen = Qlen/ itrs;
    if (party == ALICE)
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Alice Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Alice Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Alice Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Alice Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Alice SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Alice Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Alice Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_A_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D" << endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread-D" << endl;
      outfile << " Alice Total Time:" << Qtime << " s" << endl;
      outfile << " Alice Read Time:" << Qread << " s" << endl;
      outfile << " Alice Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Alice Mask Time:" << Qmask << " s" << endl;
      outfile << " Alice SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Alice Select Time:" << Qselect << " s" << endl;
      outfile << " Alice Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Alice Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Alice Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
    else
    {
      cout << "n = " + to_string(n) + " " + dataname << endl;
      cout << "Bob Skyline Number\t" << RED << Qlen << RESET << endl;
      cout << "Bob Total Time\t" << RED << Qtime << " s" << RESET << endl;
      cout << "Bob Read Time\t" << RED << Qread << " s" << RESET << endl;
      cout << "Bob Dummy Time\t" << RED << Qdummy << " s" << RESET << endl;
      cout << "Alice Mask Time\t" << RED << Qmask << " s" << RESET << endl;
      cout << "Bob SelectQua Time\t" << RED << QselectSky << " s" << RESET << endl;
      cout << "Bob Select Time\t" << RED << Qselect << " s" << RESET << endl;
      cout << "Bob Communication\t" << BLUE << QCom << " Bytes" << RESET << endl;
      ofstream outfile;
      outfile.open("../../tests/out_B_qua.txt", ios::app | ios::in);
      outfile << " -------------------------------------" << endl;
      outfile << "n = " + to_string(n) + " " + dataname + "NonThread-D"<< endl;
      // outfile << "n = " + to_string(n) + " " + dataname + "Thread-D"<< endl;
      outfile << " Bob Total Time:" << Qtime << " s" << endl;
      outfile << " Bob Read Time:" << Qread << " s" << endl;
      outfile << " Bob Dummy Time:" << Qdummy << " s" << endl;
      outfile << " Bob Mask Time:" << Qmask << " s" << endl;
      outfile << " Bob SelectQua Time:" << QselectSky << " s" << endl;
      outfile << " Bob Select Time:" << Qselect << " s" << endl;
      outfile << " Bob Mask Communication:" << QCom1 << " Bytes" << endl;
      outfile << " Bob Select Communication:" << QCom2 << " Bytes" << endl;
      outfile << " Bob Communication:" << QCom << " Bytes" << endl;
      // for (uint64_t b : XX)
      // {
      //   outfile << b << "\t";
      // }
      outfile << endl;
      outfile << " -------------------------------------" << endl;
      outfile.close();
    }
  }
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}
