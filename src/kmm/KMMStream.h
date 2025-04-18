
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include <kmm/kmm.hpp>

#include "Stream.h"

#define IMPLEMENTATION_STRING "KMM"

#define TBSIZE 1024

template <class T>
class KMMStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;

    // Number of blocks for dot kernel
    int dot_num_blocks;

  public:

    KMMStream(const int, const int);
    ~KMMStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
