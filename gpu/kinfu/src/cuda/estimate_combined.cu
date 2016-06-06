/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

//#include <pcl/gpu/utils/device/block.hpp>
//#include <pcl/gpu/utils/device/funcattrib.hpp>
#include "device.hpp"

#define RANGA_MODIFICATION_DEPTHWEIGHT_CUTOFF 1
//#define RANGA_MODIFICATION_DEPTHWEIGHT_CURRENT_FRAME 1

namespace pcl
{
  namespace device
  {
    typedef double float_type;

    template<int CTA_SIZE_, typename T>
    static __device__ __forceinline__ void reduce(volatile T* buffer)
    {
      int tid = Block::flattenedThreadId();
      T val =  buffer[tid];

      if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
      if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
      if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
      if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = val + buffer[tid +  64]; __syncthreads(); }

      if (tid < 32)
      {
        if (CTA_SIZE_ >=   64) { buffer[tid] = val = val + buffer[tid +  32]; }
        if (CTA_SIZE_ >=   32) { buffer[tid] = val = val + buffer[tid +  16]; }
        if (CTA_SIZE_ >=   16) { buffer[tid] = val = val + buffer[tid +   8]; }
        if (CTA_SIZE_ >=    8) { buffer[tid] = val = val + buffer[tid +   4]; }
        if (CTA_SIZE_ >=    4) { buffer[tid] = val = val + buffer[tid +   2]; }
        if (CTA_SIZE_ >=    2) { buffer[tid] = val = val + buffer[tid +   1]; }
      }
    }

    struct Combined
    {
      enum
      {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,
        CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
      };


      Mat33 Rcurr;
      float3 tcurr;

      PtrStep<float> vmap_curr;
      PtrStep<float> nmap_curr;

      Mat33 Rprev_inv;
      float3 tprev;

      Intr intr;

      PtrStep<float> vmap_g_prev;
      PtrStep<float> nmap_g_prev;

      float distThres;
      float angleThres;

      int cols;
      int rows;

      mutable PtrStep<float_type> gbuf;

      __device__ __forceinline__ bool
      search (int x, int y, float3& n, float3& d, float3& s) const
      {
        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];

        if (isnan (ncurr.x))
          return (false);

        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;

        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);         // prev camera coo space

        int2 ukr;         //projection
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);      //4
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);                      //4

        if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
          return (false);

        float3 nprev_g;
        nprev_g.x = nmap_g_prev.ptr (ukr.y)[ukr.x];

        if (isnan (nprev_g.x))
          return (false);

        float3 vprev_g;
        vprev_g.x = vmap_g_prev.ptr (ukr.y       )[ukr.x];
        vprev_g.y = vmap_g_prev.ptr (ukr.y + rows)[ukr.x];
        vprev_g.z = vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

        float dist = norm (vprev_g - vcurr_g);
        if (dist > distThres)
          return (false);

        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;

        nprev_g.y = nmap_g_prev.ptr (ukr.y + rows)[ukr.x];
        nprev_g.z = nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

        float sine = norm (cross (ncurr_g, nprev_g));

        if (sine >= angleThres)
          return (false);
        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;
        return (true);
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        float3 n, d, s, temp_1, temp_2;
        bool found_coresp = false;
		float depth = 0.f, min_depth = 0.8, max_depth = 4, weight = 1, step = 1;
		 float new_min_depth = 0;
		 float mind = 0.8, maxd = 4;

        if (x < cols && y < rows)
          found_coresp = search (x, y, n, d, s);

        float row[7];

        if (found_coresp)
        {
			depth = vmap_curr.ptr (y + 2 * rows)[x];

			if ( depth >= 0.8 && depth <= max_depth )
			{
#if RANGA_MODIFICATION_DEPTHWEIGHT_CUTOFF
				step = ((1/(min_depth*min_depth)) - (1/(max_depth*max_depth)));
				weight =  (((1/(depth*depth)) - (1/(max_depth*max_depth))) / step);
				weight = fabs(sqrt(weight));

				if(weight < 0.25)
					weight = 0.25;
#elif RANGA_MODIFICATION_DEPTHWEIGHT_CURRENT_FRAME
				int less_then_1500 = intr.number_less_than_1000 + intr.number_less_than_1500;
				int less_then_2000 = less_then_1500 + intr.number_less_than_2000;
				int less_then_2500 = less_then_2000 + intr.number_less_than_2500;
				int less_then_3000 = less_then_2500 + intr.number_less_than_3000;
				int less_then_3500 = less_then_3000 + intr.number_less_than_3500;
				int less_then_4000 = less_then_3500 + intr.number_less_than_4000;
				int disable_weights = 0;

				if(intr.number_less_than_1000 > (640*480/5)) // &&  ((intr.depth_max - intr.depth_min) > 1000))

				{
					new_min_depth = 0.8; //0.5;
				}
				else if( less_then_1500 > (640*480/5))
				{
					new_min_depth = 1.25;
				}
				else if( less_then_2000 > (640*480/5))
				{
					new_min_depth = 1.75;
				}
				else if( less_then_2500 > (640*480/5))
				{
					new_min_depth = 2.25;
				}
				else if( less_then_3000 > (640*480/5))
				{
					new_min_depth = 2.75;
				}
				else if( less_then_3500 > (640*480/5))
				{
					new_min_depth = 3.25;
					disable_weights = 1;
				}
				else
				{
					new_min_depth = 3.25;
					disable_weights = 1;
				}

				//if(depth < 0.8)
					//depth = 0.8;
				if(!disable_weights)
				{
					//if(intr.depth_min != 0)
						//mind = ((float)intr.depth_min)/1000;
					mind = new_min_depth;

					//if(intr.depth_max != 0)
						maxd = ((float)max_depth);

					float temp_max_sqr = ((mind *  mind * maxd * maxd * 15/16)/ (mind*mind - maxd*maxd/16));

					step = ((1/(mind*mind)) - (1/(temp_max_sqr)));

					weight =  (((1/(depth*depth)) - (1/(temp_max_sqr))) / step);
					//weight = weight * 64;
					//weight = fabs(sqrt(weight));

				}
				else
				// Not enough point near the camera to apply weighted ICP (i.e., without big error in measurements)
				// Switch to un-weighted ICP
				{
					weight = 1;
				}

#if RANGA_MODIFICATION_ORIENTATION
				//if(intr.number_less_than > (640*480/5))//((intr.depth_max - intr.depth_min) > 500))
				{
					float3 rayvector;
					rayvecto .x = x - intr.cx;
					rayvector.y = y - intr.cy;
					rayvector.z = (intr.fx + intr.fy)/2;

					float norm_value = norm(rayvector);

					float3 normalvector;
					float weight1 = 0.0f;
					normalvector.x = nmap_curr.ptr(y ) [x];
					normalvector.y = nmap_curr.ptr(y + rows) [x];
					normalvector.z = nmap_curr.ptr(y + 2 * rows) [x];

					float norm_value1 = norm(normalvector);

					weight1 = abs(dot(rayvector, normalvector))/(norm_value * norm_value1);

					if(weight1 > 0.6 && weight1 <= 1.0)
					{
						weight1 = (weight1 - 0.5)/ 0.5;
					}
					else if(weight1 > 1)
					{
						// This should not be reached
						weight1 = 0;
					}
					else
						weight1 = 1;

					weight = weight * weight1;

					//weight = fabs(sqrt(weight));
				}

				
#endif
				//weight = weight * 4;
				weight = fabs(sqrt(weight));
				//if(weight < 0.25)
					//weight = 0.25;
#else
				step = ((1/(min_depth)) - (1/(max_depth)));
				weight =  (((1/(depth)) - (1/(max_depth))) / step);
				weight = fabs(sqrt(weight));
#endif
			}
			else if(depth > max_depth) // || depth < min_depth)  // Minimum depth is removed as I found a case where in minimum depth is less than 0.4 m 
			// 0.8 is the minimum valid value for the kinect V1 sensor in default mode
			// 4 is the maximum valid value for kinect V1 sensor
			// http://msdn.microsoft.com/en-us/library/hh973078.aspx
			{
				weight = 0.25;
			}
			else
			{
				// As it should be square root of the actual weight
				weight = 1; //8;
			}

		  temp_1 = cross (s, n);
		  temp_2 = n;

		  temp_1.x = temp_1.x * weight ;
		  temp_1.y = temp_1.y * weight ;
		  temp_1.z = temp_1.z * weight ;

		  temp_2.x = n.x * weight;
		  temp_2.y = n.y * weight;
		  temp_2.z = n.z * weight;

#if 1
          *(float3*)&row[0] = temp_1;
          *(float3*)&row[3] = temp_2;

          row[6] = weight * dot (n, d - s);
#else
         *(float3*)&row[0] = cross (s, n);
          *(float3*)&row[3] = n;

          row[6] = dot (n, d - s);
#endif
        }
        else
          row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

        __shared__ float_type smem[CTA_SIZE];
        int tid = Block::flattenedThreadId ();

        int shift = 0;
        for (int i = 0; i < 6; ++i)        //rows
        {
          #pragma unroll
          for (int j = i; j < 7; ++j)          // cols + b
          {
            __syncthreads ();
            smem[tid] = row[i] * row[j];
            __syncthreads ();

            reduce<CTA_SIZE>(smem);

            if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
          }
        }
      }
    };

    __global__ void
    combinedKernel (const Combined cs) 
    {
      cs ();
    }

    struct TranformReduction
    {
      enum
      {
        CTA_SIZE = 512,
        STRIDE = CTA_SIZE,

        B = 6, COLS = 6, ROWS = 6, DIAG = 6,
        UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
        TOTAL = UPPER_DIAG_MAT + B,

        GRID_X = TOTAL
      };

      PtrStep<float_type> gbuf;
      int length;
      mutable float_type* output;

      __device__ __forceinline__ void
      operator () () const
      {
        const float_type *beg = gbuf.ptr (blockIdx.x);
        const float_type *end = beg + length;

        int tid = threadIdx.x;

        float_type sum = 0.f;
        for (const float_type *t = beg + tid; t < end; t += STRIDE)
          sum += *t;

        __shared__ float_type smem[CTA_SIZE];

        smem[tid] = sum;
        __syncthreads ();

		reduce<CTA_SIZE>(smem);

        if (tid == 0)
          output[blockIdx.x] = smem[0];
      }
    };

    __global__ void
    TransformEstimatorKernel2 (const TranformReduction tr) 
    {
      tr ();
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::estimateCombined (const Mat33& Rcurr, const float3& tcurr, 
                               const MapArr& vmap_curr, const MapArr& nmap_curr, 
                               const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
                               const MapArr& vmap_g_prev, const MapArr& nmap_g_prev, 
                               float distThres, float angleThres,
                               DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, 
                               float_type* matrixA_host, float_type* vectorB_host)
{
  int cols = vmap_curr.cols ();
  int rows = vmap_curr.rows () / 3;

  Combined cs;

  cs.Rcurr = Rcurr;
  cs.tcurr = tcurr;

  cs.vmap_curr = vmap_curr;
  cs.nmap_curr = nmap_curr;

  cs.Rprev_inv = Rprev_inv;
  cs.tprev = tprev;

  cs.intr = intr;

  cs.vmap_g_prev = vmap_g_prev;
  cs.nmap_g_prev = nmap_g_prev;

  cs.distThres = distThres;
  cs.angleThres = angleThres;

  cs.cols = cols;
  cs.rows = rows;

//////////////////////////////

  dim3 block (Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  mbuf.create (TranformReduction::TOTAL);
  if (gbuf.rows () != TranformReduction::TOTAL || gbuf.cols () < (int)(grid.x * grid.y))
    gbuf.create (TranformReduction::TOTAL, grid.x * grid.y);

  cs.gbuf = gbuf;

  combinedKernel<<<grid, block>>>(cs);
  cudaSafeCall ( cudaGetLastError () );
  //cudaSafeCall(cudaDeviceSynchronize());

  //printFuncAttrib(combinedKernel);

  TranformReduction tr;
  tr.gbuf = gbuf;
  tr.length = grid.x * grid.y;
  tr.output = mbuf;

  TransformEstimatorKernel2<<<TranformReduction::TOTAL, TranformReduction::CTA_SIZE>>>(tr);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());

  float_type host_data[TranformReduction::TOTAL];
  mbuf.download (host_data);

  int shift = 0;
  for (int i = 0; i < 6; ++i)  //rows
    for (int j = i; j < 7; ++j)    // cols + b
    {
      float_type value = host_data[shift++];
      if (j == 6)       // vector b
        vectorB_host[i] = value;
      else
        matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
    }
}
