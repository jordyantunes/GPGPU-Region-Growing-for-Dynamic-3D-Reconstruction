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

#include "device.hpp"

using namespace pcl::device;

namespace pcl
{
  namespace device
  {
    template<typename T>
    __global__ void
    initializeVolume (PtrStep<T> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      
      
      if (x < VOLUME_X && y < VOLUME_Y)
      {
          T *pos = volume.ptr(y) + x;
          int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
          for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
             pack_tsdf (0.f, 0, *pos);
      }
    }   
  }
}

void
pcl::device::initVolume (PtrStep<short2> volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);      
  grid.y = divUp (VOLUME_Y, block.y);

  initializeVolume<<<grid, block>>>(volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    struct Tsdf
    {
      enum
      {
        CTA_SIZE_X = 32, CTA_SIZE_Y = 8,
        MAX_WEIGHT = 1 << 7
      };

      mutable PtrStep<short2> volume;
      float3 cell_size;

      Intr intr;

      Mat33 Rcurr_inv;
      float3 tcurr;

      PtrStepSz<ushort> depth_raw; //depth in mm

      float tranc_dist_mm;

      __device__ __forceinline__ float3
      getVoxelGCoo (int x, int y, int z) const
      {
        float3 coo = make_float3 (x, y, z);
        coo += 0.5f;         //shift to cell center;

        coo.x *= cell_size.x;
        coo.y *= cell_size.y;
        coo.z *= cell_size.z;

        return coo;
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

        short2 *pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(*pos);

        for (int z = 0; z < VOLUME_Z; ++z, pos += elem_step)
        {
          float3 v_g = getVoxelGCoo (x, y, z);            //3 // p

          //tranform to curr cam coo space
          float3 v = Rcurr_inv * (v_g - tcurr);           //4

          int2 coo;           //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)           //6
          {
            int Dp = depth_raw.ptr (coo.y)[coo.x];

            if (Dp != 0)
            {
              float xl = (coo.x - intr.cx) / intr.fx;
              float yl = (coo.y - intr.cy) / intr.fy;
              float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

              float sdf = 1000 * norm (tcurr - v_g) * lambda_inv - Dp; //mm

              sdf *= (-1);

              if (sdf >= -tranc_dist_mm)
              {
                float tsdf = fmin (1.f, sdf / tranc_dist_mm);

                int weight_prev;
                float tsdf_prev;

                //read and unpack
                unpack_tsdf (*pos, tsdf_prev, weight_prev);

                const int Wrk = 1;

                float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                int weight_new = min (weight_prev + Wrk, MAX_WEIGHT);

                pack_tsdf (tsdf_new, weight_new, *pos);
              }
            }
          }
        }
      }
    };

    __global__ void
    integrateTsdfKernel (const Tsdf tsdf) {
      tsdf ();
    }

    __global__ void
    tsdf2 (PtrStep<short2> volume, const float tranc_dist_mm, const Mat33 Rcurr_inv, float3 tcurr,
           const Intr intr, const PtrStepSz<ushort> depth_raw, const float3 cell_size)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      short2 *pos = volume.ptr (y) + x;
      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_x = Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z;
      float v_y = Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z;
      float v_z = Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z; ++z)
      {
        float3 vr;
        vr.x = v_g_x;
        vr.y = v_g_y;
        vr.z = (v_g_z + z * cell_size.z);

        float3 v;
        v.x = v_x + Rcurr_inv.data[0].z * z * cell_size.z;
        v.y = v_y + Rcurr_inv.data[1].z * z * cell_size.z;
        v.z = v_z + Rcurr_inv.data[2].z * z * cell_size.z;

        int2 coo;         //project to current cam
        coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
        coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);


        if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)         //6
        {
          int Dp = depth_raw.ptr (coo.y)[coo.x]; //mm

          if (Dp != 0)
          {
            float xl = (coo.x - intr.cx) / intr.fx;
            float yl = (coo.y - intr.cy) / intr.fy;
            float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

            float sdf = Dp - norm (vr) * lambda_inv * 1000; //mm


            if (sdf >= -tranc_dist_mm)
            {
              float tsdf = fmin (1.f, sdf / tranc_dist_mm);

              int weight_prev;
              float tsdf_prev;

              //read and unpack
              unpack_tsdf (*pos, tsdf_prev, weight_prev);

              const int Wrk = 1;

              float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
              int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

              pack_tsdf (tsdf_new, weight_new, *pos);
            }
          }
        }
        pos += elem_step;
      }       /* for(int z = 0; z < VOLUME_Z; ++z) */
    }      /* __global__ */
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth_raw, const Intr& intr, const float3& volume_size,
                                  const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, 
                                  PtrStep<short2> volume)
{
  Tsdf tsdf;

  tsdf.volume = volume;  
  tsdf.cell_size.x = volume_size.x / VOLUME_X;
  tsdf.cell_size.y = volume_size.y / VOLUME_Y;
  tsdf.cell_size.z = volume_size.z / VOLUME_Z;
  
  tsdf.intr = intr;

  tsdf.Rcurr_inv = Rcurr_inv;
  tsdf.tcurr = tcurr;
  tsdf.depth_raw = depth_raw;

  tsdf.tranc_dist_mm = tranc_dist*1000; //mm

  dim3 block (Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

#if 0
   //tsdf2<<<grid, block>>>(volume, tranc_dist, Rcurr_inv, tcurr, intr, depth_raw, tsdf.cell_size);
   integrateTsdfKernel<<<grid, block>>>(tsdf);
#endif
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}


namespace pcl
{
  namespace device
  {
    __global__ void
    scaleDepth (const PtrStepSz<ushort> depth, PtrStep<float> scaled, const Intr intr)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= depth.cols || y >= depth.rows)
        return;

      int Dp = depth.ptr (y)[x];

      float xl = (x - intr.cx) / intr.fx;
      float yl = (y - intr.cy) / intr.fy;
      float lambda = sqrtf (xl * xl + yl * yl + 1);

      scaled.ptr (y)[x] = Dp * lambda/1000.f; //meters
    }

    __global__ void
    tsdf23 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
            const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos = volume.ptr (y) + x;
      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos += elem_step)
      {
        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          {
            float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos, tsdf_prev, weight_prev);

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            pack_tsdf (tsdf_new, weight_new, *pos);
          }
        }
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__

    __global__ void
    tsdf23normal_hack (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
                  const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
            return;

        const float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
        const float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
        float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

        float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

        float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
        float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
        float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

        float z_scaled = 0;

        float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
        float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

        float tranc_dist_inv = 1.0f / tranc_dist;

        short2* pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(short2);

        //#pragma unroll
        for (int z = 0; z < VOLUME_Z;
            ++z,
            v_g_z += cell_size.z,
            z_scaled += cell_size.z,
            v_x += Rcurr_inv_0_z_scaled,
            v_y += Rcurr_inv_1_z_scaled,
            pos += elem_step)
        {
            float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
            if (inv_z < 0)
                continue;

            // project to current cam
            int2 coo =
            {
                __float2int_rn (v_x * inv_z + intr.cx),
                __float2int_rn (v_y * inv_z + intr.cy)
            };

            if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
            {
                float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

                float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

                if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
                {
                    float tsdf = fmin (1.0f, sdf * tranc_dist_inv);                                              

                    bool integrate = true;
                    if ((x > 0 &&  x < VOLUME_X-2) && (y > 0 && y < VOLUME_Y-2) && (z > 0 && z < VOLUME_Z-2))
                    {
                        const float qnan = numeric_limits<float>::quiet_NaN();
                        float3 normal = make_float3(qnan, qnan, qnan);

                        float Fn, Fp;
                        int Wn = 0, Wp = 0;
                        unpack_tsdf (*(pos + elem_step), Fn, Wn);
                        unpack_tsdf (*(pos - elem_step), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.z = (Fn - Fp)/cell_size.z;

                        unpack_tsdf (*(pos + volume.step/sizeof(short2) ), Fn, Wn);
                        unpack_tsdf (*(pos - volume.step/sizeof(short2) ), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.y = (Fn - Fp)/cell_size.y;

                        unpack_tsdf (*(pos + 1), Fn, Wn);
                        unpack_tsdf (*(pos - 1), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.x = (Fn - Fp)/cell_size.x;

                        if (normal.x != qnan && normal.y != qnan && normal.z != qnan)
                        {
                            float norm2 = dot(normal, normal);
                            if (norm2 >= 1e-10)
                            {
                                normal *= rsqrt(norm2);

                                float nt = v_g_x * normal.x + v_g_y * normal.y + v_g_z * normal.z;
                                float cosine = nt * rsqrt(v_g_x * v_g_x + v_g_y * v_g_y + v_g_z * v_g_z);

                                if (cosine < 0.5)
                                    integrate = false;
                            }
                        }
                    }

                    if (integrate)
                    {
                        //read and unpack
                        float tsdf_prev;
                        int weight_prev;
                        unpack_tsdf (*pos, tsdf_prev, weight_prev);

                        const int Wrk = 1;

                        float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                        int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

                        pack_tsdf (tsdf_new, weight_new, *pos);
                    }
                }
            }
        }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__

	//__global__ void
	//	tsdf23_weight_aware (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
	//	const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
	//	, const ConfigDASS cfg, const PtrStepSz<unsigned char> indicator
	//	)

    __global__ void
		tsdf23_weight_aware (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
		const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
		, const ConfigDASS cfg)
	{

		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= VOLUME_X || y >= VOLUME_Y)
			return;

		float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
		float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
		float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

		float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

		float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
		float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
		float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

		float z_scaled = 0;

		float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
		float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

		float tranc_dist_inv = 1.0f / tranc_dist;

		short2* pos = volume.ptr (y) + x;
		int elem_step = volume.step * VOLUME_Y / sizeof(short2);

		//#pragma unroll
		for (int z = 0; z < VOLUME_Z;
			++z,
			v_g_z += cell_size.z,
			z_scaled += cell_size.z,
			v_x += Rcurr_inv_0_z_scaled,
			v_y += Rcurr_inv_1_z_scaled,
			pos += elem_step)
		{
			float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
			if (inv_z < 0)
				continue;

			// project to current cam
			int2 coo =
			{
				__float2int_rn (v_x * inv_z + intr.cx),
				__float2int_rn (v_y * inv_z + intr.cy)
			};

			if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)
			{
				float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

				float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

				if (Dp_scaled != 0 && sdf >= -tranc_dist ) //meters
				{
					float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

					//read and unpack
					float tsdf_prev;
					int weight_prev;
					unpack_tsdf (*pos, tsdf_prev, weight_prev);
					// ********************************************************************
					// Various weight definitions and updating strategies:
					//    DA:   weight_mode = 5,   update_mode = 7
					//    DASS: weight_mode = 5,   update_mode = 8
					// ********************************************************************
					int Wrk = 1;
					float step = 1.0f;

					// to choose weight definition based on 'weight_mode'
					switch ( cfg.weight_mode )
					{
					default: // UNKNOWN mode !!!
					case 1: // constant weight = 1 (KinFu default) multiplied by scale
						Wrk = (int)(1 * cfg.weight_scale);
						break;

					case 2: // weight ~ distance
					case 3:
					case 4:					
					case 5: // weight ~ 1/distance^2
						// checks the limits
						if ( Dp_scaled >= cfg.minimum_distance && Dp_scaled <= cfg.maximum_distance )
						{
							step = ((1/(cfg.minimum_distance*cfg.minimum_distance)) - (1/(cfg.maximum_distance*cfg.maximum_distance))) / cfg.weight_bins;
							Wrk = (int)( (((1/(Dp_scaled*Dp_scaled)) - (1/(cfg.maximum_distance*cfg.maximum_distance))) / step) * cfg.weight_scale);
						}
						else
						{
							Wrk = 0;
						}
						break;

					

					//  case 6: // weight ~ Indicator 
					//	Wrk = indicator.ptr (coo.y)[coo.x] ;
					//	// Wrk = 0; 
					//	// if (Wrk == 51)
					//	// printf("%d %d %d\n", Wrk, coo.x, coo.y);
					//	break;
					}

					float tsdf_new = 0;
					float difference_by_stable_value(0);
					int new_weight_temporary = 0;

					int tmp = 0;
					unsigned char weight_8bit = 0;
					unsigned char last_weight_8bit = 0;

					// to choose weight-update strategy based on 'weight_update'
					switch ( cfg.weight_update )
					{
						//case 0: // NEVER UPDATE
						//    be_update = false;
						//    break;

					default: // UNKNOWN update strategy !!!
					case 1: // accumulating (tsdf default)
						tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
						new_weight_temporary = weight_prev + Wrk;
						break;

					case 2: 
					case 3:
					case 4:
					case 5:
					case 6:
						break;

					case 7: // maximizing: updates UNTIL graeter values are available (in a range)

						// weight_prev (HL 16-bit)   =   last_weight_8bit (H 8-bit)   +   weight_8bit (L 8-bit)
						weight_8bit = weight_prev;
						last_weight_8bit = weight_prev >> 8;

						if ( Wrk >= last_weight_8bit*cfg.weight_range )
						{
							// same as original
							tsdf_new = (tsdf_prev * weight_8bit + Wrk * tsdf) / (weight_8bit + Wrk);

							// updates the last_weight_8bit
							new_weight_temporary = min(255, weight_8bit + Wrk); // 8-bit so maximum =255

							if ( Wrk > last_weight_8bit ) last_weight_8bit = Wrk;

							new_weight_temporary += last_weight_8bit << 8;
						}
						else
						{
							tsdf_new = tsdf_prev;
							new_weight_temporary = weight_prev;
						}
						break;

					case 8: // maximizing: updates UNTIL greater values are available (in a range) Wrk=1
						// try to be more similar to the original tsdf model

						// weight_prev (HL 16-bit)   =   last_weight_8bit (H 8-bit)   +   weight_8bit (L 8-bit)
						weight_8bit = weight_prev;
						last_weight_8bit = weight_prev >> 8;

						if ( Wrk >= last_weight_8bit*cfg.weight_range )
						{
							// same as original
							tsdf_new = (tsdf_prev * weight_8bit + 1 * tsdf) / (weight_8bit + 1);

							// updates the last_weight_8bit
							new_weight_temporary = min(cfg.maximum_weight, weight_8bit + 1);

							if ( Wrk > last_weight_8bit ) last_weight_8bit = Wrk;

							new_weight_temporary += last_weight_8bit << 8;
						}
						else
						{
							tsdf_new = tsdf_prev;
							new_weight_temporary = weight_prev;
						}
						break;
					}

					// checks the maximum weight limit
					int weight_new = 0;
					if ( cfg.weight_update == 7 || cfg.weight_update == 8 ) weight_new = new_weight_temporary;
					else weight_new = min (new_weight_temporary, cfg.maximum_weight );

					// ********************************************************************
					// ********************************************************************
					if ( cfg.weight_update > 0 ) pack_tsdf (tsdf_new, weight_new, *pos);
				}
			}
		}       // for(int z = 0; z < VOLUME_Z; ++z)
	}      // __global__

  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth, const Intr& intr,
                                  const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr, 
                                  float tranc_dist,
                                  PtrStep<short2> volume, DeviceArray2D<float>& depthScaled)
{
  // Maximum value the weight can take = 8
  // Maximum value total weight can take = 255
  // Mode = 5 and Update = 8
  // Bins = 1 ??
  // Minimum Distance = 0.8 m
  // Maximum Distance = 10 m
  // Ratio of maximum weight for updation = 80%
  ConfigDASS cfg(5, 8, 1, 255, 0.8, 4.0, 0.80, 255);
  // ConfigDASS cfg(5, 8, 1, 255, 0.4, 6.0, 0.80, 255);
  depthScaled.create (depth.rows, depth.cols);

  dim3 block_scale (32, 8);
  dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

  //scales depth along ray and converts mm -> meters. 
  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
  cudaSafeCall ( cudaGetLastError () );

  float3 cell_size;
  cell_size.x = volume_size.x / VOLUME_X;
  cell_size.y = volume_size.y / VOLUME_Y;
  cell_size.z = volume_size.z / VOLUME_Z;

  //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 block (16, 16);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

  PtrStepSz<unsigned char> indicator;
  //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);
  //tsdf23_weight_aware<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, cfg, indicator);
  tsdf23_weight_aware<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, cfg);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

//// modified version for indicator map
//void
//pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth, const Intr& intr,
//                                  const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr, 
//                                  float tranc_dist,
//                                  PtrStep<short2> volume, DeviceArray2D<float>& depthScaled, const PtrStepSz<unsigned char>& indicator)
//{
//  // Maximum value the weight can take = 8
//  // Maximum value total weight can take = 255
//  // Mode = 5 and Update = 8
//  // Bins = 1 ??
//  // Minimum Distance = 0.8 m
//  // Maximum Distance = 10 m
//  // Ratio of maximum weight for updation = 80%
//
// 
//
//  ConfigDASS cfg(6, 8, 1, 255, 0.8, 4.0, 0.80, 255);
//  // ConfigDASS cfg(5, 8, 1, 255, 0.4, 6.0, 0.80, 255);
//  depthScaled.create (depth.rows, depth.cols);
//
//  dim3 block_scale (32, 8);
//  dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));
//
//  //scales depth along ray and converts mm -> meters. 
//  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
//  cudaSafeCall ( cudaGetLastError () );
//
//  /*
//  int a = 200;
//  int b = 200;
//
//  int test = depthScaled.ptr(a)[b];
//  printf("%i\n",test);
//  */
//
//  float3 cell_size;
//  cell_size.x = volume_size.x / VOLUME_X;
//  cell_size.y = volume_size.y / VOLUME_Y;
//  cell_size.z = volume_size.z / VOLUME_Z;
//
//  //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
//  dim3 block (16, 16);
//  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));
//
//  //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);
//  tsdf23_weight_aware<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, cfg, indicator);
//
//  cudaSafeCall ( cudaGetLastError () );
//  cudaSafeCall (cudaDeviceSynchronize ());
//}