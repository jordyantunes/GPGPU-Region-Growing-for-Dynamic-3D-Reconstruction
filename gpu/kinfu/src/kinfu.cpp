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

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  #include <opencv2/gpu/gpu.hpp>
#endif

#define RANGA_MODIFICATION 1

#ifdef RANGA_MODIFICATION
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#endif

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
  namespace gpu
  {
    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
  }
}

#ifdef RANGA_MODIFICATION
std::vector<unsigned short> source_depth_data_temp_; // ready for gpu memory , depth map for TSDF (surface reconstruction)
std::vector<unsigned short> source_depth_data_temp1_; // ready for gpu memory , depth map for ICP (point to planar alignment algorithm)
std::vector<unsigned char> source_depth_data_temp2_; // ready for gpu memory, indicator map for TSDF (weight) 
PtrStepSz<const unsigned short> depth_temp_;
char basepath_kinfu[500];
int bilateral_filter_kinfu = 1;
int indicator = 0;

FILE * fp = NULL; // TSDF -> Refined Depth Map
FILE * fp_1 = NULL; // ICP -> Raw Depth Map
FILE * fp_2 = NULL; // TSDF -> Indicator Map for Weight
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker (int rows, int cols) : rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), disable_icp_(false)
{
  const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);
   
  tsdf_volume_ = TsdfVolume::Ptr( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize(volume_size);
  
  setDepthIntrinsics (KINFU_DEFAULT_DEPTH_FOCAL_X, KINFU_DEFAULT_DEPTH_FOCAL_Y); // default values, can be overwritten
  
  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  setIcpCorespFilteringParams (default_distThres, default_angleThres);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  allocateBufffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getDepthIntrinsics (float& fx, float& fy, float& cx, float& cy)
{
  fx = fx_;
  fy = fy_;
  cx = cx_;
  cy = cy_;
}

void
pcl::gpu::KinfuTracker::initPath(char * base_path, int indicator_, int bilateral_filter1, int mode_)
{
	strcpy(basepath_kinfu, base_path);

#ifdef RANGA_MODIFICATION
	// Open the Associate file
	char icp_str[500],sr_str[500], tsdf_str[500];
	strcpy(icp_str, basepath_kinfu);
	strcpy(sr_str, basepath_kinfu);

	int mode = mode_ ;

	if (mode == 0) // Bilateral
	{
		strcat(sr_str, "depth_assort.txt");
		fp = fopen(sr_str, "r");
	}
	
	if (mode == 1) // AD
	{
		strcat(sr_str, "depth_assort_ad.txt");
		fp = fopen(sr_str, "r");
	}

	if (mode == 2) // Modified AD
	{
		strcat(sr_str, "depth_assort_adg.txt");
		fp = fopen(sr_str, "r");
	}

	strcat(icp_str, "depth_assort.txt");
	fp_1 = fopen(icp_str, "r"); // ICP

	//int indicator = indicator_;
	//if (indicator == 1) // TSDF->Indicator
	//{
	//strcpy(tsdf_str, basepath_kinfu);
	//strcat(tsdf_str, "indicator.txt");
	//fp_2 = fopen(tsdf_str,"r"); 
	//}

	bilateral_filter_kinfu = bilateral_filter1;
#endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setInitalCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::rows ()
{
  return (rows_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::reset()
{
  if (global_time_)
    cout << "Reset" << endl;

  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  tsdf_volume_->reset();
    
  if (color_volume_) // color integration mode is enabled
    color_volume_->reset();    
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::allocateBufffers (int rows, int cols)
{    
  depths_curr_.resize (LEVELS);
  vmaps_g_curr_.resize (LEVELS);
  nmaps_g_curr_.resize (LEVELS);

  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

    coresps_[i].create (pyr_rows, pyr_cols);
  }  
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);
}

// Load File for ICP from Raw Depth Map Dataset
void pcl::gpu::KinfuTracker::loadFile(PtrStepSz<const unsigned short>& depth_temp)
{
#ifdef RANGA_MODIFICATION
		// Open the PNG files one by one 

		char temp_str[200], remove_str[200];
		char string_filename[500]; // = "C:\\Zhengyu\\3D Dataset\\rgbd_dataset_freiburg1_xyz\\"; // ICP
		unsigned short min = 0xffff, max = 1, temp;
		unsigned int count_less_than_1500 = 0, count_less_than_1000 = 0, count_less_than_2000 = 0, count_less_than_2500 = 0, count_less_than_3000 = 0;
		unsigned int count_less_than_3500 = 0, count_less_than_4000 = 0;

		strcpy(string_filename, basepath_kinfu) ; // basepath_kinfu from base
        strcat(string_filename, "output\\");
			
		// if not Bilateral filter
		 //if(!bilateral_filter_kinfu)
		 //{
			// strcat(string_filename, "output\\");
		 //}

		// Read depth file name from sorted file
		fscanf(fp_1, "%s %s\n",remove_str, temp_str ); // ICP
		strcat(string_filename, temp_str);
	
		cv::Mat im = cv::imread(string_filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR );
		if (im.empty()) 
		{
			cout << "Cannot load image!" << endl;
			return;
		}

		source_depth_data_temp1_.resize(im.cols * im.rows);

		depth_temp.cols = im.cols;
		depth_temp.rows = im.rows;
		depth_temp.step = depth_temp.cols * depth_temp.elemSize();

		// do the for loop to read from this 
		for(int y = 0; y < im.rows; y++)
		{
			for(int x = 0; x < im.cols; x++)
			{
				//if(bilateral_filter_kinfu)
				//	source_depth_data_temp1_[y * im.cols + x]  = (unsigned short) im.at<unsigned short>(cv::Point(x, y));
				//else
					source_depth_data_temp1_[y * im.cols + x]  = temp = (unsigned short) im.at<unsigned short>(cv::Point(x, y));

					if(temp != 0)
					{
						if(temp > max)
							max = temp;
						if(temp < min)
							min = temp;
						if(temp > 3500)
						{
							count_less_than_4000++;
						}
						else if(temp > 3000)
						{
							count_less_than_3500++;
						}
						else if(temp > 2500)
						{
							count_less_than_3000++;
						}
						else if(temp > 2000)
						{
							count_less_than_2500++;
						}
						else if(temp > 1500)
						{
							count_less_than_2000++;
						}
						else if(temp > 1000)
						{
							count_less_than_1500++;
						}
						else 
						{
							count_less_than_1000++;
						}
					}
			}
		}

		number_less_than_1000 = count_less_than_1000;
		number_less_than_1500 = count_less_than_1500;
		number_less_than_2000 = count_less_than_2000;
		number_less_than_2500 = count_less_than_2500;
		number_less_than_3000 = count_less_than_3000;
		number_less_than_3500 = count_less_than_3500;
		number_less_than_4000 = count_less_than_4000;
		depth_min = min;
		depth_max = max;


		depth_temp.data = &source_depth_data_temp1_[0];		
#endif
}

// Load Indicator Map File
//void pcl::gpu::KinfuTracker::loadindFile(PtrStepSz<const unsigned char>& depth_temp)
//{
//#ifdef RANGA_MODIFICATION
//		// Open the PNG files one by one 
//
//		char temp_str[200];
//		char string_filename[500]; // = "C:\\Zhengyu\\3D Dataset\\rgbd_dataset_freiburg1_xyz\\"; // Indicator Map for 3D Reconstruction
//		char temp;
//
//		strcpy(string_filename, basepath_kinfu) ; // basepath_kinfu from base
//			
//		// if not Bilateral filter
//		 //if(!bilateral_filter_kinfu)
//		 //{
//			// strcat(string_filename, "output\\");
//		 //}
//
//		// Read depth file name from sorted file
//		fscanf(fp_2, "%s\n", temp_str ); // ICP
//		strcat(string_filename, temp_str);
//	
//		cv::Mat im = cv::imread(string_filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR );
//		if (im.empty()) 
//		{
//			cout << "Cannot load indicator map!" << endl;
//			return;
//		}
//
//		source_depth_data_temp2_.resize(im.cols * im.rows);
//
//		depth_temp.cols = im.cols;
//		depth_temp.rows = im.rows;
//		depth_temp.step = depth_temp.cols * depth_temp.elemSize();
//
//		// do the for loop to read from this 
//		for(int y = 0; y < im.rows; y++)
//		{
//			for(int x = 0; x < im.cols; x++)
//			{
//				//if(bilateral_filter_kinfu)
//				//	source_depth_data_temp1_[y * im.cols + x]  = (unsigned short) im.at<unsigned short>(cv::Point(x, y));
//				//else
//					source_depth_data_temp2_[y * im.cols + x]  = temp = (unsigned char) im.at<unsigned char>(cv::Point(x, y));
//			}
//		}
//				
//		depth_temp.data = &source_depth_data_temp2_[0];		
//#endif
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (DepthMap& depth_raw, IndicatorMap& depth_indicator, Eigen::Affine3f *hint)
{  
  device::Intr intr (fx_, fy_, cx_, cy_);
		intr.number_less_than_1000 = number_less_than_1000;
		intr.number_less_than_1500 = number_less_than_1500;
		intr.number_less_than_2000 = number_less_than_2000;
		intr.number_less_than_2500 = number_less_than_2500;
		intr.number_less_than_3000 = number_less_than_3000;
		intr.number_less_than_3500 = number_less_than_3500;
		intr.number_less_than_4000 = number_less_than_4000;
		intr.depth_max = depth_max;
		intr.depth_min = depth_min;

  if (!disable_icp_)
  {
      {
        //ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		  if(bilateral_filter_kinfu)
		  {
			device::bilateralFilter (depth_raw, depths_curr_[0]);
		  }
		  else
			depth_raw.copyTo(depths_curr_[0]);
        

#ifdef RANGA_MODIFICATION // Read Depth Information for Surface Recon (TSDF)
		// Open the PNG files one by one 

		char temp_str[200], remove_str[200];
		char string_filename[500]; // = "C:\\Zhengyu\\3D Dataset\\rgbd_dataset_freiburg1_xyz\\"; // Surface Recon

		strcpy(string_filename, basepath_kinfu) ;
		
		// For the latest experimentation
		strcat(string_filename, "output\\");

		// Read depth file name from sorted file
		fscanf(fp, "%s %s\n", remove_str, temp_str); // SR
		strcat(string_filename, temp_str);
	
		Mat im = imread(string_filename, IMREAD_ANYDEPTH | IMREAD_ANYCOLOR );
		if (im.empty()) 
		{
			cout << "Cannot load image!" << endl;
			return false;
		}

		source_depth_data_temp_.resize(im.cols * im.rows);

		depth_temp_.cols = im.cols;
		depth_temp_.rows = im.rows;
		depth_temp_.step = depth_temp_.cols * depth_temp_.elemSize();
		//depth_raw.download((void *)&source_depth_data_temp_[0], depth_temp_.step);
		// do the for loop to read from this 
		for(int y = 0; y < im.rows; y++)
		{
			for(int x = 0; x < im.cols; x++)
			{
				source_depth_data_temp_[y * im.cols + x]  = (unsigned short) im.at<unsigned short>(Point(x, y));
			}
		}

		depth_temp_.data = &source_depth_data_temp_[0];		
		//depth_temp_.data = im.ptr<unsigned short>();     
		
		depth_raw.upload(depth_temp_.data, depth_temp_.step, depth_temp_.rows, depth_temp_.cols); // this data is for surface reconstruction
#endif

        if (max_icp_distance_ > 0)
          device::truncateDepth(depths_curr_[0], max_icp_distance_);

        for (int i = 1; i < LEVELS; ++i)
          device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

        for (int i = 0; i < LEVELS; ++i)
        {
          device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
          //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
          computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
        }
        pcl::device::sync ();
      }

      //can't perform more on first frame
      if (global_time_ == 0)
      {
        Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

        Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
        float3& device_tcam = device_cast<float3>(init_tcam);

        Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
        Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
        float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

        //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);    
        device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);



        for (int i = 0; i < LEVELS; ++i)
          device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);

        ++global_time_;
        return (false);
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // Iterative Closest Point
      Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
      Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
      Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

      //Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
      Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
      float3& device_tprev     = device_cast<float3> (tprev);
      Matrix3frm Rcurr;
      Vector3f tcurr;
      if(hint)
      {
        Rcurr = hint->rotation().matrix();
        tcurr = hint->translation().matrix();
      }
      else
      {
        Rcurr = Rprev; // tranform to global coo for ith camera pose
        tcurr = tprev;
      }
      {
        //ScopeTime time("icp-all");
        for (int level_index = LEVELS-1; level_index>=0; --level_index)
        {
          int iter_num = icp_iterations_[level_index];

          MapArr& vmap_curr = vmaps_curr_[level_index];
          MapArr& nmap_curr = nmaps_curr_[level_index];

          //MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
          //MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

          MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
          MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

          //CorespMap& coresp = coresps_[level_index];

          for (int iter = 0; iter < iter_num; ++iter)
          {
            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
            Eigen::Matrix<double, 6, 1> b;
    #if 0
            device::tranformMaps(vmap_curr, nmap_curr, device_Rcurr, device_tcurr, vmap_g_curr, nmap_g_curr);
            findCoresp(vmap_g_curr, nmap_g_curr, device_Rprev_inv, device_tprev, intr(level_index), vmap_g_prev, nmap_g_prev, distThres_, angleThres_, coresp);
            device::estimateTransform(vmap_g_prev, nmap_g_prev, vmap_g_curr, coresp, gbuf_, sumbuf_, A.data(), b.data());

            //cv::gpu::GpuMat ma(coresp.rows(), coresp.cols(), CV_32S, coresp.ptr(), coresp.step());
            //cv::Mat cpu;
            //ma.download(cpu);
            //cv::imshow(names[level_index] + string(" --- coresp white == -1"), cpu == -1);
    #else
			device::Intr intr1 = intr (level_index);
			intr1.number_less_than_1000 = number_less_than_1000;
			intr1.number_less_than_1500 = number_less_than_1500;
			intr1.number_less_than_2000 = number_less_than_2000;
			intr1.number_less_than_2500 = number_less_than_2500;
			intr1.number_less_than_3000 = number_less_than_3000;
			intr1.number_less_than_3500 = number_less_than_3500;
			intr1.number_less_than_4000 = number_less_than_4000;
			intr1.depth_max = depth_max;
			intr1.depth_min = depth_min;
			estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr1,
                              vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
    #endif
            //checking nullspace
            double det = A.determinant ();

            if (fabs (det) < 1e-15 || pcl_isnan (det))
            {
              if (pcl_isnan (det)) cout << "qnan" << endl;

              reset ();
              return (false);
            }
            //float maxc = A.maxCoeff();

            Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
            //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

            float alpha = result (0);
            float beta  = result (1);
            float gamma = result (2);

            Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
            Vector3f tinc = result.tail<3> ();

            //compose
            tcurr = Rinc * tcurr + tinc;
            Rcurr = Rinc * Rcurr;
          }
        }
      }
      //save tranform
      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);
  } 
  else /* if (disable_icp_) */
  {
      if (global_time_ == 0)
        ++global_time_;

      Matrix3frm Rcurr = rmats_[global_time_ - 1];
      Vector3f   tcurr = tvecs_[global_time_ - 1];

      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);

  }

  Matrix3frm Rprev = rmats_[global_time_ - 1];
  Vector3f   tprev = tvecs_[global_time_ - 1];

  Matrix3frm Rcurr = rmats_.back();
  Vector3f   tcurr = tvecs_.back();

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move.  
  float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
  float tnorm = (tcurr - tprev).norm();  
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;

  if (disable_icp_)
    integrate = true;

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  Matrix3frm Rcurr_inv = Rcurr.inverse ();
  Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
  float3& device_tcurr = device_cast<float3> (tcurr);
  if (integrate)
  {
    //ScopeTime time("tsdf");
    //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
	integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
  {
    //ScopeTime time("ray-cast-all");
    raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }
    pcl::device::sync ();
  }

  ++global_time_;
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////// -> original KinfuTracker
bool
pcl::gpu::KinfuTracker::operator() (DepthMap& depth_raw, Eigen::Affine3f *hint)
{  
  device::Intr intr (fx_, fy_, cx_, cy_);
		intr.number_less_than_1000 = number_less_than_1000;
		intr.number_less_than_1500 = number_less_than_1500;
		intr.number_less_than_2000 = number_less_than_2000;
		intr.number_less_than_2500 = number_less_than_2500;
		intr.number_less_than_3000 = number_less_than_3000;
		intr.number_less_than_3500 = number_less_than_3500;
		intr.number_less_than_4000 = number_less_than_4000;
		intr.depth_max = depth_max;
		intr.depth_min = depth_min;

  if (!disable_icp_)
  {
      {
        //ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		  if(bilateral_filter_kinfu)
		  {
			device::bilateralFilter (depth_raw, depths_curr_[0]);
		  }
		  else
			depth_raw.copyTo(depths_curr_[0]);
        

#ifdef RANGA_MODIFICATION // Read Depth Information for Surface Recon (TSDF)
		// Open the PNG files one by one 

		char temp_str[200], remove_str[200];
		char string_filename[500]; // = "C:\\Zhengyu\\3D Dataset\\rgbd_dataset_freiburg1_xyz\\"; // Surface Recon

		strcpy(string_filename, basepath_kinfu) ;
		
		// For the latest experimentation
		strcat(string_filename, "output\\");

		// Read depth file name from sorted file
		fscanf(fp, "%s %s\n", remove_str, temp_str); // SR
		strcat(string_filename, temp_str);
	
		Mat im = imread(string_filename, IMREAD_ANYDEPTH | IMREAD_ANYCOLOR );
		if (im.empty()) 
		{
			cout << "Cannot load image!" << endl;
			return false;
		}

		source_depth_data_temp_.resize(im.cols * im.rows);

		depth_temp_.cols = im.cols;
		depth_temp_.rows = im.rows;
		depth_temp_.step = depth_temp_.cols * depth_temp_.elemSize();
		//depth_raw.download((void *)&source_depth_data_temp_[0], depth_temp_.step);
		// do the for loop to read from this 
		for(int y = 0; y < im.rows; y++)
		{
			for(int x = 0; x < im.cols; x++)
			{
				source_depth_data_temp_[y * im.cols + x]  = (unsigned short) im.at<unsigned short>(Point(x, y));
			}
		}

		depth_temp_.data = &source_depth_data_temp_[0];		
		//depth_temp_.data = im.ptr<unsigned short>();     
		
		depth_raw.upload(depth_temp_.data, depth_temp_.step, depth_temp_.rows, depth_temp_.cols); // this data is for surface reconstruction
#endif

        if (max_icp_distance_ > 0)
          device::truncateDepth(depths_curr_[0], max_icp_distance_);

        for (int i = 1; i < LEVELS; ++i)
          device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

        for (int i = 0; i < LEVELS; ++i)
        {
          device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
          //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
          computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
        }
        pcl::device::sync ();
      }

      //can't perform more on first frame
      if (global_time_ == 0)
      {
        Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

        Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
        float3& device_tcam = device_cast<float3>(init_tcam);

        Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
        Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
        float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

        //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);    
        device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);

        for (int i = 0; i < LEVELS; ++i)
          device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);

        ++global_time_;
        return (false);
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // Iterative Closest Point
      Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
      Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
      Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

      //Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
      Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
      float3& device_tprev     = device_cast<float3> (tprev);
      Matrix3frm Rcurr;
      Vector3f tcurr;
      if(hint)
      {
        Rcurr = hint->rotation().matrix();
        tcurr = hint->translation().matrix();
      }
      else
      {
        Rcurr = Rprev; // tranform to global coo for ith camera pose
        tcurr = tprev;
      }
      {
        //ScopeTime time("icp-all");
        for (int level_index = LEVELS-1; level_index>=0; --level_index)
        {
          int iter_num = icp_iterations_[level_index];

          MapArr& vmap_curr = vmaps_curr_[level_index];
          MapArr& nmap_curr = nmaps_curr_[level_index];

          //MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
          //MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

          MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
          MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

          //CorespMap& coresp = coresps_[level_index];

          for (int iter = 0; iter < iter_num; ++iter)
          {
            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
            Eigen::Matrix<double, 6, 1> b;
    #if 0
            device::tranformMaps(vmap_curr, nmap_curr, device_Rcurr, device_tcurr, vmap_g_curr, nmap_g_curr);
            findCoresp(vmap_g_curr, nmap_g_curr, device_Rprev_inv, device_tprev, intr(level_index), vmap_g_prev, nmap_g_prev, distThres_, angleThres_, coresp);
            device::estimateTransform(vmap_g_prev, nmap_g_prev, vmap_g_curr, coresp, gbuf_, sumbuf_, A.data(), b.data());

            //cv::gpu::GpuMat ma(coresp.rows(), coresp.cols(), CV_32S, coresp.ptr(), coresp.step());
            //cv::Mat cpu;
            //ma.download(cpu);
            //cv::imshow(names[level_index] + string(" --- coresp white == -1"), cpu == -1);
    #else
			device::Intr intr1 = intr (level_index);
			intr1.number_less_than_1000 = number_less_than_1000;
			intr1.number_less_than_1500 = number_less_than_1500;
			intr1.number_less_than_2000 = number_less_than_2000;
			intr1.number_less_than_2500 = number_less_than_2500;
			intr1.number_less_than_3000 = number_less_than_3000;
			intr1.number_less_than_3500 = number_less_than_3500;
			intr1.number_less_than_4000 = number_less_than_4000;
			intr1.depth_max = depth_max;
			intr1.depth_min = depth_min;
			estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr1,
                              vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
    #endif
            //checking nullspace
            double det = A.determinant ();

            if (fabs (det) < 1e-15 || pcl_isnan (det))
            {
              if (pcl_isnan (det)) cout << "qnan" << endl;

              reset ();
              return (false);
            }
            //float maxc = A.maxCoeff();

            Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
            //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

            float alpha = result (0);
            float beta  = result (1);
            float gamma = result (2);

            Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
            Vector3f tinc = result.tail<3> ();

            //compose
            tcurr = Rinc * tcurr + tinc;
            Rcurr = Rinc * Rcurr;
          }
        }
      }
      //save tranform
      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);
  } 
  else /* if (disable_icp_) */
  {
      if (global_time_ == 0)
        ++global_time_;

      Matrix3frm Rcurr = rmats_[global_time_ - 1];
      Vector3f   tcurr = tvecs_[global_time_ - 1];

      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);

  }

  Matrix3frm Rprev = rmats_[global_time_ - 1];
  Vector3f   tprev = tvecs_[global_time_ - 1];

  Matrix3frm Rcurr = rmats_.back();
  Vector3f   tcurr = tvecs_.back();

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move.  
  float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
  float tnorm = (tcurr - tprev).norm();  
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;

  if (disable_icp_)
    integrate = true;

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  Matrix3frm Rcurr_inv = Rcurr.inverse ();
  Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
  float3& device_tcurr = device_cast<float3> (tcurr);
  if (integrate)
  {
    //ScopeTime time("tsdf");
    //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
    integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
  {
    //ScopeTime time("ray-cast-all");
    raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }
    pcl::device::sync ();
  }

  ++global_time_;
  return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
pcl::gpu::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume& 
pcl::gpu::KinfuTracker::volume() const 
{ 
  return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume& 
pcl::gpu::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const ColorVolume& 
pcl::gpu::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume& 
pcl::gpu::KinfuTracker::colorVolume()
{
  return *color_volume_;
}
     
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  device::LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  device::convert (vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  device::convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::gpu::KinfuTracker::disableIcp() { disable_icp_ = true; }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{     
  color_volume_ = pcl::gpu::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool 
pcl::gpu::KinfuTracker::operator() (DepthMap& depth, const View& colors)
{ 
  bool res = (*this)(depth);

  if (res && color_volume_)
  {
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    device::Intr intr(fx_, fy_, cx_, cy_);

    Matrix3frm R_inv = rmats_.back().inverse();
    Vector3f   t     = tvecs_.back();
    
    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
    float3& device_tcurr = device_cast<float3> (t);
    
    device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
        colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }

  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    PCL_EXPORTS void 
    paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
    {
      device::paint3DView(rgb24, view, colors_weight);
    }

    PCL_EXPORTS void
    mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
    {
      const size_t size = min(cloud.size(), normals.size());
      output.create(size);

      const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
      const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
      const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
      device::mergePointNormal(c, n, o);           
    }

    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
    {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
      Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

      double rx = R(2, 1) - R(1, 2);
      double ry = R(0, 2) - R(2, 0);
      double rz = R(1, 0) - R(0, 1);

      double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
      double c = (R.trace() - 1) * 0.5;
      c = c > 1. ? 1. : c < -1. ? -1. : c;

      double theta = acos(c);

      if( s < 1e-5 )
      {
        double t;

        if( c > 0 )
          rx = ry = rz = 0;
        else
        {
          t = (R(0, 0) + 1)*0.5;
          rx = sqrt( std::max(t, 0.0) );
          t = (R(1, 1) + 1)*0.5;
          ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
          t = (R(2, 2) + 1)*0.5;
          rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

          if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
            rz = -rz;
          theta /= sqrt(rx*rx + ry*ry + rz*rz);
          rx *= theta;
          ry *= theta;
          rz *= theta;
        }
      }
      else
      {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
      }
      return Eigen::Vector3d(rx, ry, rz).cast<float>();
    }
  }
}
