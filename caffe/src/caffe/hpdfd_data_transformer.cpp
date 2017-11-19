// Author: Xinshuo Weng
// Email: xinshuo.weng@gmail.com
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>

#include "caffe/hpdfd_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
//#include <omp.h>


namespace caffe {
// ********************************************************************external call functions************************************************************//
template<typename Dtype> HPDFDDataTransformer<Dtype>::HPDFDDataTransformer(const HPDFDTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
	// check if we want to use mean_file
	if (param_.has_mean_file()) {
		CHECK_EQ(param_.mean_value_size(), 0) <<
		"Cannot specify mean_file and mean_value at the same time";
		const string& mean_file = param.mean_file();
		if (Caffe::root_solver()) {
			LOG(INFO) << "Loading mean file from: " << mean_file;
		}
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
		data_mean_.FromProto(blob_proto);
	}

	// check if we want to use mean_value
	if (param_.mean_value_size() > 0) {
		CHECK(param_.has_mean_file() == false) <<
		"Cannot specify mean_file and mean_value at the same time";
		for (int c = 0; c < param_.mean_value_size(); ++c) {
			mean_values_.push_back(param_.mean_value(c));
		}
	}
	LOG(INFO) << "HPDFDDataTransformer constructor done.";
	num_pts = param_.num_pts();

	// print important information for debug
	VLOG(2) << " number of parts from dataset is " << num_pts;
	LOG(INFO) << INT_MAX;
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::Transform_nv(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) {
	// read parameter dimension
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	const int im_channels = transformed_data->channels();
	const int im_height = transformed_data->height();
	const int im_width = transformed_data->width();
	const int im_num = transformed_data->num();

	const int label_channels = transformed_label->channels();
	const int label_height = transformed_label->height();
	const int label_width = transformed_label->width();
	const int label_num = transformed_label->num();

	// parameter checking and logging
	VLOG(2) << "image shape: " << im_num << " " << im_channels << " " << im_height << " " << im_width;
	VLOG(2) << "label shape: " << label_num << " " << label_channels << " " << label_height << " " << label_width;
	VLOG(2) << "lmdb data shape: " << " " << datum_channels << " " << datum_height << " " << datum_width;
	CHECK_EQ(datum_channels, 4);
	CHECK_EQ(im_channels, 3);
	CHECK_EQ(im_num, label_num);
	// CHECK_LE(im_height, datum_height);
	// CHECK_EQ(im_width, datum_width);
	CHECK_GE(im_num, 1);

	//const int crop_size = param_.crop_size();
	// if (crop_size) {
	//   CHECK_EQ(crop_size, im_height);
	//   CHECK_EQ(crop_size, im_width);
	// } else {
	//   CHECK_EQ(datum_height, im_height);
	//   CHECK_EQ(datum_width, im_width);
	// }

	Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
	Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
	CPUTimer timer;
	timer.Start();
	Transform_nv(datum, transformed_data_pointer, transformed_label_pointer); // call function 1
	VLOG(2) << "Transform_nv: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label) {
	bool visualize_mode = param_.visualize();
	bool time_count = false;
	// LOG(INFO) << "encoded: " << datum.encoded();

  	// read parameters
	int clahe_tileSize_min = param_.clahe_tile_size_min();
	int clahe_tileSize_max = param_.clahe_tile_size_max();
	int clahe_clipLimit = param_.clahe_clip_limit();
	int crop_x = param_.crop_size_x();
	int crop_y = param_.crop_size_y();

	// read data
	const string& data = datum.data();
	// const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();
	//const int crop_size = param_.crop_size();
	//const Dtype scale = param_.scale();
	//const bool do_mirror = param_.mirror() && Rand(2);
	//const bool has_mean_file = param_.has_mean_file();
	const bool has_uint8 = data.size() > 0;
	//const bool has_mean_values = mean_values_.size() > 0;

  	// float targetDist = 41.0/35.0;
	AugmentSelection as = {false, 0.0, Size(), 0};

	// ************************ get the image from datum before any transformation
	float read_rgb_time, color_time, meta_time, aug_time;
	CPUTimer timer1;
	timer1.Start();
	Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);		
	int offset = img.rows * img.cols;
	int dindex;
	Dtype d_element;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b& rgb = img.at<Vec3b>(i, j);
			for (int c = 0; c < 3; c++) {
				dindex = c*offset + i*img.cols + j;
				if (has_uint8)
					d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
				else
					d_element = datum.float_data(dindex);
				rgb[c] = d_element;
			}
		}
	}
	read_rgb_time = timer1.MicroSeconds()/1000.0;
	timer1.Start();		

	// ****************************************** color, increase contrast
	if (param_.do_clahe()) {
		float clahe_dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 		//[0,1]
		if (clahe_dice > param_.clahe_prob()) {
			float tilesize_dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 		//[0,1]
			int clahe_tileSize = (int)((clahe_tileSize_max - clahe_tileSize_min) * tilesize_dice) + clahe_tileSize_min;
			clahe(img, clahe_tileSize, clahe_clipLimit);
		}
	}

	if (param_.gray() == 1) {
		float gray_dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 		//[0,1]
		if (gray_dice > param_.gray_prob()) {
			cv::cvtColor(img, img, CV_BGR2GRAY);
			cv::cvtColor(img, img, CV_GRAY2BGR);
		}
	}
	color_time = timer1.MicroSeconds()/1000.0;
	timer1.Start();   

	// *********************************************************** transformation
	int offset_num_pixels = 3 * offset;
	int offset_datum_width = datum_width;
	MetaData meta;
	ReadMetaData(meta, data, offset_num_pixels, offset_datum_width);
	// TransformMetaJoints(meta);			// transform the joint for different datasets to a same order
	meta_time = timer1.MicroSeconds()/1000.0;
	timer1.Start();

	// Start transforming for training
	Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
	Mat img_temp, img_temp2, img_temp3; // size determined by scale
	if (phase_ == TRAIN) {
		if (visualize_mode)
			visualize_image_with_pts(img, "original", meta);
		as.scale  = augmentation_scale(img, img_temp, meta);
		if (visualize_mode)
			visualize_image_with_pts(img_temp, "scaled", meta);		
		as.degree = augmentation_rotate(img_temp, img_temp2, meta);
		if (visualize_mode)
			visualize_image_with_pts(img_temp2, "rotated", meta);		
		as.crop   = augmentation_croppad(img_temp2, img_temp3, meta);
		if (visualize_mode)
			visualize_image_with_pts(img_temp3, "croppadded", meta);
		img_aug = img_temp3.clone();
		// as.flip   = augmentation_flip(img_temp3, img_aug, meta);		// too specific to use right now
		as.flip   = 0;
		VLOG(2) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height << "); flip:" << as.flip << "; degree: " << as.degree;
		// VLOG(2) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height << "); degree: " << as.degree;
	}
	else {
		img_aug = img.clone();
		as.scale = 1;
		as.crop = Size();
		as.flip = 0;
		as.degree = 0;
	}
	aug_time = timer1.MicroSeconds()/1000.0;
	timer1.Start();
	
	// visualize data augmentation
	if (visualize_mode) {
		visualize(img, "original", meta);
		visualize(img_temp, "augmentation_scale", meta);
		visualize(img_temp2, "augmentation_rotate", meta);
		visualize(img_temp3, "augmentation_croppad", meta);
		visualize(img_aug, "augmentation_flip", meta);
	}

	// copy transformed img (img_aug) into transformed_data, do the mean-subtraction and integer to float image here
	offset = img_aug.rows * img_aug.cols;
	for (int i = 0; i < img_aug.rows; ++i) {
		for (int j = 0; j < img_aug.cols; ++j) {
			Vec3b& rgb = img_aug.at<Vec3b>(i, j);
			// LOG(INFO) << rgb;
			transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128) / 255.0;
			transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128) / 255.0;
			transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128) / 255.0;
		}
	}
	// starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
	// if (visualize_mode) {
		// dumpEverything(transformed_data, transformed_label, meta);
	// }

	// ********************************************** generate label heatmap,  size is image size / downsample
	generateLabelMap(transformed_label, img_aug, meta);			// generate the last 57 channels, including 38 vector fields, 18 keypoints heatmap and 1 background

	// ********************************************** time measurement
	if (time_count) {
		VLOG(2) << "  Time measurement: ";
		VLOG(2) << "  rgb[:] = datum: " << read_rgb_time << " ms";
		VLOG(2) << "  color: " << color_time << " ms";
		VLOG(2) << "  ReadMeta+MetaJoints: " << meta_time << " ms";
		VLOG(2) << "  Aug: " << aug_time << " ms";
		VLOG(2) << "  putGauss+genLabel: " << timer1.MicroSeconds()/1000.0 << " ms";
	}
}


// ********************************************************************image transformation************************************************************//
template<typename Dtype>
float HPDFDDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, MetaData& meta) {
	float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 		//[0,1]
	float scale_multiplier;
	//float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
	if (dice > param_.scale_prob()) {
		img_temp = img_src.clone();
		scale_multiplier = 1;
	}
	else {
		float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); // linear shear into [scale_min, scale_max]
	}
	// float scale_abs = param_.target_dist() / meta.scale_self;
	float scale_abs = 1;
	float scale = scale_abs * scale_multiplier;
	resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);

	// modify meta data
	meta.pts_center *= scale;
	for (int i=0; i<meta.num_pts; i++)
		meta.pts.coordinate[i] *= scale;
	return scale_multiplier;
}

template<typename Dtype>
bool HPDFDDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
	if (p.x < 0 || p.y < 0) return false;
	if (p.x >= img_size.width || p.y >= img_size.height) return false;
	return true;
}

template<typename Dtype>
Size HPDFDDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, MetaData& meta) {
	float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
	float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
	int crop_x = param_.crop_size_x();
	int crop_y = param_.crop_size_y();

	float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
	float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

	//LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
	//LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
	Point2i center = meta.pts_center + Point2f(x_offset, y_offset);
	int offset_left = -(center.x - (crop_x/2));
	int offset_up = -(center.y - (crop_y/2));
	// int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
	// int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);

	img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
	for (int i=0;i<crop_y;i++) {
		for (int j=0;j<crop_x;j++) { //i,j on cropped
			int coord_x_on_img = center.x - crop_x/2 + j;
			int coord_y_on_img = center.y - crop_y/2 + i;
			if (onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))) 
				img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
		}
	}

	// modify meta data
	Point2f offset(offset_left, offset_up);
	meta.pts_center += offset;
	for (int i=0; i<meta.num_pts; i++) {
		meta.pts.coordinate[i] += offset;
	}
	return Size(x_offset, y_offset);
}

// template<typename Dtype>
// void HPDFDDataTransformer<Dtype>::swapLeftRight(Joints& j) {
// 	if (num_parts == 56) {
// 		int right[8] = {3,4,5, 9,10,11,15,17}; 
// 		int left[8] =  {6,7,8,12,13,14,16,18}; 
// 		for(int i=0; i<8; i++){    
// 			int ri = right[i] - 1;
// 			int li = left[i] - 1;
// 			Point2f temp = j.joints[ri];
// 			j.joints[ri] = j.joints[li];
// 			j.joints[li] = temp;
// 			int temp_v = j.isVisible[ri];
// 			j.isVisible[ri] = j.isVisible[li];
// 			j.isVisible[li] = temp_v;
// 		}
// 	}

// 	else if(num_parts == 43) {
// 		int right[6] = {3,4,5,9,10,11}; 
// 		int left[6] = {6,7,8,12,13,14}; 
// 		for (int i=0; i<6; i++) {   
// 			int ri = right[i] - 1;
// 			int li = left[i] - 1;
// 			Point2f temp = j.joints[ri];
// 			j.joints[ri] = j.joints[li];
// 			j.joints[li] = temp;
// 			int temp_v = j.isVisible[ri];
// 			j.isVisible[ri] = j.isVisible[li];
// 			j.isVisible[li] = temp_v;
// 		}
// 	}
// }

// template<typename Dtype>
// bool HPDFDDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, MetaData& meta) {
// 	bool doflip;
// 	if(param_.aug_way() == "rand"){
// 		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
// 		doflip = (dice <= param_.flip_prob());
// 	}
// 	else if(param_.aug_way() == "table"){
// 		doflip = (aug_flips[meta.data_index][meta.epoch % param_.num_total_augs()] == 1);
// 	}
// 	else {
// 		doflip = 0;
// 		LOG(ERROR) << "Unhandled exception!!!!!!";
// 	}

// 	if(doflip){
// 		flip(img_src, img_aug, 1);
// 		int w = img_src.cols;

// 		meta.pts_center.x = w - 1 - meta.pts_center.x;
// 		for(int i=0; i<meta.num_pts; i++){
// 			meta.pts[i].x = w - 1 - meta.pts[i].x;
// 		}
// 		if(param_.transform_body_joint())
// 			swapLeftRight(meta.joint_self);

// 		// for(int p=0; p<meta.numOtherPeople; p++){
// 		// 	meta.pts_center_other[p].x = w - 1 - meta.pts_center_other[p].x;
// 		// 	for(int i=0; i<np_in_lmdb; i++){
// 		// 		meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
// 		// 	}
// 		// 	if(param_.transform_body_joint())
// 		// 		swapLeftRight(meta.joint_others[p]);
// 		// }
// 	}
// 	else {
// 		img_aug = img_src.clone();
// 	}
// 	return doflip;
// }

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat R){
	Mat point(3,1,CV_64FC1);
	point.at<double>(0,0) = p.x;
	point.at<double>(1,0) = p.y;
	point.at<double>(2,0) = 1;
	Mat new_point = R * point;
	p.x = new_point.at<double>(0,0);
	p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
float HPDFDDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, MetaData& meta) {
	float degree;
	if (param_.aug_way() == "rand") {
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
	}
	else if (param_.aug_way() == "table") {
		degree = aug_degs[meta.data_index][meta.epoch % param_.num_total_augs()];
	}
	else {
		degree = 0;
		LOG(INFO) << "Unhandled exception!!!!!!";
	}

	Point2f center(img_src.cols/2.0, img_src.rows/2.0);
	Mat R = getRotationMatrix2D(center, degree, 1.0);
	Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
	
	// adjust transformation matrix
	R.at<double>(0,2) += bbox.width/2.0 - center.x;
	R.at<double>(1,2) += bbox.height/2.0 - center.y;
	//LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
	//          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
	warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));

	// adjust meta data
	RotatePoint(meta.pts_center, R);
	for (int i=0; i<meta.num_pts; i++) {
		RotatePoint(meta.pts.coordinate[i], R);
	}
	return degree;
}


// **************************************************************************label*******************************************************************//
template<typename Dtype>
void HPDFDDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int downsample, int grid_x, int grid_y, float sigma){
	float start = downsample/2.0 - 0.5; //0 if downsample = 1, 0.5 if downsample = 2, 1.5 if downsample = 4, ...
	for (int g_y = 0; g_y < grid_y; g_y++) {
		for (int g_x = 0; g_x < grid_x; g_x++) {
			float x = start + g_x * downsample;
			float y = start + g_y * downsample;
			float dist = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);		// >= 0
			float exponent = -dist / 2.0 / sigma / sigma;
			float pixel_value = exp(exponent);
			
			entry[g_y*grid_x + g_x] = pixel_value;
		}
	}
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::putLaplacianMaps(Dtype* entry, Point2f center, int downsample, int grid_x, int grid_y, float sigma){
	float start = downsample/2.0 - 0.5; 			//0 if downsample = 1, 0.5 if downsample = 2, 1.5 if downsample = 4, ...
	for (int g_y = 0; g_y < grid_y; g_y++) {
		for (int g_x = 0; g_x < grid_x; g_x++) {
			float x = start + g_x * downsample;
			float y = start + g_y * downsample;
			float dist = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);		// >= 0
			float exponent = -dist / 2.0 / sigma / sigma;
			float pixel_value = (1 + exponent) * exp(exponent);

			entry[g_y*grid_x + g_x] = pixel_value;
		}
	}
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat& img_aug, MetaData meta) {
	int rezX = img_aug.cols;
	int rezY = img_aug.rows;
	int downsample = param_.downsample();
	int grid_x = rezX / downsample;
	int grid_y = rezY / downsample;
	int channelOffset = grid_y * grid_x;

	float labelmap_type = param_.gaussian();
	float sigma = meta.sigma;

	// initialize zero to all points. 
	for (int g_y = 0; g_y < grid_y; g_y++) {
		for (int g_x = 0; g_x < grid_x; g_x++){
			for (int i = 0; i < meta.num_pts + 1; i++)			// include a background channel
				transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
		}
	}

	// put gaussian or laplacian maps on the labels
	for (int i = 0; i < meta.num_pts; i++) {		
		Point2f center = meta.pts.coordinate[i];
		if (meta.pts.isVisible[i] == 1 || meta.pts.isVisible[i] == 0) {			// occluded or visible 
			if (labelmap_type) {
				VLOG(2) << "gaussian label maps are used.....";
				putGaussianMaps(transformed_label + i*channelOffset, center, downsample, grid_x, grid_y, sigma); 	
			}
			else {
				VLOG(2) << "laplacian label maps are used.....";
				putLaplacianMaps(transformed_label + i*channelOffset, center, downsample, grid_x, grid_y, sigma); 	
			}
		}
	}

	for (int g_y = 0; g_y < grid_y; g_y++){					// put background channel the last channel
		for (int g_x = 0; g_x < grid_x; g_x++){
			float maximum = 0;
			for (int i = 0; i < meta.num_pts; i++)
				maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
			transformed_label[meta.num_pts*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
		}
	}

 	// visualize
	if (param_.visualize()) {
		Mat label_map;
		VLOG(2) << "save label map.....";
		for (int i = 0; i <= meta.num_pts; i++) {      
			Mat src_image = Mat::zeros(rezY, rezX, CV_8UC3);
			src_image = img_aug.clone();
			Mat vis_image = Mat::zeros(rezY, rezX, CV_32FC3);
			src_image.convertTo(vis_image, CV_32FC3, 1.0/255.0);
			label_map = Mat::zeros(grid_y, grid_x, CV_32FC1);
			for (int g_y = 0; g_y < grid_y; g_y++) {
				for (int g_x = 0; g_x < grid_x; g_x++)
					label_map.at<float>(g_y, g_x) = transformed_label[i*channelOffset + g_y*grid_x + g_x];
			}
			resize(label_map, label_map, Size(), downsample, downsample, INTER_LINEAR);			// resize the heatmap to the resolution as same as original input image
			applyColorMap(label_map, label_map, COLORMAP_JET);
			cvtColor(label_map, label_map, CV_GRAY2BGR);										// convert to 3 channels

			for (int o_y = 0; o_y < rezY; o_y++) {
				for (int o_x = 0; o_x < rezX; o_x++) {
					Vec3f& rgb_vis = vis_image.at<Vec3f>(o_y, o_x);
					Vec3f& rgb_label = label_map.at<Vec3f>(o_y, o_x);
					
					rgb_vis[0] = (rgb_label[0] + rgb_vis[0]) / 2.0;
					rgb_vis[1] = (rgb_label[1] + rgb_vis[1]) / 2.0;
					rgb_vis[2] = (rgb_label[2] + rgb_vis[2]) / 2.0;

					vis_image.at<Vec3f>(o_y, o_x) = rgb_vis;
				}
			}
			Mat save_image = Mat::zeros(rezY, rezX, CV_8UC3);
			vis_image.convertTo(save_image, CV_8UC3, 255);
			// addWeighted(label_map, 0.5, vis_image, 0.5, 0.0, label_map);

			// save
			char imagename [100];
			sprintf(imagename, "output/visualization/labelmap_writenumber_%04d_channel_%02d.jpg", meta.data_index, i);
			imwrite(imagename, save_image);
		}
	}
}

// ******************************************************************global help functions************************************************************//
template<typename Dtype>
void DecodeFloats(const string& data, size_t idx, Dtype* pf, size_t len) {
	memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
}

// specific reader to generate_LMDB.py
template<typename Dtype>
void HPDFDDataTransformer<Dtype>::ReadMetaData(MetaData& meta, const string& data, size_t offset_num_pixels, size_t offset_datum_width) { 	
	// define constant for offset
	const int nBytes_float = 4;
	// const int nBytes_bool = 1;
	// const int nBytes_uint8 = 1;
	int line_index = 0;

  	// 1st line: dataset name (string)
	meta.dataset = DecodeString(data, offset_num_pixels + line_index*offset_datum_width);	// dataset name
	line_index += 1;

  	// 2nd line: filename (string)
	meta.filename = DecodeString(data, offset_num_pixels + line_index*offset_datum_width);	// image path
	line_index += 1;

  	// 3rd line: image height (float), image width (float)
	float height, width;
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width, &height, 1);					// read image height
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width + nBytes_float, &width, 1);	// read image width
	meta.img_size = Size(width, height);
	line_index += 1;

  	// 4th line: data_index (float), num_data (float)
	float data_index;
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width, &data_index, 1);	// read number of image written
	meta.data_index = (int)data_index;
	float num_data;
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width + nBytes_float, &num_data, 1);	
	meta.num_data = (int)num_data;
	line_index += 1;

  	// count epochs according to counters
	static int cur_epoch = -1;
	if (meta.data_index == 0) {
		cur_epoch++;
	}
	meta.epoch = cur_epoch;

	// calculate current sigma
	meta.sigma = param_.sigma() - floor((meta.epoch) / param_.sigma_frequency()) * param_.sigma_interval();
	if (meta.sigma < 0.5) 		// make sure no less than 0.5
		meta.sigma = 0.5;

	if (meta.data_index % 1000 == 0) {
		LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
		<< "; meta.data_index: " << meta.data_index << "; meta.epoch: " << meta.epoch << "; size of variance: " << meta.sigma;
	}

	// 5th line: num_pts (float)
	float num_pts;
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width, &num_pts, 1);	
	meta.num_pts = (int)num_pts;
	line_index += 1;

	// 6th, 7th and 8th lines: points (2 * num_pts) (float) and visibility (1 * num_pts) (float)
	meta.pts.coordinate.resize(meta.num_pts);
	meta.pts.isVisible.resize(meta.num_pts);
	for (int i = 0; i < meta.num_pts; i++) {
		DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width + nBytes_float*i, &meta.pts.coordinate[i].x, 1);
		DecodeFloats(data, offset_num_pixels + (line_index+1)*offset_datum_width + nBytes_float*i, &meta.pts.coordinate[i].y, 1);
		meta.pts.coordinate[i] -= Point2f(1, 1); 							// from matlab 1-index to c++ 0-index
		
		float isVisible;										// -1: not annotated, 0: occluded, 1: visible
		DecodeFloats(data, offset_num_pixels + (line_index+2)*offset_datum_width + nBytes_float*i, &isVisible, 1);
		meta.pts.isVisible[i] = isVisible;
		if (meta.pts.coordinate[i].x < 0 || meta.pts.coordinate[i].y < 0 || meta.pts.coordinate[i].x >= meta.img_size.width || meta.pts.coordinate[i].y >= meta.img_size.height) {
			meta.pts.isVisible[i] = -1; 							// out of boundary
		}
	}	
	line_index += 3;

	// 9th line: center point (1 * 2) (float)
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width, &meta.pts_center.x, 1);
	DecodeFloats(data, offset_num_pixels + line_index*offset_datum_width + nBytes_float, &meta.pts_center.y, 1);
	meta.pts_center -= Point2f(1, 1);
	line_index += 1;
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::visualize(Mat& img, const string imagename, MetaData meta) {
	Mat img_vis = img.clone();
	char fullpath[100];
	sprintf(fullpath, "output/visualization/%s_epoch_%03d_index_%03d.jpg", imagename.c_str(), meta.epoch, meta.data_index);
	VLOG(2) << "save image to " << fullpath;
	imwrite(fullpath, img_vis);
}

template<typename Dtype>
void HPDFDDataTransformer<Dtype>::visualize_image_with_pts(Mat& img, const string imagename, MetaData meta) {
	Mat img_vis = img.clone();
	char fullpath[100];

	rectangle(img_vis, meta.pts_center - Point2f(3,3), meta.pts_center + Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);		// draw center
	// draw all points
	for (int i=0; i<num_pts; i++) {
        if (meta.pts.isVisible[i] == 1) 
            circle(img_vis, meta.pts.coordinate[i], 3, CV_RGB(0,0,255), -1);
		else if (meta.pts.isVisible[i] == 0)
			circle(img_vis, meta.pts.coordinate[i], 3, CV_RGB(200,200,255), -1);
	}

	// line(img_vis, meta.objpos + Point2f(-368/2,-368/2), meta.objpos + Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
	// line(img_vis, meta.objpos + Point2f(368/2,-368/2), meta.objpos + Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
	// line(img_vis, meta.objpos + Point2f(368/2,368/2), meta.objpos + Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
	// line(img_vis, meta.objpos + Point2f(-368/2,368/2), meta.objpos + Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);
	
	// save
	if (phase_ == TRAIN) {
		// rectangle(img_vis, Point(0, 0+img_vis.rows), Point(param_.crop_size_x(), param_.crop_size_y()+img_vis.rows), Scalar(255,255,255), 1);
		sprintf(fullpath, "output/visualization/%s_with_pts_epoch_%03d_index_%03d.jpg", imagename.c_str(), meta.epoch, meta.data_index);
	}
	else
		sprintf(fullpath, "output/visualization/%s_with_pts.jpg", imagename.c_str());
	VLOG(2) << "save image to " << fullpath;
	imwrite(fullpath, img_vis);
}

// increase the contrast of the images
template <typename Dtype>     
void HPDFDDataTransformer<Dtype>::clahe(Mat& bgr_image, int tileSize, int clipLimit) {
	Mat lab_image;
	cvtColor(bgr_image, lab_image, CV_BGR2Lab);

	// Extract the L channel
	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

	// apply the CLAHE algorithm to the L channel
	Ptr<CLAHE> clahe = createCLAHE(clipLimit, Size(tileSize, tileSize));
	//clahe->setClipLimit(4);
	Mat dst;
	clahe->apply(lab_planes[0], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	merge(lab_planes, lab_image);

	// convert back to RGB
	Mat image_clahe;
	cvtColor(lab_image, image_clahe, CV_Lab2BGR);
	bgr_image = image_clahe.clone();
}


template <typename Dtype>
void HPDFDDataTransformer<Dtype>::InitRand() {
	const bool needs_rand = param_.mirror() ||
	(phase_ == TRAIN && param_.crop_size());
	if (needs_rand) {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	} else {
		rng_.reset();
	}
}

template <typename Dtype>
int HPDFDDataTransformer<Dtype>::Rand(int n) {
	CHECK(rng_);
	CHECK_GT(n, 0);
	caffe::rng_t* rng =
	static_cast<caffe::rng_t*>(rng_->generator());
	return ((*rng)() % n);
}


INSTANTIATE_CLASS(HPDFDDataTransformer);

}  // namespace caffe
