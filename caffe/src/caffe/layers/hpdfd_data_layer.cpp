#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/layers/hpdfd_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
	HPDFDDataLayer<Dtype>::HPDFDDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param),
	reader_(param),
	hpdfd_transform_param_(param.hpdfd_transform_param()){
	}

template <typename Dtype>
	HPDFDDataLayer<Dtype>::~HPDFDDataLayer() {
		this->StopInternalThread();
	}

// set up the top blob shape 
template <typename Dtype>
void HPDFDDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	hpdfd_data_transformer_.reset(new HPDFDDataTransformer<Dtype>(hpdfd_transform_param_, this->phase_));
	hpdfd_data_transformer_->InitRand();
	bool debug = this->layer_param_.hpdfd_transform_param().debug();
  	if (debug)
  		FLAGS_v = 2;

	// Read a data point, and use it to initialize the top blob.
	Datum& datum = *(reader_.full().peek());
	LOG(INFO) << "Input data size: " << datum.height() << " " << datum.width() << " " << datum.channels();

	bool force_color = this->layer_param_.data_param().force_encoded_color();
	if ((force_color && DecodeDatum(&datum, true)) || DecodeDatumNative(&datum)) {
		LOG(INFO) << "Decoding Datum";
	}

	// read data parameter
	const int batch_size = this->layer_param_.data_param().batch_size();

	// crop	input
	const int height = this->layer_param_.hpdfd_transform_param().crop_size_y();
	const int width = this->layer_param_.hpdfd_transform_param().crop_size_x();
	// const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.hpdfd_transform_param().crop_size_y();
	// const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.hpdfd_transform_param().crop_size_x();
	LOG(INFO) << "PREFETCH_COUNT is " << this->PREFETCH_COUNT;
	top[0]->Reshape(batch_size, 3, height, width);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
		this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
	
	this->transformed_data_.Reshape(1, 3, height, width);
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	// label
	if (this->output_labels_) {
		const int downsample = this->layer_param_.hpdfd_transform_param().downsample();
		const int height = this->layer_param_.hpdfd_transform_param().crop_size_y();
		const int width = this->layer_param_.hpdfd_transform_param().crop_size_x();
		// const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.hpdfd_transform_param().crop_size_y();
		// const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.hpdfd_transform_param().crop_size_x();

		// int num_parts = this->layer_param_.hpdfd_transform_param().num_parts();
		int num_pts = this->layer_param_.hpdfd_transform_param().num_pts();
		top[1]->Reshape(batch_size, num_pts+1, height/downsample, width/downsample);			// +1 for additional background channel
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
			this->prefetch_[i].label_.Reshape(batch_size, num_pts+1, height/downsample, width/downsample);
		this->transformed_label_.Reshape(1, num_pts+1, height/downsample, width/downsample);
		LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
	}
}

// This function is called on prefetch thread, including load image and label, executing augmentation
template<typename Dtype>
void HPDFDDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double deque_time = 0;
	double decod_time = 0;
	double trans_time = 0;
	// static int cnt = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	// Reshape on single input batches for inputs of varying dimension.
	const int batch_size = this->layer_param_.data_param().batch_size();
	// const int crop_size = this->layer_param_.hpdfd_transform_param().crop_size();
	bool force_color = this->layer_param_.data_param().force_encoded_color();
	// if (batch_size == 1) {
	// 	Datum& datum = *(reader_.full().peek());
	// 	if (datum.encoded()) {
	// 		if (force_color) 
	// 			DecodeDatum(&datum, true);
	// 		else 
	// 			DecodeDatumNative(&datum);
	// 	}
	// 	batch->data_.Reshape(1, 3, datum.height(), datum.width());
	// 	// this->transformed_data_.Reshape(1, 3, datum.height(), datum.width());
	// }

	Dtype* top_data = batch->data_.mutable_cpu_data();
	Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

	if (this->output_labels_) 
		top_label = batch->label_.mutable_cpu_data();
	
	// get a blob
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		timer.Start();
		// LOG(INFO) << "LOAD DATA before";
		Datum& datum = *(reader_.full().pop("Waiting for data"));
		deque_time += timer.MicroSeconds();
		// LOG(INFO) << "LOAD DATA after";

		timer.Start();
		cv::Mat cv_img;
		if (datum.encoded()) {
			LOG(INFO) << "error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
			if (force_color) {
				cv_img = DecodeDatumToCVMat(datum, true);
			} else {
				cv_img = DecodeDatumToCVMatNative(datum);
			}
			if (cv_img.channels() != this->transformed_data_.channels()) {
				LOG(WARNING) << "Your dataset contains encoded images with mixed "
				<< "channel sizes. Consider adding a 'force_color' flag to the "
				<< "model definition, or rebuild your dataset using "
				<< "convert_imageset.";
			}
		}
		decod_time += timer.MicroSeconds();

		// Apply data transformations (mirror, scale, crop...)
		timer.Start();
		const int offset_data = batch->data_.offset(item_id);
		const int offset_label = batch->label_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset_data);
		this->transformed_label_.set_cpu_data(top_label + offset_label);
		// if (datum.encoded()) 
			// this->hpdfd_data_transformer_->Transform(cv_img, &(this->transformed_data_));
		// else 		// call hpdfd data transformer to generate label
		// LOG(INFO) << "encoded: " << datum.encoded();
		this->hpdfd_data_transformer_->Transform_nv(datum, &(this->transformed_data_), &(this->transformed_label_));
		
		// if (this->output_labels_) {
		//   top_label[item_id] = datum.label();
		// }
		trans_time += timer.MicroSeconds();
		reader_.free().push(const_cast<Datum*>(&datum));
	}
	batch_timer.Stop();

	VLOG(2) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	VLOG(2) << "  Dequeue time: " << deque_time / 1000 << " ms.";
	VLOG(2) << "   Decode time: " << decod_time / 1000 << " ms.";
	VLOG(2) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(HPDFDDataLayer);
REGISTER_LAYER_CLASS(HPDFDData);

}  // namespace caffe
