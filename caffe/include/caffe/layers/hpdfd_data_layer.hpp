#ifndef CAFFE_HPDFD_DATA_LAYER_HPP_
#define CAFFE_HPDFD_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/hpdfd_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class HPDFDDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit HPDFDDataLayer(const LayerParameter& param);
  virtual ~HPDFDDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // HPDFDDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "HPDFDData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
  Blob<Dtype> transformed_label_; // add another blob

  HPDFDTransformationParameter hpdfd_transform_param_;
  shared_ptr<HPDFDDataTransformer<Dtype> > hpdfd_data_transformer_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
