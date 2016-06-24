#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/*Here starts the code implemented by J.M. Facil-Ledesma*/
/*The similar structure loss is a function that not only caread about
 * the value of the pixel but also the gradient on x and y, ensuring a
 * similar local structure.*/
template <typename Dtype>
class SimilarStructureLossLayer : public LossLayer<Dtype> {
 public:
  explicit SimilarStructureLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
  //virtual void compute_grad_shape();
  virtual void setUp_x_conv();
  virtual void compute_x_gradient();
  virtual void compute_x_gradient_diff();
  virtual void setUp_y_conv();
  virtual void compute_y_gradient();
  virtual void compute_y_gradient_diff();
  virtual inline const char* type() const { return "SimilarStructureLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc SimilarStructureLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> grad_x_;
  Blob<Dtype> grad_y_;
  Blob<Dtype> ones_;
  vector<Blob<Dtype>*> grad_x_bottom;
  vector<Blob<Dtype>*> grad_x_top;
  vector<Blob<Dtype>*> grad_y_bottom;
  vector<Blob<Dtype>*> grad_y_top;
  vector<bool> propagate_down_conv_;
  vector<Dtype> sums;
  Layer<Dtype>* x_conv_;
  Layer<Dtype>* y_conv_;
  Dtype MSloss;
  Dtype sum_of_sq;
  Dtype SIloss;
  Dtype GXloss;
  Dtype GYloss;
  Dtype lambda;
  int height_, width_,channels_,num_;

};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
