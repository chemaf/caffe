#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/similar_structure_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{


/*template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::compute_grad_shape() {
  int height_out_ = (this->height_ + 2 * 0 - 1)/ 1 + 1;
  int width_out_ = (this->width_ + 2 * 0 - 3) / 1 + 1;
  grad_x_.Reshape(1, 1, height_out_,width_out_);
  this -> grad_x_count_=height_out_*width_out_;
  height_out_ = (this->height_ + 2 * 0 - 3)/ 1 + 1;
  width_out_ = (this->width_ + 2 * 0 - 1) / 1 + 1;
  grad_y_.Reshape(1, 1, height_out_,width_out_);
  this -> grad_y_count_=height_out_*width_out_;
}*/
template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::setUp_x_conv(){
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
	convolution_param->set_kernel_h(1);
	convolution_param->set_kernel_w(3);
	convolution_param->set_stride_h(1);
	convolution_param->set_stride_w(1);
	convolution_param->set_group(this->channels_);
	convolution_param->set_num_output(this->channels_);
	convolution_param->set_bias_term(false);
	x_conv_ = new ConvolutionLayer<Dtype>(layer_param);
	x_conv_->blobs().resize(1);
	x_conv_->blobs()[0].reset(new Blob<Dtype>(1, this->channels_, 1, 3));
	x_conv_->set_param_propagate_down(0,false);
	Dtype* weights = x_conv_->blobs()[0]->mutable_cpu_data();
	for (int c = 0; c < this->channels_; ++c) {
	  int i = c * 3;  // 1 x 3 filter
	  weights[i +  0] = -0.5;
	  weights[i +  1] =  0;
	  weights[i +  2] =  0.5;
	}
	grad_x_bottom.push_back(&diff_);
	grad_x_top.push_back(&grad_x_);
	x_conv_->SetUp(this->grad_x_bottom, this->grad_x_top);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::setUp_y_conv(){
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
	convolution_param->set_kernel_h(3);
	convolution_param->set_kernel_w(1);
	convolution_param->set_stride_h(1);
    convolution_param->set_stride_w(1);
	convolution_param->set_group(this->channels_);
	convolution_param->set_num_output(this->channels_);
	convolution_param->set_bias_term(false);
	y_conv_ = new ConvolutionLayer<Dtype>(layer_param);
	y_conv_->blobs().resize(1);
	y_conv_->blobs()[0].reset(new Blob<Dtype>(1, this->channels_, 3, 1));
	y_conv_->set_param_propagate_down(0,false);
	Dtype* weights = y_conv_->blobs()[0]->mutable_cpu_data();
	for (int c = 0; c < this->channels_; ++c) {
	  int i = c * 3;  // 3 x 1 filter
	  weights[i +  0] = -0.5;
	  weights[i +  1] =  0;
	  weights[i +  2] =  0.5;
	}
	grad_y_bottom.push_back(&diff_);
	grad_y_top.push_back(&grad_y_);
	y_conv_->SetUp(this->grad_y_bottom, this->grad_y_top);
	//printf("(%d,%d,%d,%d)\n",grad_x_.num(),grad_x_.channels(),grad_x_.height(),grad_x_.width());
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::compute_x_gradient(){
	x_conv_->Forward(grad_x_bottom, grad_x_top);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::compute_x_gradient_diff(){
	x_conv_->Backward(grad_x_top, this->propagate_down_conv_, grad_x_bottom);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::compute_y_gradient(){
	y_conv_->Forward(grad_y_bottom, grad_y_top);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::compute_y_gradient_diff(){
	y_conv_->Backward(grad_y_top, this->propagate_down_conv_, grad_y_bottom);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  diff_.Reshape(bottom[0]->shape());
  ones_.Reshape(bottom[0]->shape());
  Dtype *one = ones_.mutable_cpu_data();
  int count = bottom[0]->count();
  for(int i = 0; i<count; i++) one[i]=1;
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();
  num_ = bottom[0]->channels();
  sums.resize(num_,0);
  lambda = Dtype(-1)/Dtype(2)/(Dtype(height_*width_));
  setUp_x_conv();
  setUp_y_conv();
  this->propagate_down_conv_.push_back(true);
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 1) d = D - D' (it is saved in diff_)
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // 2) Calculate the mean squared error
  Dtype  dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  MSloss = dot / bottom[0]->num() / Dtype(2);
  // 2.1) Sub the scale invariant component
  sum_of_sq = 0;
  Dtype singular_sum;
  int i;
  int sum_count=channels_*height_*width_;
  for(int n = 0; n<num_; n++){
	  i=n*channels_*height_*width_;
	  singular_sum = caffe_cpu_dot(sum_count, &diff_.cpu_data()[i],&ones_.cpu_data()[i] );
	  sums[n]=singular_sum;
	  sum_of_sq+=singular_sum*singular_sum/bottom[0]->num() /Dtype(2);
  }
  SIloss = sum_of_sq;
  // 3) calculate the gradient of d in x and y
  compute_x_gradient();
  compute_y_gradient();
  // 4) sum the mean gradient squared error
  Dtype  dot_x =  caffe_cpu_dot(grad_x_.count(), grad_x_.cpu_data(), grad_x_.cpu_data());
  Dtype  dot_y =  caffe_cpu_dot(grad_y_.count(), grad_y_.cpu_data(), grad_y_.cpu_data());
  GXloss = (dot_x)/bottom[0]->num() / Dtype(2);
  GYloss = (dot_y)/bottom[0]->num() / Dtype(2);
  // 5) calculate the final loss
  top[0]->mutable_cpu_data()[0] = (MSloss+lambda*SIloss+GXloss+GYloss)/(Dtype(height_*width_));
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num()/(Dtype(height_*width_));
      // 1) Diff of the euclidian loss component (MSComponent)
      caffe_cpu_axpby(
          bottom[i]->count(),                 // count
          sign*alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());     // b
      // 2) Diff of the scale invariant component SIloss
      int sum_count=channels_*height_*width_;
      int j;
	  for(int n = 0; n<num_; n++){
		  j=n*channels_*height_*width_;
		  caffe_add_scalar(sum_count,sign*alpha*lambda*sums[n],&bottom[i]->mutable_cpu_diff()[j]);
	  }

      // 3) Diff of the gradient x component
      caffe_cpu_axpby(
                this->grad_x_top[0]->count(),                 // count
                alpha,                              // alpha
                this->grad_x_top[0]->cpu_data(),                   // a
                Dtype(0),                           // beta
                this->grad_x_top[0]->mutable_cpu_diff());     // b
      this->compute_x_gradient_diff();
      if(sign>0){
		  caffe_add(
			  bottom[i]->count(),                 // count
			  bottom[i]->cpu_diff(),
			  this->grad_x_bottom[0]->cpu_diff(),
			  bottom[i]->mutable_cpu_diff());
      }else{
    	  caffe_sub(
			  bottom[i]->count(),                 // count
			  bottom[i]->cpu_diff(),
			  this->grad_x_bottom[0]->cpu_diff(),
			  bottom[i]->mutable_cpu_diff());
      }
      // 4) Diff of the gradient y component
      caffe_cpu_axpby(
			  this->grad_y_top[0]->count(),                 // count
			  alpha,                              // alpha
			  this->grad_y_top[0]->cpu_data(),                   // a
			  Dtype(0),                           // beta
			  this->grad_y_top[0]->mutable_cpu_diff());     // b
      this->compute_y_gradient_diff();
		if(sign>0){
			caffe_add(
			  bottom[i]->count(),                 // count
			  bottom[i]->cpu_diff(),
			  this->grad_y_bottom[0]->cpu_diff(),
			  bottom[i]->mutable_cpu_diff());
		}else{
			caffe_sub(
			  bottom[i]->count(),                 // count
			  bottom[i]->cpu_diff(),
			  this->grad_y_bottom[0]->cpu_diff(),
			  bottom[i]->mutable_cpu_diff());
		}
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimilarStructureLossLayer);
#endif

INSTANTIATE_CLASS(SimilarStructureLossLayer);
REGISTER_LAYER_CLASS(SimilarStructureLoss);
} // namespace caffe
