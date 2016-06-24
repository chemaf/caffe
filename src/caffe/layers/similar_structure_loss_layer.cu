#include <vector>

#include "caffe/layers/similar_structure_loss_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// 1) d = D - D' (it is saved in diff_)
	  int count = bottom[0]->count();
	  caffe_gpu_sub(
	      count,
	      bottom[0]->gpu_data(),
	      bottom[1]->gpu_data(),
	      diff_.mutable_gpu_data());
	  // 2) Calculate the mean squared error
	  Dtype  dot;
	  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	  MSloss = dot / bottom[0]->num() / Dtype(2);
	  //printf("MSLOS=%f. RMSE = %f\n",MSloss,sqrt(MSloss/(Dtype(height_*width_))));
	  // 2.1) Sub the scale invariant component
	  sum_of_sq = 0;
	  Dtype singular_sum;
	  int i;
	  int sum_count=channels_*height_*width_;
	  for(int n = 0; n<num_; n++){
	    i=n*channels_*height_*width_;
	    caffe_gpu_dot(sum_count, &diff_.gpu_data()[i],&ones_.gpu_data()[i], &singular_sum );
	    sums[n]=singular_sum;
	    sum_of_sq+=singular_sum*singular_sum/bottom[0]->num() /Dtype(2);
	  }
	  SIloss = sum_of_sq;

	  // 3) calculate the gradient of d in x and y
	  compute_x_gradient();
	  compute_y_gradient();
	  // 4) sum the mean gradient squared error
	  Dtype  dot_x;
	  caffe_gpu_dot(grad_x_.count(), grad_x_.gpu_data(), grad_x_.gpu_data(),&dot_x);
	  Dtype  dot_y;
	  caffe_gpu_dot(grad_y_.count(), grad_y_.gpu_data(), grad_y_.gpu_data(),&dot_y);
	  GXloss = (dot_x)/bottom[0]->num() / Dtype(2);
	  GYloss = (dot_y)/bottom[0]->num() / Dtype(2);
	  // 5) calculate the final loss
	  top[0]->mutable_cpu_data()[0] = (MSloss+lambda*SIloss+GXloss+GYloss)/(Dtype(height_*width_));
}

template <typename Dtype>
void SimilarStructureLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num() /(Dtype(height_*width_));

      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          sign * alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
      	//alpha = top[0]->cpu_diff()[0] / bottom[i]->num();
      // 2) Diff of the scale invariant component SIloss
	   int sum_count=channels_*height_*width_;
	    int j;
	   for(int n = 0; n<num_; n++){
		   j=n*channels_*height_*width_;
		   caffe_gpu_add_scalar(sum_count,sign*alpha*lambda*sums[n],&bottom[i]->mutable_gpu_diff()[j]);
	   }


      	caffe_gpu_axpby(
				  this->grad_x_top[0]->count(),                 // count
				  alpha,                              // alpha
				  this->grad_x_top[0]->gpu_data(),                   // a
				  Dtype(0),                           // beta
				  this->grad_x_top[0]->mutable_gpu_diff());     // b
		this->compute_x_gradient_diff();
		if(sign>0){
		  caffe_gpu_add(
			  bottom[i]->count(),                 // count
			  bottom[i]->gpu_diff(),
			  this->grad_x_bottom[0]->gpu_diff(),
			  bottom[i]->mutable_gpu_diff());
		}else{
		  caffe_gpu_sub(
			  bottom[i]->count(),                 // count
			  bottom[i]->gpu_diff(),
			  this->grad_x_bottom[0]->gpu_diff(),
			  bottom[i]->mutable_gpu_diff());
		}
		//this->grad_y_top[0]->mutable_gpu_diff();
		//printf("count=%d\n",grad_y_top[0]->count());
		caffe_gpu_axpby(
			  this->grad_y_top[0]->count(),                 // count
			  alpha,                              // alpha
			  this->grad_y_top[0]->gpu_data(),                   // a
			  Dtype(0),                           // beta
			  this->grad_y_top[0]->mutable_gpu_diff());     // b
		this->compute_y_gradient_diff();
		if(sign>0){
			caffe_gpu_add(
			  bottom[i]->count(),                 // count
			  bottom[i]->gpu_diff(),
			  this->grad_y_bottom[0]->gpu_diff(),
			  bottom[i]->mutable_gpu_diff());
		}else{
			caffe_gpu_sub(
			  bottom[i]->count(),                 // count
			  bottom[i]->gpu_diff(),
			  this->grad_y_bottom[0]->gpu_diff(),
			  bottom[i]->mutable_gpu_diff());
		}
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SimilarStructureLossLayer);

}  // namespace caffe
