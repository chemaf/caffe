#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/similar_structure_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SimilarStructureLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SimilarStructureLossLayerTest()
      : blob_bottom_data1_(new Blob<Dtype>(1, 1, 10, 11)),
        blob_bottom_data2_(new Blob<Dtype>(1, 1, 10, 11)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data1_);
    blob_bottom_vec_.push_back(blob_bottom_data1_);
    filler.Fill(this->blob_bottom_data2_);
    blob_bottom_vec_.push_back(blob_bottom_data2_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SimilarStructureLossLayerTest() {
    delete blob_bottom_data1_;
    delete blob_bottom_data2_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    SimilarStructureLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    SimilarStructureLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-4;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data1_;
  Blob<Dtype>* const blob_bottom_data2_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SimilarStructureLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SimilarStructureLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SimilarStructureLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SimilarStructureLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
