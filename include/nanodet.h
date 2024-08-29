//
// Create by RangiLyu
// 2020 / 10 / 2
//

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "layer.h"
#include "net.h"
#include <benchmark.h>

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
}HeadInfo;

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

struct ObjectDetect
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class NanoDet
{
public:
    NanoDet(const char* param, const char* bin, bool useGPU);
    ~NanoDet();

    static NanoDet* detector;
    ncnn::Net* Net;
    static bool hasGPU;
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = { 320, 320 }; // input height and width
    int num_class = 1; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in theki training config. Default: 7.
    const int color_list[1][3] = { {216 , 82 , 24} };

    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);
    std::vector<std::string> labels{ "meter" };
    int resize_uniform(const cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area);
    void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi);
    void convert_boxes(const std::vector<BoxInfo>& bboxes, const object_rect& effect_roi, int src_w, int src_h, std::vector<ObjectDetect>& objects);
private:
    void preprocess(cv::Mat& image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
};


#endif //NANODET_H
