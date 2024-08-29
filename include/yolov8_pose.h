#ifndef YOLOV8_POSE_H
#define YOLOV8_POSE_H

#pragma warning(disable:4996)

#include "layer.h"
#include "net.h"
#include "nanodet.h"
#include "opencv2/opencv.hpp"

#include <float.h>
#include <stdio.h>
#include <vector>

#define MAX_STRIDE 32 // if yolov8-p6 model modify to 64
#define SCALE_BEGINNING 0 // -0.1
#define SCALE_END  1.0

const int target_size = 320;
const float prob_threshold = 0.25f;
const float nms_threshold = 0.45f;
const float aspectRatioThreshold = 0.90f;

const std::vector<std::vector<unsigned int>> KPS_COLORS =
{ {0,   255, 0},
 {255, 128, 0},
 {51,  153, 255},
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> kps;
};

struct meterPoints
{
    cv::Point2f pointer_locations;
    cv::Point2f scale_locations[2];
    cv::Point2f center_location;
};

static const char* class_names[] = {
        "pointer_rect",
        "left_rect",
        "right_rect",
};


class Yolov8Pose
{
public:
    Yolov8Pose(const char* param, const char* bin, bool useGPU);
    ~Yolov8Pose();

    static Yolov8Pose* yolov8_Detector;
    ncnn::Net* yolov8_Net;
    static bool hasGPU;

    int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects); 
    bool process_objects(const cv::Mat& image, const std::vector<Object>& objs, const std::vector<std::vector<unsigned int>>& KPS_COLORS, float& scale_value);
    bool get_results(const std::vector<cv::Mat>& meters_image, std::vector<float>& meterScales);
    cv::Mat result_visualizer(const cv::Mat& bgr, const std::vector<ObjectDetect>& objects_remains, const std::vector<float> scale_values);
    std::vector<cv::Mat> cut_roi_img(const cv::Mat& bgr, const std::vector<ObjectDetect>& objects);
    std::string floatToString(const float& val);
    bool isValidROI(const std::vector<ObjectDetect>& objects);

private:
    cv::Point2f getPointerPoint(cv::Point2f center_point, cv::Vec4i pointer_line);
    bool getPointerLines(const cv::Mat& img, const cv::Rect_<float>& rect, cv::Vec4i& l_line);
    bool Thining_Rosenfeld(cv::Mat& src, cv::Mat& dst);
    void circleCenter(cv::Point2f point2, cv::Point2f point1, cv::Vec4i line3, cv::Point2f& center_point);

    float getAngleRatio(cv::Point2f& start_locations, cv::Point2f& end_locations, cv::Point2f& pointer_head_location, cv::Point2f& center_location);
    float getScaleValue(float angleRatio);

    float sigmod(const float in);
    float softmax(const float* src, float* dst, int length);
    void generate_proposals(int stride, const ncnn::Mat& feat_blob, const float prob_threshold, std::vector<Object>& objects);
    float clamp(float val, float min = 0.f, float max = 1280.f);
    void non_max_suppression(std::vector<Object>& proposals, std::vector<Object>& results, int orin_h, int orin_w, float dh = 0, float dw = 0, float ratio_h = 1.0f, float ratio_w = 1.0f, float conf_thres = 0.25f, float iou_thres = 0.65f);
};


#endif // YOLOV8_POSE_H