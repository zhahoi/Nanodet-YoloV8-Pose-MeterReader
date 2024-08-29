#include "nanodet.h"
#include "yolov8_pose.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define DET_PARAM  "/home/hit/Project/MeterReader_Image/weights/nanodet.param"
#define DET_BIN "/home/hit/Project/MeterReader_Image/weights/nanodet.bin"
#define SEG_PARAM "/home/hit/Project/MeterReader_Image/weights/fastseg.param"
#define SEG_BIN "/home/hit/Project/MeterReader_Image/weights/fastseg.bin"
#define SAVE_DIR "/home/hit/Project/MeterReader_Image/outputs/"

#define DET_PARAM  "C:/CPlusPlus/HuaRayDemo/weights/nanodet.param"
#define DET_BIN "C:/CPlusPlus/HuaRayDemo/weights/nanodet.bin"
#define YOLOV8_PARAM "C:/CPlusPlus/HuaRayDemo/weights/yolov8s-pose-opt.param"
#define YOLOV8_BIN "C:/CPlusPlus/HuaRayDemo/weights/yolov8s-pose-opt.bin"

std::unique_ptr<NanoDet> nanoDet(new NanoDet(DET_PARAM, DET_BIN, false));
std::unique_ptr<Yolov8Pose> yolov8Pose(new Yolov8Pose(YOLOV8_PARAM, YOLOV8_BIN, false));

#define SAVE_IMG

void detectProcess(const cv::Mat& input_image) {
    cv::Mat img_copy = input_image.clone();
    std::vector<ObjectDetect> objects;
    object_rect effect_roi;
    cv::Mat resized_img;

    nanoDet->resize_uniform(img_copy, resized_img, cv::Size(320, 320), effect_roi);

    auto results = nanoDet->detect(resized_img, 0.3, 0.3);  // 0.4, 0.3
    nanoDet->convert_boxes(results, effect_roi, img_copy.cols, img_copy.rows, objects);

    std::cout << "Object size: " << objects.size() << std::endl;

    if (!objects.empty() && yolov8Pose->isValidROI(objects))
    {
        std::vector<cv::Mat> meters_image = yolov8Pose->cut_roi_img(img_copy, objects);
        // �Բü����Ǳ��������ж���
        std::vector<float> scale_values;
        bool isGet = yolov8Pose->get_results(meters_image, scale_values);

        if (isGet)
        {
            cv::Mat save_frame = yolov8Pose->result_visualizer(img_copy, objects, scale_values);
            cv::imshow("image", save_frame);
        }
        else
        {
            cv::imshow("image", img_copy);
        }

        // nanoDet->draw_bboxes(img_copy, results, effect_roi);
        // cv::imshow("Frame", img_copy);
        cv::waitKey(0);   // �ȴ�1ms�Դ���opencv������Ϣ
    }
    else
    {
        cv::imshow("image", img_copy);
        cv::waitKey(0);   // �ȴ�1ms�Դ���opencv������Ϣ
    }

    return;
}

int main(int argc, char* argv[]) {
    
        // ����ͼƬ·��
    std::string path = "";

    cv::Mat image = cv::imread(path);

    if (image.empty()) {
        std::cerr << "Could not open or find the image at " << path << std::endl;
        return -1;
    }

    // ��¼��ʼʱ��
    // auto start = std::chrono::high_resolution_clock::now();

    // ���� detectProcess ����
    detectProcess(image);

    // ��¼����ʱ��
    // auto end = std::chrono::high_resolution_clock::now();

    // �������ʱ��
    // std::chrono::duration<double, std::milli> elapsed = end - start;

    // �������ʱ��
    // std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Processing complete" << std::endl;
    return 0;
}
