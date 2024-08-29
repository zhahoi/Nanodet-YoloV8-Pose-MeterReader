#include "yolov8_pose.h"
// #define VISUALIZER

bool Yolov8Pose::hasGPU = true;
Yolov8Pose* Yolov8Pose::yolov8_Detector = nullptr;


Yolov8Pose::Yolov8Pose(const char* param, const char* bin, bool useGPU)
{
	this->yolov8_Net = new ncnn::Net();
	// opt
#if NCNN_VULKAN
	this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
	this->yolov8_Net->opt.use_vulkan_compute = this->hasGPU && useGPU;
	this->yolov8_Net->opt.use_fp16_arithmetic = false;
	this->yolov8_Net->opt.num_threads = 4;
	this->yolov8_Net->load_param(param);
	this->yolov8_Net->load_model(bin);
}


Yolov8Pose::~Yolov8Pose()
{
	delete this->yolov8_Net;
}


std::vector<cv::Mat> Yolov8Pose::cut_roi_img(const cv::Mat& bgr, const std::vector<ObjectDetect>& objects)
{
	std::vector<cv::Mat> cut_images;
	cut_images.clear();
	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const ObjectDetect& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		// 限制 ROI 在图像边界内
		cv::Rect roi = obj.rect;
		roi = roi & cv::Rect(0, 0, image.cols, image.rows);  // 修正为 intersection

		// 如果调整后的 ROI 仍有有效区域，则进行裁剪
		if (roi.width > 0 && roi.height > 0)
		{
			cv::Mat cut_image = image(roi);
			cut_images.push_back(cut_image);

#ifdef VISUALIZE
			cv::imshow("sub_image", cut_image);
			cv::waitKey(0);
#endif // VISUALIZE
		}
	}
	return cut_images;
}




float Yolov8Pose::getAngleRatio(cv::Point2f& start_locations, cv::Point2f& end_locations, cv::Point2f& pointer_head_location, cv::Point2f& center_location)
{
	// 刻度开始点与x轴正方向的夹角，刻度结束点与x轴正方向的夹角，刻度开始点与刻度结束点的夹角
	float beginning_x_angle = atan2(center_location.y - start_locations.y,
		start_locations.x - center_location.x);
	float end_x_angle = atan2(center_location.y - end_locations.y,
		end_locations.x - center_location.x);
	float beginning_end_angle = 2 * CV_PI - (end_x_angle - beginning_x_angle);

	float pointer_x_angle = atan2(center_location.y - pointer_head_location.y,
		pointer_head_location.x - center_location.x);
	float beginning_pointer_angle = 0;
	if (pointer_head_location.y > center_location.y && pointer_head_location.x < center_location.x) {
		beginning_pointer_angle = pointer_x_angle - beginning_x_angle;
	}
	else {
		beginning_pointer_angle = 2 * CV_PI - (pointer_x_angle - beginning_x_angle);
	}

	float angleRatio = beginning_pointer_angle / beginning_end_angle;
	return angleRatio;
}


float Yolov8Pose::getScaleValue(float angleRatio)
{
	float range = SCALE_END - SCALE_BEGINNING;
	float value = range * angleRatio + SCALE_BEGINNING;
	return value;
}


float Yolov8Pose::sigmod(const float in)
{
	return 1.f / (1.f + expf(-1.f * in));
}


float Yolov8Pose::softmax(const float* src, float* dst, int length)
{
	float alpha = -FLT_MAX;
	for (int c = 0; c < length; c++)
	{
		float score = src[c];
		if (score > alpha)
		{
			alpha = score;
		}
	}

	float denominator = 0;
	float dis_sum = 0;
	for (int i = 0; i < length; ++i)
	{
		dst[i] = expf(src[i] - alpha);
		denominator += dst[i];
	}
	for (int i = 0; i < length; ++i)
	{
		dst[i] /= denominator;
		dis_sum += i * dst[i];
	}
	return dis_sum;
}


void Yolov8Pose::generate_proposals(int stride, const ncnn::Mat& feat_blob, const float prob_threshold, std::vector<Object>& objects)
{
	const int reg_max = 16;
	float dst[16];
	const int num_w = feat_blob.w;
	const int num_grid_y = feat_blob.c;
	const int num_grid_x = feat_blob.h;

	// std::cout << "Res shape: " << num_w << ", " << num_grid_y << ", " << num_grid_x << std::endl;

	const int kps_num = 2;
	const int num_class = 3;

	for (int i = 0; i < num_grid_y; i++) {
		for (int j = 0; j < num_grid_x; j++) {
			const float* matat = feat_blob.channel(i).row(j);

			float score = 0;
			int clsid = -1;
			for (int c = 0; c < num_class; c++) {
				float cls_score = sigmod(matat[c]);
				if (cls_score > score) {
					score = cls_score;
					clsid = c;
				}
			}

			if (score < prob_threshold) {
				continue;
			}

			float x0 = j + 0.5f - softmax(matat + num_class, dst, reg_max);
			float y0 = i + 0.5f - softmax(matat + (num_class + reg_max), dst, reg_max);
			float x1 = j + 0.5f + softmax(matat + (num_class + 2 * reg_max), dst, reg_max);
			float y1 = i + 0.5f + softmax(matat + (num_class + 3 * reg_max), dst, reg_max);

			x0 *= stride;
			y0 *= stride;
			x1 *= stride;
			y1 *= stride;

			std::vector<float> kps;
			for (int k = 0; k < kps_num; k++) {
				float kps_x = (matat[num_class + 4 * reg_max + k * 3] * 2.f + j) * stride;
				float kps_y = (matat[num_class + 4 * reg_max + k * 3 + 1] * 2.f + i) * stride;
				float kps_s = sigmod(matat[num_class + 4 * reg_max + k * 3 + 2]);

				kps.push_back(kps_x);
				kps.push_back(kps_y);
				kps.push_back(kps_s);
			}

			Object obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = clsid;
			obj.prob = score;
			obj.kps = kps;
			objects.push_back(obj);
		}
	}
}


float Yolov8Pose::clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}


void Yolov8Pose::non_max_suppression(std::vector<Object>& proposals, std::vector<Object>& results, int orin_h, int orin_w, float dh, float dw, float ratio_h, float ratio_w, float conf_thres, float iou_thres)
{
	results.clear();
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> labels;
	std::vector<int> indices;
	std::vector<std::vector<float>> kpss;

	for (auto& pro : proposals)
	{
		bboxes.push_back(pro.rect);
		scores.push_back(pro.prob);
		labels.push_back(pro.label);
		kpss.push_back(pro.kps);
	}

	cv::dnn::NMSBoxes(
		bboxes,
		scores,
		conf_thres,
		iou_thres,
		indices
	);

	for (auto i : indices)
	{
		auto& bbox = bboxes[i];
		float x0 = bbox.x;
		float y0 = bbox.y;
		float x1 = bbox.x + bbox.width;
		float y1 = bbox.y + bbox.height;
		float& score = scores[i];
		int& label = labels[i];

		x0 = (x0 - dw) / ratio_w;
		y0 = (y0 - dh) / ratio_h;
		x1 = (x1 - dw) / ratio_w;
		y1 = (y1 - dh) / ratio_h;

		x0 = clamp(x0, 0.f, orin_w);
		y0 = clamp(y0, 0.f, orin_h);
		x1 = clamp(x1, 0.f, orin_w);
		y1 = clamp(y1, 0.f, orin_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
		obj.kps = kpss[i];
		for (int n = 0; n < obj.kps.size(); n += 3)
		{
			obj.kps[n] = clamp((obj.kps[n] - dw) / ratio_w, 0.f, orin_w);
			obj.kps[n + 1] = clamp((obj.kps[n + 1] - dh) / ratio_h, 0.f, orin_h);
		}

		results.push_back(obj);
	}
}


int Yolov8Pose::detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects)
{
	// ncnn::Net preprocess
	int img_w = bgr.cols;
	int img_h = bgr.rows;

	// letterbox pad to multiple of MAX_STRIDE
	int w = img_w;
	int h = img_h;
	float scale = 1.f;
	if (w > h)
	{
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else
	{
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

	// pad to target_size rectangle
	// ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox
	int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	int top = hpad / 2;
	int bottom = hpad - hpad / 2;
	int left = wpad / 2;
	int right = wpad - wpad / 2;

	ncnn::Mat in_pad;

	ncnn::copy_make_border(in,
		in_pad,
		top,
		bottom,
		left,
		right,
		ncnn::BORDER_CONSTANT,
		114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = yolov8_Net->create_extractor();

	ex.input("images", in_pad);

	std::vector<Object> proposals;

	// stride 8
	{
		ncnn::Mat out;
		ex.extract("output0", out);

		std::vector<Object> objects8;
		generate_proposals(8, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}

	// stride 16
	{
		ncnn::Mat out;
		ex.extract("378", out);  // 378 905

		std::vector<Object> objects16;
		generate_proposals(16, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}

	// stride 32
	{
		ncnn::Mat out;
		ex.extract("403", out);  // 403 930
		std::vector<Object> objects32;
		generate_proposals(32, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}

	non_max_suppression(proposals, objects,
		img_h, img_w, hpad / 2, wpad / 2,
		scale, scale, prob_threshold, nms_threshold);


	return 0;
}


bool Yolov8Pose::process_objects(const cv::Mat& image, const std::vector<Object>& objs, const std::vector<std::vector<unsigned int>>& KPS_COLORS, float& scale_value)
{
	cv::Mat res = image.clone();
	// std::cout << "image size: " << res.size() << std::endl;

	const int num_point = 2;

	bool getLines = false;
	cv::Vec4i pointer_line;
	cv::Point2f start_point, end_point, center_point, pointer_point, center_point_ori;

	for (auto& obj : objs) {

#ifdef VISUALIZER
		cv::rectangle(
			res,
			obj.rect,
			{ 0, 0, 255 },
			2
		);

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		std::cout << "current class_names: " << class_names[obj.label] << std::endl;

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(
			text,
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			1,
			&baseLine
		);

		int x = (int)obj.rect.x;
		int y = (int)obj.rect.y + 1;

		if (y > res.rows)
			y = res.rows;

		cv::rectangle(
			res,
			cv::Rect(x, y, label_size.width, label_size.height + baseLine),
			{ 0, 0, 255 },
			-1
		);

		cv::putText(
			res,
			text,
			cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			{ 255, 255, 255 },
			1
		);
#endif

		// get center_point
		if (class_names[obj.label] == "pointer_rect")
		{

			cv::Mat pointer_rect = image(obj.rect);
			// 获取指针的直线
			getLines = getPointerLines(pointer_rect, obj.rect, pointer_line);
			// getCenterLocation(pointer_rect);
			// cv::circle(res, {meter_points.center_location.x, meter_points.center_location.y}, 2, {128,  153, 255}, -1);
		}

		auto& kps = obj.kps;
		for (int k = 0; k < num_point + 2; k++)
		{
			if (k < num_point)
			{
				int kps_x = std::round(kps[k * 3]);
				int kps_y = std::round(kps[k * 3 + 1]);
				float kps_s = kps[k * 3 + 2];
				if (kps_s > 0.5f)
				{
#ifdef VISUALIZER
					cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
					cv::circle(res, { kps_x, kps_y }, 2, kps_color, -1);
					std::cout << "current point: " << k << ", " << "kps x: " << kps_x << ", " << "kps y: " << kps_y << std::endl;
#endif

					if ((class_names[obj.label] == "pointer_rect") && (k == 0))
					{
						center_point_ori.x = kps_x;
						center_point_ori.y = kps_y;
					}
					// if ((class_names[obj.label] == "pointer_rect") && (k == 1))
					// {
					// 	meter_points.pointer_locations.x = kps_x;
					// 	meter_points.pointer_locations.y = kps_y;
					// }
					if (class_names[obj.label] == "left_rect")
					{
						start_point.x = kps_x;
						start_point.y = kps_y;
					}
					if (class_names[obj.label] == "right_rect")
					{
						end_point.x = kps_x;
						end_point.y = kps_y;
					}
				}
			}
		}
	}

#ifdef VISUALIZER
	cv::imshow("visualzer_img", res);
	cv::waitKey(0);
#endif

	if (getLines)
	{
		circleCenter(end_point, start_point, pointer_line, center_point);

		if ((center_point.x == -1) && (center_point.y == -1))  // 如果指针直线垂直于x轴,或者指针平行于起始点与终止点连线
		{
			/*
			// 使用检测目标框来估算仪表中心
			float r = (image.rows + image.cols) / 4;   // 求表盘半径
			// 估算表盘中心
			center_point.x =  r;
			center_point.y =  r;
			*/
			center_point.x = center_point_ori.x;
			center_point.y = center_point_ori.y;
		}
			
		// calculate pointer_point
		pointer_point = getPointerPoint(center_point, pointer_line);
		// std::cout << "start_x: " <<  start_point.x  << ", " << "start_y" << start_point.y << std::endl;
		// std::cout << "end_x: " <<  end_point.x  << ", " << "start_y" << end_point.y << std::endl;
		// std::cout << "center_x: " <<  center_point.x  << ", " << "center_y" << center_point.y << std::endl;

#ifdef VISUALIZER
		cv::line(res, cv::Point(pointer_line[0], pointer_line[1]), cv::Point(pointer_line[2], pointer_line[3]), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
		cv::circle(res, pointer_point, 3, cv::Scalar(255, 0, 0), -1);
		cv::circle(res, center_point, 3, cv::Scalar(0, 0, 255), -1);
		cv::circle(res, start_point, 3, cv::Scalar(15, 242, 235), -1);
		cv::circle(res, end_point, 3, cv::Scalar(15, 242, 235), -1);

		cv::imshow("image", res);
		cv::waitKey(0);
#endif

		float angleRatio = getAngleRatio(start_point, end_point, pointer_point, center_point);
		std::cout << "angleRatio: " << floatToString(angleRatio) + " Degrees" << std::endl;

		scale_value = abs(getScaleValue(angleRatio));
		std::cout << "scale_value: " << floatToString(scale_value) + " Mpa" << std::endl;

		return true;
	}
	else
	{
		return false;
	}
}


cv::Mat Yolov8Pose::result_visualizer(const cv::Mat& bgr, const std::vector<ObjectDetect>& objects_remains, const std::vector<float> scale_values)
{
	cv::Mat output_image = bgr.clone();
	for (int i_results = 0; i_results < objects_remains.size(); i_results++)
	{
		cv::Rect bounding_box = objects_remains[i_results].rect;

		float result;
		if (scale_values[i_results] <= 0.50f)
		{
			result = scale_values[i_results] + 0.012f;  // 0.01f
		}
		else
		{
			result = scale_values[i_results] + 0.005f;  // 0.08f
		}
		
		cv::Scalar color = cv::Scalar(237, 189, 101);
		cv::rectangle(output_image, bounding_box, color);  // 目标框

		std::string class_name = "Meter";
		cv::putText(output_image,
			class_name + " : " + floatToString(result) + " Mpa",
			cv::Point2d(bounding_box.x, bounding_box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
	}

	return output_image;
}


std::string Yolov8Pose::floatToString(const float& val)
{
	char* chCode;
	chCode = new char[20];
	sprintf(chCode, "%.3lf", val);
	std::string str(chCode);
	delete[]chCode;
	return str;
}


bool Yolov8Pose::getPointerLines(const cv::Mat& img, const cv::Rect_<float>& rect, cv::Vec4i& l_line)
{
	cv::Mat kernel_ = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	cv::Mat hist_img;
	cv::equalizeHist(gray, hist_img);

	// cv::imshow("hist_img", hist_img);
	// cv::waitKey(0);

	cv::Mat img_not;
	cv::bitwise_not(hist_img, img_not);

	// cv::imshow("img_not", img_not);
	// cv::waitKey(0);

	cv::Mat img_blur;
	cv::medianBlur(img_not, img_blur, 3);

	// cv::imshow("img_blur", img_blur);
	// cv::waitKey(0);

	cv::Mat img_erode;
	cv::erode(img_blur, img_erode, kernel_);

	// cv::imshow("img_erode", img_erode);
	// cv::waitKey(0);

	cv::Mat img_open;
	cv::morphologyEx(img_erode, img_open, cv::MORPH_CLOSE, kernel_);

	// cv::imshow("img_open", img_open);
	// cv::waitKey(0);

	cv::Mat img_thres;
	cv::threshold(img_open, img_thres, 210, 255, cv::THRESH_BINARY);  // 210, 255

	// cv::imshow("img_thres", img_thres);
	// cv::waitKey(0);

	cv::Mat thining_img;
	bool isThin = Thining_Rosenfeld(img_thres, thining_img);

	// cv::imshow("thining_img", thining_img);
	// cv::waitKey(0);

	if (isThin)
	{
		std::vector<cv::Vec4i> lines;
		int threshold = 50;
		int miniLineLenth = 100;  // 100 
		int maxLineGrap = 150;

		cv::HoughLinesP(thining_img, lines, 1, CV_PI / 180, threshold, miniLineLenth, maxLineGrap);

		/*
		if (lines.size() == 0)
		{
			//MessageBox(NULL, "未检测到直线,正在尝试新的检测方法", "提示", MB_OK);
			cv::HoughLinesP(thining_img, lines, 1, CV_PI / 180, 45, 30, maxLineGrap);
			for (int i = 0; i < lines.size(); i++)
			{
				cv::Vec4i l = lines[i];
				cv::line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
			}
		}
		else
		{
			for (int i = 0; i < lines.size(); i++)
			{
				cv::Vec4i l = lines[i];
				cv::line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			}
		}
		*/
		if (lines.size() >= 1)
		{
			l_line[0] = lines[0][0] + rect.x;
			l_line[1] = lines[0][1] + rect.y;
			l_line[2] = lines[0][2] + rect.x;
			l_line[3] = lines[0][3] + rect.y;

			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}


bool Yolov8Pose::Thining_Rosenfeld(cv::Mat& src, cv::Mat& dst)
{
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return false;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j, n;
	int width, height;
	//之所以减1，是方便处理8邻域，防止越界
	width = src.cols - 1;
	height = src.rows - 1;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	cv::Mat tmpimg;
	int dir[4] = { -step, step, 1, -1 };

	while (1)
	{
		//分四个子迭代过程，分别对应北，南，东，西四个边界点的情况
		ifEnd = false;
		for (n = 0; n < 4; n++)
		{
			dst.copyTo(tmpimg);
			img = tmpimg.data;
			for (i = 1; i < height; i++)
			{
				img += step;
				for (j = 1; j < width; j++)
				{
					uchar* p = img + j;
					//如果p点是背景点或者且为方向边界点，依次为北南东西，继续循环
					if (p[0] == 0 || p[dir[n]] > 0) continue;
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					//8 simple判定
					int is8simple = 1;
					if (p2 == 0 && p6 == 0)
					{
						if ((p9 == 1 || p8 == 1 || p7 == 1) && (p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p8 == 0)
					{
						if ((p9 == 1 || p2 == 1 || p3 == 1) && (p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p2 == 0)
					{
						if (p9 == 1 && (p3 == 1 || p4 == 1 || p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p2 == 0)
					{
						if (p3 == 1 && (p5 == 1 || p6 == 1 || p7 == 1 || p8 == 1 || p9 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p6 == 0)
					{
						if (p7 == 1 && (p3 == 9 || p2 == 1 || p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p6 == 0)
					{
						if (p5 == 1 && (p7 == 1 || p8 == 1 || p9 == 1 || p2 == 1 || p3 == 1))
							is8simple = 0;
					}
					int adjsum;
					adjsum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					//判断是否是邻接点或孤立点,0,1分别对于那个孤立点和端点
					if (adjsum != 1 && adjsum != 0 && is8simple == 1)
					{
						dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
						ifEnd = true;
					}
				}
			}
		}

		//printf("\n");
		//PrintMat(dst);
		//PrintMat(dst);
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}

	return true;  // 返回处理成功
}


void Yolov8Pose::circleCenter(cv::Point2f point2, cv::Point2f point1, cv::Vec4i line3, cv::Point2f& center_point)
{
	float k3 = float(line3[3] - line3[1]) / float(line3[2] - line3[0]);; // 指针直线的斜率
	std::cout << line3[3] << ", " << line3[1] << ", " << line3[2] << ", " << line3[0] << ", " << k3 << std::endl;

	const float epsilon = 1e-6; // 浮点数比较的容忍度

	if ((std::fabs(line3[2] - line3[0]) < epsilon) || (k3 < -10.0f)) // 检查是否为垂直线或者接近垂线
	{
		std::cout << "指针直线垂直，斜率不存在" << std::endl;
		// 返回特殊值表示无法计算
		center_point.x = -1;
		center_point.y = -1;
	}
	else  // 指针直线不垂直于x轴，即斜率存在
	{
		if ((point2.y - point1.y) == 0)
		{
			std::cout << "标定的起始点和终点为一条直线上的点" << std::endl;
			int line_k0_centerPoint_x = (point1.x + point2.x) / 2;		//当标定点与y轴平行时的直线
			int y = float((k3 * (line_k0_centerPoint_x - line3[0])) + line3[1]);
			center_point.x = line_k0_centerPoint_x;
			center_point.y = y;
		}
		else
		{
			//求起始点到终点直线的斜率
			float k1 = float(point2.y - point1.y) / float(point2.x - point1.x);
			//垂线line2的斜率为
			float k2 = float(-1) / float(k1);
			//垂线line2与line1相交点为(x,y)
			cv::Point2f line2(float((point2.x + point1.x) / 2), float((point2.y + point1.y) / 2));
			std::cout << "起始点与终点的中点坐标为: (" << line2.x << ", " << line2.y << ")" << std::endl;

			// 检查指针直线和中垂线是否平行
			if (std::fabs(k2 - k3) < epsilon)
			{
				std::cout << "指针直线与中垂线平行，无法确定唯一的圆心" << std::endl;
				center_point.x = -1; // 返回特殊值表示无法计算
				center_point.y = -1;
			}
			else
			{
				//指针与中垂线的交点为
				float x = float((k2 * line2.x) - (k3 * line3[0]) - line2.y + line3[1]) / float(k2 - k3);
				float y = float(k3 * (x - line3[0])) + float(line3[1]);
				x = fabs(x);
				y = fabs(y);
				std::cout << "圆心位置为x: " << x << "  y: " << y << std::endl;
				center_point.x = x;
				center_point.y = y;
			}
		}
	}

	return;
}


bool Yolov8Pose::isValidROI(const std::vector<ObjectDetect>& objects)
{
	for (auto& object : objects)
	{
		float width = object.rect.width;
		float height = object.rect.height;

		// 确保宽度和高度大于0
		if (width <= 0 || height <= 0) {
			std::cerr << "Invalid width or height in ROI." << std::endl;
			return false;
		}

		// calculate width-height ratio
		float aspectRatio = width / height;

		if (aspectRatio < aspectRatioThreshold || aspectRatio > 1.0f / aspectRatioThreshold)
		{
			std::cerr << "Aspect ratio of ROI is out of bounds: " << aspectRatio << std::endl;
			return false;
		}
	}

	return true;
}


bool Yolov8Pose::get_results(const std::vector<cv::Mat>& meters_image, std::vector<float>& meterScales)
{
	int meter_num = meters_image.size();

	for (int i_num = 0; i_num < meter_num; i_num++)
	{
		cv::Mat input_image = meters_image[i_num].clone();

		// 获取处理结果, objects存放yolov8 pose 目标结果
		std::vector<Object> objects;
		detect_yolov8(input_image, objects);

		if (objects.size() == 0)
		{
			std::cerr << "Doesn't detect any objects for image " << i_num << std::endl;
			return false;  // 立即终止并返回 false
		}

		try
		{
			// 获取指针仪表的点
			float scale_value = 0.0f;
			bool isGetResult = process_objects(input_image, objects, KPS_COLORS, scale_value);

			// 获取指针值
			if (isGetResult)
			{
				meterScales.push_back(scale_value);
			}
			else
			{
				return false;
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << "Error processing image " << i_num << ": " << e.what() << std::endl;
			return false;  // 立即终止并返回 false
		}
	}

	return true;  // 如果所有图像都成功处理，返回 true
}


cv::Point2f Yolov8Pose::getPointerPoint(cv::Point2f center_point, cv::Vec4i pointer_line)
{
	cv::Point2f pointer_point1 = cv::Point2f(pointer_line[0], pointer_line[1]);
	cv::Point2f pointer_point2 = cv::Point2f(pointer_line[2], pointer_line[3]);

	// 计算两个点到center_point的欧几里得距离
	float distance1 = cv::norm(pointer_point1 - center_point);
	float distance2 = cv::norm(pointer_point2 - center_point);

	// 返回距离更远的点
	if (distance1 > distance2)
	{
		return pointer_point1;
	}
	else
	{
		return pointer_point2;
	}
}