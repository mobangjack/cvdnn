/**
 * Copyright (c) 2017, Jack Mo (mobangjack@foxmail.com).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

void help()
{
	std::cout << "Usage: cvdnn <image> <proto> <model> <label>" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		help();
		return -1;
	}
   std::string image_path(argv[1]);
   std::string proto_path(argv[2]);
   std::string model_path(argv[3]);
   std::string label_path(argv[4]);
	
	std::ifstream label_stream(argv[4]);
	if (!label_stream)
	{
		std::cout << "ERROR: cannot open label file: " << label_path << std::endl;
		return -1;
	}

	std::vector<std::string> labels;
	
	while (!label_stream.eof())
	{
		std::string line;
		std::getline(label_stream, line);
		size_t label_beg_idx = line.find(" ") + 1;
		size_t label_end_idx = line.find(",", label_beg_idx + 1);
		if (label_end_idx == std::string::npos) label_end_idx = line.length();
		size_t label_len = label_end_idx - label_beg_idx;
		std::string label = line.substr(label_beg_idx, label_len);
		labels.push_back(label);
	}
	
	label_stream.close();

	cv::Mat image = cv::imread(image_path);
	
	// create blob from image
	cv::Mat blob = cv::dnn::blobFromImage(image, 1, cv::Size(224, 224), cv::Scalar(104, 117, 123));

	// read net from caffe
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(proto_path, model_path);
	
	// set net input
	net.setInput(blob);

	// time token for benchmark
	double t = cv::getTickCount();

	// forward
	cv::Mat probs = net.forward();
	
	// calculate time cost
	t = (cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();

	// show forward time cost
	std::cout << "[INFO] Forward time cost: " << (int)t << "ms" << std::endl;
	
	// debug
	//std::cout << "[INFO] probs.rows: " << probs.rows << ", probs.cols: " << probs.cols << std::endl;

	// sort result
	cv::Mat idxs; // = probs.reshape(1, 1);
	cv::sortIdx(probs, idxs, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

	// debug
	//std::cout << "[INFO] idxs.rows: " << idxs.rows << ", idxs.cols: " << idxs.cols << std::endl;

	// show the top 3 scored classes
	for (int i = 0; i < 3; i++)
	{
		int idx = idxs.at<int>(0, i);
		std::string label = labels[idx];
		float prob = probs.at<float>(0, idx);

		std::stringstream ss;
		ss << label << ": " << prob;
		std::string text = ss.str();

		std::cout << "[INFO] " << (i + 1) << ". " << text << std::endl;
		
		if (i == 0)
			cv::putText(image, text, cv::Size(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("cvdnn", image);
	cv::waitKey(0);
	
	cv::destroyAllWindows();

	return 0;
}
