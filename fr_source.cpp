#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>
#include <ctime>
#include <cstdio>
#include <stdio.h>

//#include <stdio.h>
//static int cnt=0 ;

#include <Windows.h>
#include <MMsystem.h>

#ifdef _WIN32
#include "dirent.h"
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#include <dirent.h>
#else
#error "Not Mac. Find an alternative to dirent"
#endif
#elif __linux__
#include <dirent.h>
#elif __unix__ // all unices not caught above
#include <dirent.h>
#else
#error "Unknown compiler"
#endif


using namespace std;
using namespace cv;
using namespace dlib;
//using namespace dlib;

#define test
#define SKIP_FRAMES 2
#define THRESHOLD 0.4
#define model2
#define playsound

const char key[6] = "apv03";


// ----------------------------------------------------------------------------------------

//
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

// Reads files, folders and symbolic links in a directory
void listdir(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames) {
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(dirName.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			// ignore . and ..
			if ((strcmp(ent->d_name, ".") == 0) || (strcmp(ent->d_name, "..") == 0)) {
				continue;
			}
			string temp_name = ent->d_name;
			switch (ent->d_type) {
			case DT_REG:
				fileNames.push_back(temp_name);
				break;
			case DT_DIR:
				folderNames.push_back(dirName + "/" + temp_name);
				break;
			case DT_LNK:
				symlinkNames.push_back(temp_name);
				break;
			default:
				break;
			}
			// cout << temp_name << endl;
		}
		// sort all the files
		std::sort(folderNames.begin(), folderNames.end());
		std::sort(fileNames.begin(), fileNames.end());
		std::sort(symlinkNames.begin(), symlinkNames.end());
		closedir(dir);
	}
}

// filter files having extension ext i.e. jpg
void filterFiles(string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, string ext, std::vector<int>& imageLabels, int index) {
	for (int i = 0; i < fileNames.size(); i++) {
		string fname = fileNames[i];
		if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos) {
			filteredFilePaths.push_back(dirPath + "/" + fname);
			imageLabels.push_back(index);
		}
	}
}

template<typename T>
void printVector(std::vector<T>& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << i << " " << vec[i] << "; ";
	}
	cout << endl;
}

// read names and labels mapping from file
static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels,
	std::map<int, string>& labelNameMap, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line;
	string name, labelStr;
	// read lines from file one by one
	while (getline(file, line)) {
		stringstream liness(line);
		// read first word which is person name
		getline(liness, name, separator);
		// read second word which is integer label
		getline(liness, labelStr);
		if (!name.empty() && !labelStr.empty()) {
			names.push_back(name);
			// convert label from string format to integer
			int label = atoi(labelStr.c_str());
			labels.push_back(label);
			// add (integer label, person name) pair to map
			labelNameMap[label] = name;
		}
	}
}

// read descriptors saved on disk
static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<matrix<float, 0, 1>>& faceDescriptors, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	// each line has:
	// 1st element = face label
	// rest 128 elements = descriptor elements
	string line;
	string faceLabel;
	// valueStr = one element of descriptor in string format
	// value = one element of descriptor in float
	string valueStr;
	float value;
	std::vector<float> faceDescriptorVec;
	// read lines from file one by one
	while (getline(file, line)) {
		stringstream liness(line);
		// read face label
		// read first word on a line till separator
		getline(liness, faceLabel, separator);
		if (!faceLabel.empty()) {
			faceLabels.push_back(std::atoi(faceLabel.c_str()));
		}

		faceDescriptorVec.clear();
		// read rest of the words one by one using separator
		while (getline(liness, valueStr, separator)) {
			if (!valueStr.empty()) {
				// convert descriptor element from string to float
				faceDescriptorVec.push_back(atof(valueStr.c_str()));
			}
		}

		// convert face descriptor from vector of float to Dlib's matrix format
		dlib::matrix<float, 0, 1> faceDescriptor = dlib::mat(faceDescriptorVec);
		faceDescriptors.push_back(faceDescriptor);
	}
}

int key_validation(char *ptr1, const char *ptr2) {

	if (*ptr1 == *ptr2)
		return 1;
	else
		return 0;

}



// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearestNeighbor(dlib::matrix<float, 0, 1>& faceDescriptorQuery,
	std::vector<dlib::matrix<float, 0, 1>>& faceDescriptors,
	std::vector<int>& faceLabels, int& label, float& minDistance) {
	int minDistIndex = 0;
	minDistance = 2.0;
	label = -1;
	// Calculate Euclidean distances between face descriptor calculated on face dectected
	// in current frame with all the face descriptors we calculated while enrolling faces
	// Calculate minimum distance and index of this face
	for (int i = 0; i < faceDescriptors.size(); i++) {
		double distance = length(faceDescriptors[i] - faceDescriptorQuery);
		if (distance < minDistance) {
			minDistance = distance;
			minDistIndex = i;
		}
	}
	

	// This threshold will vary depending upon number of images enrolled
	// and various variations (illuminaton, camera quality) between
	// enrolled images and query image
	// We are using a threshold of 0.5
	// if minimum distance is greater than a threshold
	// assign integer label -1 i.e. unknown face
	if (minDistance > THRESHOLD) {
		label = -1;
	}
	else {
		label = faceLabels[minDistIndex];
	}
}



int frEnroll();

int frRecognize();

/*
====================================================================================================================================
Main


====================================================================================================================================
*/

int main() {

	int ret;
	int n;

	printf("\n1.Enroll\n2.detect\n3.Exit");
	while (1)
	{
		printf("\nEnter your choice:");
		scanf("%d", &n);
		switch (n)
		{
		case 1:	ret = frEnroll();
				break;
		case 2:	ret=frRecognize();
				break;
		case 3:
			exit(0);
			break;
		}
	}

	//ret=frDetect();

	//frRecognize();
	return ret;
}


int frRecognize() {

	//system("pause");
	// current date/time based on current system

	time_t now = time(0);
	int cnt = 0;
	
	ofstream outdata;
	outdata.open("log.csv",ios::out); // opens the file

	// Initialize face detector, facial landmarks detector and face recognizer
	String predictorPath, faceRecognitionModelPath;
	cout << "<<<<<<<<< Model initialization is starting >>>>>>>>" << endl;
#ifndef model2
	predictorPath = "../models/model_v1_ld.dat";
#else
	predictorPath = "../models/model_v2_ld.dat";
#endif 
	faceRecognitionModelPath = "../models/model_v1_fr.dat";

	cout << "||             Face_detector                     ||" << endl;
	cout << "||                                               ||" << endl;

	frontal_face_detector faceDetector = get_frontal_face_detector();

	cout << "||             LandmarkDetector                  ||" << endl;
	cout << "||                                               ||" << endl;

	shape_predictor landmarkDetector;
	deserialize(predictorPath) >> landmarkDetector;

	cout << "||            faceRecognitionModel               ||" << endl;
	cout << "||                                               ||" << endl;

	anet_type net;
	deserialize(faceRecognitionModelPath) >> net;

	cout << "<<<<<<<< Model initialization done >>>>>>>>>>>>>>>>>>" << endl;
	cout << "||        Authetication is required              ||" << endl;

	char ptr[10];
	char *ptr2 = ptr;
	const char *ptr1 = key;
	//fgets(ptr);
	cout << "||         Authetication key is=                 ||" << endl;

	//---- system call
	HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
	DWORD mode = 0;
	GetConsoleMode(hStdin, &mode);
	SetConsoleMode(hStdin, mode & (~ENABLE_ECHO_INPUT));
	//cout << "";
	cin >> ptr;
	//---- system call
	SetConsoleMode(hStdin, mode);
	cout << endl;


	for (int q = 0; q < 5; q++) {
		int rt = key_validation(ptr2, ptr1);
		if (rt == 0) {
			cout << "||         Authentication is denied                  ||" << endl;
			cout << "||================================================||" << endl;
			return 0;
		}
			
				
				ptr2++;
				ptr1++;

	}

	cout << "||         Authentication is done                  ||" << endl;
	cout << "||================================================||"<<endl;

	// read names, labels and labels-name-mapping from file
	std::map<int, string> labelNameMap;
	std::vector<string> names;
	std::vector<int> labels;
	const string labelNameFile = "label_name.txt";
	readLabelNameMap(labelNameFile, names, labels, labelNameMap);

	// read descriptors of enrolled faces from file
	const string faceDescriptorFile = "descriptors.csv";
	std::vector<int> faceLabels;
	std::vector<matrix<float, 0, 1>> faceDescriptors;
	readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);

	// Create a VideoCapture object
	VideoCapture cap;
	//cap.open("../data/videos/face1.mp4");
	cap.open(0);// apv // change this to 1 if externla cam is used

	// Check if OpenCV is able to read feed from camera
	if (!cap.isOpened()) {
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	int count = 0;
	//double t = cv::getTickCount();

	while (1) {
		//t = cv::getTickCount();

		// Capture frame
		Mat im;
		cap >> im;

		// If the frame is empty, break immediately
		if (im.empty()) {
			break;
		}

		// We will be processing frames after an interval
		// of SKIP_FRAMES to increase processing speed
		if ((count % SKIP_FRAMES) == 0) {

			// convert image from BGR to RGB
			// because Dlib used RGB format
			Mat imRGB = im.clone();
			cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
			// convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
			// Dlib's dnn module doesn't accept Dlib's cv_image template
			dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

			// detect faces in image
			std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
			// Now process each face we found
			int payvid = 0;
			for (int i = 0; i < faceRects.size(); i++) {

				// Find facial landmarks for each detected face
				full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i]);

				// current date/time based on current system
				time_t now = time(0);

				// convert now to string form
				char* dt = ctime(&now);
				dt[24] = '\0';
				//printf("%d", strlen(dt));

				// object to hold preProcessed face rectangle cropped from image
				matrix<rgb_pixel> face_chip;

				// original face rectangle is warped to 150x150 patch.
				// Same pre-processing was also performed during training.
				extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

				// Compute face descriptor using neural network defined in Dlib.
				// It is a 128D vector that describes the face in img identified by shape.
				matrix<float, 0, 1> faceDescriptorQuery = net(face_chip);

				// Find closest face enrolled to face found in frame
				int label;
				float minDistance;
				nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
				// Name of recognized person from map
				string name = labelNameMap[label];

				//cout << "Time taken = " << ((double)cv::getTickCount() - t) / cv::getTickFrequency() << endl;

				// Draw a rectangle for detected face
				Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
				Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
				cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);

				// Draw circle for face recognition
				Point2d center = Point((faceRects[i].left() + faceRects[i].right()) / 2.0,
					(faceRects[i].top() + faceRects[i].bottom()) / 2.0);
				//int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top()) / 2.0);
				//cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);

				// Write text on image specifying identified person and minimum distance
				stringstream stream;
				stream << name << " ";
				//stream << fixed << setprecision(4) << minDistance;
				string text = stream.str();
				//string text3 = &dt;
				//cout << strlen(dt) << endl;
				//cout<<str
				cv::putText(im, text, p1,FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
				Point2d p3= Point2d(10, 30);
				cv::putText(im, dt, p3, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
				std::string time_(dt);
				
				string st = "../data/output/" + to_string(cnt)+ ".jpg";
				//string st=st + time_ +"" ;
				//cout << time_ << endl;
				if ((label == (-1)) && (count > 5)) {
					payvid = -1;
					// Write line in file 
					outdata << "UNKNOWN" << dt << endl;
					imwrite(st, im);
					//cnt++;
				}
				else
				{
					outdata << name << dt << endl;
					imwrite(st, im);
				}

				
			}
			// Show result
			cv::imshow("webcam", im);
			int k = cv::waitKey(1);
			// Quit when Esc is pressed
			if (k == 27) {
				break;
			}
#ifdef playsound
			if ((payvid == (-1))&&(count>5)) {
				PlaySound(TEXT("videoplayback final.wav"), NULL, SND_SYNC);//apv
				//PlaySound(NULL, 0, 0);
				//count++;
			}
#endif 

		}
		// Counter used for skipping frames
		count += 1;
	}
	cv::destroyAllWindows();
	outdata.close();
	return 0;
}

int frEnroll() {

	cout << "<<<<<<<<< Model initialization is starting >>>>>>>>"<< endl;
	// Initialize face detector, facial landmarks detector and face recognizer
	String predictorPath, faceRecognitionModelPath;
#ifndef model2
	predictorPath = "../models/model_v1_ld.dat";
#else
	predictorPath = "../models/model_v2_ld.dat";
#endif
	faceRecognitionModelPath = "../models/model_v1_fr.dat";
	frontal_face_detector faceDetector = get_frontal_face_detector();
	cout << "||               face_detector                   ||" << endl;
	cout << "||                                               ||" << endl;
	shape_predictor landmarkDetector;
	deserialize(predictorPath) >> landmarkDetector;
	cout << "||               landmarkDetector                ||" << endl;
	cout << "||                                               ||" << endl;
	anet_type net;
	deserialize(faceRecognitionModelPath) >> net;	
	cout << "||               faceRecognitionModel            ||" << endl;
	cout << "<<<<<<<< Model initialization done >>>>>>>>>>>>>>>>>>" << endl;

	cout << "||         authetication is required             ||" << endl;
	char ptr[10];
	char *ptr2 = ptr;
	const char *ptr1 = key;
	//fgets(ptr);
	cout << "||         authetication key is=                 ||" << endl;

	//---- ---------------------------------------------------system call
	HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
	DWORD mode = 0;
	GetConsoleMode(hStdin, &mode);
	SetConsoleMode(hStdin, mode & (~ENABLE_ECHO_INPUT));
	//cout << "";
	cin >> ptr;
	//--------------------------------------------------------system call
	SetConsoleMode(hStdin, mode);
	cout << endl;


	for (int q = 0; q < 5; q++) {
		int rt = key_validation(ptr2, ptr1);
		if (rt == 0) {
			cout << "||         Authetication is dined                  ||" << endl;
			cout << "||================================================||" << endl;
			return 0;
		}
			

		ptr2++;
		ptr1++;

	}

	cout << "||         authetication is done                 ||" << endl;

	cout << "##=========Enrolling facess of Known Id's========##" << endl;
	// enrollment of student in our prmises 
	// data is organized assuming following structure
	// faces folder has subfolders.
	// each subfolder has images of a person
	string faceDatasetFolder = "../data/images/FaceRec";
	std::vector<string> subfolders, fileNames, symlinkNames;
	// fileNames and symlinkNames are useless here
	// as we are looking for sub-directories only
	listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

	// names: vector containing names of subfolders i.e. persons
	// labels: integer labels assigned to persons
	// labelNameMap: dict containing (integer label, person name) pairs
	std::vector<string> names;
	std::vector<int> labels;
	std::map<int, string> labelNameMap;
	// add -1 integer label for un-enrolled persons
	names.push_back("unknown");
	labels.push_back(-1);

	// imagePaths: vector containing imagePaths
	// imageLabels: vector containing integer labels corresponding to imagePaths
	std::vector<string> imagePaths;
	std::vector<int> imageLabels;

	// variable to hold any subfolders within person subFolders
	std::vector<string> folderNames;
	// iterate over all subFolders within faces folder
	for (int i = 0; i < subfolders.size(); i++) {
		string personFolderName = subfolders[i];
		// remove / or \\ from end of subFolder
		std::size_t found = personFolderName.find_last_of("/\\");
		string name = personFolderName.substr(found + 1);
		// assign integer label to person subFolder
		int label = i;
		// add person name and label to vectors
		names.push_back(name);
		labels.push_back(label);
		// add (integer label, person name) pair to map
		labelNameMap[label] = name;

		// read imagePaths from each person subFolder
		// clear vectors
		folderNames.clear();
		fileNames.clear();
		symlinkNames.clear();
		// folderNames and symlinkNames are useless here
		// as we are only looking for files here
		// read all files present in subFolder
		listdir(subfolders[i], folderNames, fileNames, symlinkNames);
		// filter only jpg files
		filterFiles(subfolders[i], fileNames, imagePaths, "jpg", imageLabels, i);//APV // change the image format if required
	}

	// process training data
	// We will store face descriptors in vector faceDescriptors
	// and their corresponding labels in vector faceLabels
	std::vector<matrix<float, 0, 1>> faceDescriptors;
	// std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
	std::vector<int> faceLabels;

	// iterate over images
	for (int i = 0; i < imagePaths.size(); i++) {
		string imagePath = imagePaths[i];
		int imageLabel = imageLabels[i];

		cout << "processing: " << imagePath << endl;

		// read image using OpenCV
		Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

		// convert image from BGR to RGB
		// because Dlib used RGB format
		Mat imRGB;
		cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

		// convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
		// Dlib's dnn module doesn't accept Dlib's cv_image template
		dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

		// detect faces in image
		std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
		cout << faceRects.size() << " Face(s) Found" << endl;
		// Now process each face we found
		for (int j = 0; j < faceRects.size(); j++) {

			// Find facial landmarks for each detected face
			full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);

			// object to hold preProcessed face rectangle cropped from image
			matrix<rgb_pixel> face_chip;

			// original face rectangle is warped to 150x150 patch.
			// Same pre-processing was also performed during training.
			extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

			// Compute face descriptor using neural network defined in Dlib.
			// It is a 128D vector that describes the face in img identified by shape.
			matrix<float, 0, 1> faceDescriptor = net(face_chip);

			// add face descriptor and label for this face to
			// vectors faceDescriptors and faceLabels
			faceDescriptors.push_back(faceDescriptor);
			// add label for this face to vector containing labels corresponding to
			// vector containing face descriptors
			faceLabels.push_back(imageLabel);
		}
	}

	cout << "number of face descriptors " << faceDescriptors.size() << endl;
	cout << "number of face labels " << faceLabels.size() << endl;

	// write label name map to disk
	const string labelNameFile = "label_name.txt";
	ofstream of;
	of.open(labelNameFile);
	for (int m = 0; m < names.size(); m++) {
		of << names[m];
		of << ";";
		of << labels[m];
		of << "\n";
	}
	of.close();

	// write face labels and descriptor to disk
	// each row of file descriptors.csv has:
	// 1st element as face label and
	// rest 128 as descriptor values
	const string descriptorsPath = "descriptors.csv";
	ofstream ofs;
	ofs.open(descriptorsPath);
	// write descriptors
	for (int m = 0; m < faceDescriptors.size(); m++) {
		matrix<float, 0, 1> faceDescriptor = faceDescriptors[m];
		std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());
		// cout << "Label " << faceLabels[m] << endl;
		ofs << faceLabels[m];
		ofs << ";";
		for (int n = 0; n < faceDescriptorVec.size(); n++) {
			ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];
			// cout << n << " " << faceDescriptorVec[n] << endl;
			if (n == (faceDescriptorVec.size() - 1)) {
				ofs << "\n";  // add ; if not the last element of descriptor
			}
			else {
				ofs << ";";  // add newline character if last element of descriptor
			}
		}
	}
	ofs.close();

	cout << "==============Enrollment is over==============" << endl;

	return 0;
}






