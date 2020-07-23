
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

bool debugcommt = true;

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
	for (auto &match : kptMatches)
    {
        const auto &currKeyPoint = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(currKeyPoint))
        {
            boundingBox.kptMatches.push_back(match);
        }
    }
  	
  	double meanDist = 0;
  	double dist;
  
  	cv::KeyPoint kptsCurr_, kptsPrev_;
  
  	for (auto &matchId : boundingBox.kptMatches)
    {
        kptsCurr_ = kptsCurr.at(matchId.trainIdx);
        kptsPrev_ = kptsPrev.at(matchId.queryIdx);

        
        dist = cv::norm(kptsCurr_.pt - kptsPrev_.pt);
        meanDist +=  dist;

    }
  
  	meanDist /= boundingBox.kptMatches.size();
  
  	for (auto iter = boundingBox.kptMatches.begin(); iter < boundingBox.kptMatches.end();)
    {
        kptsCurr_ = kptsCurr.at(iter->trainIdx);
        kptsPrev_ = kptsPrev.at(iter->queryIdx);
        
        dist = cv::norm(kptsCurr_.pt - kptsPrev_.pt);

        if (dist >= meanDist*2)
        {
            boundingBox.kptMatches.erase(iter);
        }
        else
        {
            iter++;
        }
    }
  
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // calculation of distance ratio
    vector<double> distRatio; 
    for (auto iter = kptMatches.begin(); iter != kptMatches.end() - 1; ++iter)
    { 
        
        cv::KeyPoint kptsCurr_ = kptsCurr.at(iter->trainIdx); // current keypoints
        cv::KeyPoint kptsPrev_ = kptsPrev.at(iter->queryIdx); // previous keypoints

        for (auto iter_ = kptMatches.begin() + 1; iter_ != kptMatches.end(); ++iter_)
        { 
        
            double minDist = 90.0; // mimnimum distance

            
            cv::KeyPoint kptsCurr__ = kptsCurr.at(iter_->trainIdx); // current keypoint
            cv::KeyPoint kptsPrev__ = kptsPrev.at(iter_->queryIdx); // previous keypoint

        
            double distCurr = cv::norm(kptsCurr_.pt - kptsCurr__.pt); // current distance
            double distPrev = cv::norm(kptsPrev_.pt - kptsPrev__.pt); // previous distance

            // avoid division by zero
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { 
                double distRatio_ = distCurr / distPrev;
                distRatio.push_back(distRatio_);
            }
        } 
    }    

    // eluminates empty list
    if (distRatio.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();        
        return;
    }
	
  	
    std::sort(distRatio.begin(), distRatio.end());
    long medIndex = floor(distRatio.size() / 2.0);

    double medDistRatio = distRatio.size() % 2 == 0 ? (distRatio[medIndex - 1] + distRatio[medIndex]) / 2.0 : distRatio[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    if(debugcommt)
            cout<<TTC<<":";
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{	
  
  	double meanXPre = 0;
    double meanXCur = 0;
  	double diff;
	
  	for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        meanXPre +=  it->x;
    }
  
  	meanXPre /= lidarPointsPrev.size();
  
  	for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        meanXCur +=  it->x;
    }
  
  	meanXCur /= lidarPointsCurr.size();
  
  	 for(auto it = lidarPointsPrev.begin(); it<lidarPointsPrev.end(); ++it)
    {
        if(fabs(meanXPre - it->x) >= 0.03*meanXPre)
        {
            lidarPointsPrev.erase(it);
        }
    }
    for(auto it = lidarPointsCurr.begin(); it<lidarPointsCurr.end(); ++it)
    {
        if(fabs(meanXCur - it->x) >= 0.03*meanXCur)
        {
            lidarPointsCurr.erase(it);
        }
    }
  
  	meanXPre = 0;
    meanXCur = 0;
  	
  	for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        meanXPre +=  it->x;
    }
  
  	meanXPre /= lidarPointsPrev.size();
  
  	for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        meanXCur +=  it->x;
    }
  
  	meanXCur /= lidarPointsCurr.size();
  
  	diff = meanXPre - meanXCur;

    TTC = meanXCur * (1.0 / frameRate) / diff;
    
  	if(debugcommt)
            cout<<TTC<<":";
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    int bboxPrevSize = prevFrame.boundingBoxes.size();
    int bboxCurrSize = currFrame.boundingBoxes.size();
    int ptBBox[bboxPrevSize][bboxCurrSize] = {};

    //iterations over matches
    for(auto iter = matches.begin();iter!=matches.end()-1;++iter)
    {
        
        cv::KeyPoint kptPrev = prevFrame.keypoints[iter->queryIdx]; // previous keypoint
        cv::Point ptPrev = cv::Point(kptPrev.pt.x,kptPrev.pt.y); // previous point
        
        cv::KeyPoint kptCurr = currFrame.keypoints[iter->trainIdx]; // current keypoint
        cv::Point ptCurr = cv::Point(kptCurr.pt.x,kptCurr.pt.y); // current point

        std::vector<int> bboxIdsPrev , bboxIdsCurr; // store box ids

        //adding box ids
        for (int i=0;i< bboxPrevSize;++i)
        {
            if(prevFrame.boundingBoxes[i].roi.contains(ptPrev))
            {
                bboxIdsPrev.push_back(i);
            }
        }
        for (int j=0;j< bboxCurrSize;++j)
        {
            if(currFrame.boundingBoxes[j].roi.contains(ptCurr))
            {
                bboxIdsCurr.push_back(j);
            }
        }

        for(auto prev:bboxIdsPrev)
        {
            for(auto curr:bboxIdsCurr)
            {
                ptBBox[prev][curr]+=1;
            }
        }
            
                
    }

    //to find highest count in current frame for each box in prevframe
    for(int i = 0; i< bboxPrevSize; ++i)
    {
        int maxCnt = 0;
        int maxID = 0;

        for(int j=0; j< bboxPrevSize; ++j)
        {
            if(ptBBox[i][j] > maxCnt)
            {
                maxCnt = ptBBox[i][j];
                maxID = j;
            }
        }
        bbBestMatches[i] = maxID;
    }
}
