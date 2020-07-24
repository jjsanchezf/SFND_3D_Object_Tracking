# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Implementation

### FP. 1

Keypoint cluster filter is implemented for given bounding box. The function is given below

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



### FP.2
Computation of TTC lidar for successive frames. First found mean x values for both previous and current lidar frames. Then put a threshold of 0.03 along mean value for filter. Mean value is again calculated for the equation TTC = d0 * (1.0 / frameRate) / d1-d0;

    void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
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
            cout<<TTC<<";";
        logger(std::to_string(TTC)+";");

    }

### FP.3

Computation of TTC camera is implemented as shown below. First iteration over matches to find current and previous keypoints. Under one loop, another loop with same itaration is done to find out distance ratio. Finally, median is calculated to apply in the equation TTC = -(1 / frameRate) / (1 - medDistRatio);

    void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
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
            cout<<TTC<<";";
        logger(std::to_string(TTC)+";");
    }
### FP.4

Matching of bounding box accroding to the best match using previous and current frames. Contains method is used in this function.


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

            //to find highest count in current frame for each box in previous frame
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
### FP.5

The following table shows the TTC lidar calcuation. It can be noticed that in image 3 & 4 and image 17 & 18, the Lidar sensor estimation does not seem plausible, this could be due to the influence of the filtering outliers, or reflective surfaces on thepreceding vehicle.


| images        | TTC Lidar |
|---------------|-----------|
| image 1 & 2   | 12.6641   |
| image 2 & 3   | 12.6184   |
| image 3 & 4   | 16.9723   |
| image 4 & 5   | 14.4201   |
| image 5 & 6   | 13.0322   |
| image 6 & 7   | 12.9238   |
| image 7 & 8   | 13.7459   |
| image 8 & 9   | 13.7415   |
| image 9 & 10  | 12.5512   |
| image 10 & 11 | 11.8621   |
| image 11 & 12 | 11.694    |
| image 12 & 13 | 10.3478   |
| image 13 & 14 | 9.20437   |
| image 14 & 15 | 9.71943   |
| image 15 & 16 | 8.3212    |
| image 16 & 17 | 8.89867   |
| image 17 & 18 | 11.0301   |
| image 18 & 19 | 8.3201    |

consecutive frames 2,3 and 4

<img src="Output\FAST\BRIEF\image 2_3.jpg" width="700" height="400" />
<img src="Output\FAST\BRIEF\image 3_4.jpg" width="700" height="400" />
<img src="Output\FAST\BRIEF\image 4_5.jpg" width="700" height="400" />

consecutive frames 16,17 and 18

<img src="Output\FAST\BRIEF\image 16_17.jpg" width="700" height="400" />
<img src="Output\FAST\BRIEF\image 17_18.jpg" width="700" height="400" />
<img src="Output\FAST\BRIEF\image 18_19.jpg" width="700" height="400" />


### FP.6

All combinations of detectors and descriptors were implemented in a loop. The output was collected In this [excel sheet](Output/performanceEvaluation.xlsx), from where charts can be plotted (frames against TTC lidar and TTC camera). every graph can be ploted by selecting  adetector-descriptor combination from the table. Only one combination at a time is observable.

The best 3 combinations according taking into account the RMSE and Standard Deviation are the following:

1. AKAZE / FREAK
2. AKAZE / BRISK
3. FAST / SIFT

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
