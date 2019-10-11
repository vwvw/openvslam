#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif


#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif
void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const unsigned int cam_num, const std::string& mask_img_path,
                   const float scale, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    auto video = cv::VideoCapture(cam_num);
    if (!video.isOpened()) {
        spdlog::critical("cannot open a camera {}", cam_num);
        SLAM.shutdown();
        return;
    }

    cv::Mat frame;
    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    bool is_not_end = true;
    // run the SLAM in another thread
    std::thread thread([&]() {
        while (is_not_end) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            is_not_end = video.read(frame);
            if (frame.empty()) {
                continue;
            }
            if (scale != 1.0) {
                cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_monocular_frame(frame, timestamp, mask);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}
void stereo_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const unsigned int cam_num, const std::string& mask_img_path,
                   const float scale, const std::string& map_db_path) {
    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    auto video = cv::VideoCapture(cam_num);
    auto video1 = cv::VideoCapture(cam_num+1);
    if (!video.isOpened()) {
        spdlog::critical("cannot open a camera {}", cam_num);
        SLAM.shutdown();
        return;
    }
    if (!video1.isOpened()) {
        spdlog::critical("cannot open a camera {}", cam_num+1);
        SLAM.shutdown();
        return;
    }
    video.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    video.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    video1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    video1.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

    cv::Mat frame;
    cv::Mat frameb;
    cv::Mat frame2;
    cv::Mat frame2b;
    double timestamp = 0.0;
    std::vector<double> track_times;

    // variables for stereo image rectification

    cv::Mat Proj1 = (cv::Mat_<double>(3,4) << 1354.61399523, 0.0, 799.28490448, 0.0, 0.0, 1354.61399523, 532.021461487, 0.0, 0.0, 0.0, 1.0, 0.0);
    cv::Mat Proj2 = (cv::Mat_<double>(3,4) << 1354.61399523, 0.0, 799.28490448, 3055.96096986, 0.0, 1354.61399523, 532.021461487, 0.0, 0.0, 0.0, 1.0, 0.0);
    cv::Mat Rect_1 = (cv::Mat_<double>(3,3) << 0.996937332321, 0.0121357087942, 0.0772572326557, -0.0120439939178, 0.9999261022, -0.00165298213167, -0.077271583626, 0.000717433956507, 0.997009823248);
    cv::Mat Rect_2 = (cv::Mat_<double>(3,3) << 0.998127491797, 0.00954551702489, 0.0604184841213, -0.00961720723976, 0.999953352282, 0.000895871400919, -0.0604071141811, -0.00147525095728, 0.998172732642);

    cv::Mat Matrix1 = (cv::Mat_<double>(3,3) << 1476.84824872, 0.0, 967.139052368, 0.0, 1416.0587045, 521.833122122, 0.0, 0.0, 1.0);
    cv::Mat Matrix2 = (cv::Mat_<double>(3,3) << 1476.84824872, 0.0, 906.94313055, 0.0, 1416.0587045, 536.993803971, 0.0, 0.0, 1.0);
    cv::Mat dist1 = (cv::Mat_<double>(14,1) << -0.345985969537, -0.291714403638, 0.0, 0.0, 0.0, 0.0, 0.0, -0.564446527387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cv::Mat dist2 = (cv::Mat_<double>(14,1) << -0.302062963411, -0.39871929128, 0.0, 0.0, 0.0, 0.0, 0.0, -0.633842904892, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    

    const auto cols = cfg->camera_->cols_;
    const auto rows = cfg->camera_->rows_;

    cv::Mat M1_l, M2_l, M1_r, M2_r;
    cv::Size imageSize = cv::Size(1920, 1080);
    cv::initUndistortRectifyMap(Matrix1, dist1, Rect_1, Proj1, imageSize, CV_32F, M1_l, M2_l);
    cv::initUndistortRectifyMap(Matrix2, dist2, Rect_2, Proj2, imageSize, CV_32F, M1_r, M2_r);
    unsigned int num_frame = 0;
    std::cout << M1_l.at<float>(0,0) << std::endl;

    bool is_not_end = true;
    // run the SLAM in another thread
    std::thread thread([&]() {
        while (is_not_end) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            is_not_end = video.read(frame);
            is_not_end = video1.read(frame2) && is_not_end;
            if (frame.empty() || frame2.empty()) {
                continue;
            }
            //*
            if (scale != 1.0) {
                cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
                cv::resize(frame2, frame2, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }//*/
            cv::remap(frame, frameb, M1_l, M2_l, cv::INTER_LINEAR);
            cv::remap(frame2, frame2b, M1_r, M2_r, cv::INTER_LINEAR);


            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_stereo_frame(frame2b, frameb, timestamp);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto cam_num = op.add<popl::Value<unsigned int>>("n", "number", "camera number");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto scale = op.add<popl::Value<float>>("s", "scale", "scaling ratio of images", 1.0);
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !cam_num->is_set()
        || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
    }
    else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
        stereo_tracking(cfg, vocab_file_path->value(), cam_num->value(), mask_img_path->value(),
                      scale->value(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
