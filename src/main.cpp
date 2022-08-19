// CMakeProject1.cpp: 定义应用程序的入口点。
//

#include "../include/test.h"


using namespace cv;

Json::Value aligned_images_data;
Json::Value romancia_data;
void* results;

Json::Value add_trace(Json::Value jsonobj_param, Json::Value jsonobj_face)
{
    Json::Value temp;
    temp.clear();
    temp["instance_guid"] = jsonobj_param["instance_guid"];
    temp["format"] = jsonobj_param["format"];
    temp["height"] = jsonobj_param["height"];
    temp["width"] = jsonobj_param["width"];
    temp["face"]["x"] = jsonobj_face["x"];
    temp["face"]["y"] = jsonobj_face["y"];
    temp["face"]["width"] = jsonobj_face["width"];
    temp["face"]["height"] = jsonobj_face["height"];
    return temp;
}

int test_longinus(void* parser, std::string longinus_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    //capture.open("D:/opencv/e8dda5b6c23915758ffcba00753ae2a9.mp4");
    //capture.open("D:/opencv/QQ20220809100238.mp4");
    capture.open("D:/opencv/test/weixin.jpg");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img, temp;
    std::map<int, Json::Value >faces;
    while (1)
    {

        capture >> img;
        if (img.empty())
            break;
        //cv::cvtColor(temp, img, COLOR_BGR2YUV_I420);


        Json::Reader reader(Json::Features::strictMode());
        Json::Value jsonobj_param;
        Json::Value jsonobj_result;
        Json::Value jsonobj_trace;
        jsonobj_param.clear();
        jsonobj_param["instance_guid"] = Json::Value(longinus_guid);
        jsonobj_param["format"] = Json::Int(1);
        jsonobj_param["height"] = Json::Int(img.rows);
        jsonobj_param["width"] = Json::Int(img.cols);
        jsonobj_param["min_size"] = Json::Value(48);
        jsonobj_param["threshold"] = Json::Value(0.5);
        jsonobj_param["do_attributing"] = Json::Value(false);
        char* result_str = parser_parse(parser, "Longinus.detect", writer.write(jsonobj_param).c_str(),
            reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
        reader.parse(result_str, jsonobj_result);
        Json::Value jsonobj_face;
        int flag = 0;
        if (jsonobj_result["status"]["code"].asInt() == 0)
        {
            int list_size = jsonobj_result["facerectwithfaceinfo_list"].size();
            if (list_size)flag = 1;
            romancia_data.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_face = jsonobj_result["facerectwithfaceinfo_list"][i];

                romancia_data["facerectwithfaceinfo_list"][i]["confidence"] = jsonobj_face["confidence"];
                romancia_data["facerectwithfaceinfo_list"][i]["height"] = jsonobj_face["height"];
                romancia_data["facerectwithfaceinfo_list"][i]["landmark"] = jsonobj_face["landmark"];
                romancia_data["facerectwithfaceinfo_list"][i]["width"] = jsonobj_face["width"];
                romancia_data["facerectwithfaceinfo_list"][i]["x"] = jsonobj_face["x"];
                romancia_data["facerectwithfaceinfo_list"][i]["y"] = jsonobj_face["y"];

                int x, y;
                x = jsonobj_face["x"].asInt();
                y = jsonobj_face["y"].asInt();
                int width, height;
                width = jsonobj_face["width"].asInt();
                height = jsonobj_face["height"].asInt();
                faces[x * 10000 + y] = add_trace(jsonobj_param, jsonobj_face);
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255, 255, 255), 2);

            }
        }
        else
        {
            printf("Error info : % s\n", jsonobj_result["status"]["message"].asString().c_str());
            continue;
        }
        parser_free(result_str);
        result_str = nullptr;

        // 追踪人脸  
        std::map<int, Json::Value >::iterator it;
        std::map<int, Json::Value >temp_faces;
        for (it = faces.begin(); it != faces.end();)
        {
            result_str = parser_parse(parser, "Longinus.trace", writer.write(it->second).c_str(),
                reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
            reader.parse(result_str, jsonobj_trace);

            if (jsonobj_trace["status"]["code"].asInt() == 0 && jsonobj_trace["trace_success"].asBool() == true)
            {
                flag = 1;
                int x, y;
                x = jsonobj_trace["facerectwithfaceinfo"]["x"].asInt();
                y = jsonobj_trace["facerectwithfaceinfo"]["y"].asInt();
                int width, height;
                width = jsonobj_trace["facerectwithfaceinfo"]["width"].asInt();
                height = jsonobj_trace["facerectwithfaceinfo"]["height"].asInt();
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(0, 255, 0), 2);
                //更新追踪坐标   
                it->second["face"]["x"] = jsonobj_trace["facerectwithfaceinfo"]["x"];
                it->second["face"]["y"] = jsonobj_trace["facerectwithfaceinfo"]["y"];
                it->second["face"]["width"] = jsonobj_trace["facerectwithfaceinfo"]["width"];
                it->second["face"]["height"] = jsonobj_trace["facerectwithfaceinfo"]["height"];
                temp_faces[x * 10000 + y] = it->second;
                faces.erase(it++);
            }
            else if (jsonobj_trace["trace_success"].asBool() == false)
            {
                printf("Error info : %s\n", "trace_success = false ");
                faces.erase(it++);
            }
            else
            {
                printf("Error info : %s\n", jsonobj_trace["status"]["message"].asString().c_str());
                faces.erase(it++);
            }

        }
        faces = temp_faces;

        cv::imshow("video-demo", img);
        //在视频播放期间按键退出
        if (flag && waitKey(100) >= 0)
            break;
        else if (waitKey(33) >= 0)
            break;

    }

    //释放
    capture.release();
}

int test_gungnir(void* parser, std::string gungnir_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/e8dda5b6c23915758ffcba00753ae2a9.mp4");
    //capture.open("D:/opencv/QQ20220809100238.mp4");
    //capture.open("D:/opencv/more.jpg");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    while (1)
    {
        capture >> img;
        if (img.empty())
            break;

        Json::Reader reader(Json::Features::strictMode());
        Json::Value jsonobj_param;
        Json::Value jsonobj_result;
        jsonobj_param.clear();
        jsonobj_param["instance_guid"] = Json::Value(gungnir_guid);
        jsonobj_param["format"] = Json::Int(1);
        jsonobj_param["height"] = Json::Int(img.rows);
        jsonobj_param["width"] = Json::Int(img.cols);
        char* result_str = parser_parse(parser, "Gungnir.detect", writer.write(jsonobj_param).c_str(),
            reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
        reader.parse(result_str, jsonobj_result);
        Json::Value jsonobj_face;
        if (jsonobj_result["status"]["code"].asInt() == 0)
        {
            int list_size = jsonobj_result["hatrectwithhatinfo_list"].size();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_face = jsonobj_result["hatrectwithhatinfo_list"][i];
                int x, y;
                x = jsonobj_face["x"].asInt();
                y = jsonobj_face["y"].asInt();
                int width, height;
                width = jsonobj_face["width"].asInt();
                height = jsonobj_face["height"].asInt();
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255, 255, 255), 2);

            }
        }
        else
        {
            printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
            continue;
        }
        parser_free(result_str);
        result_str = nullptr;


        cv::imshow("video-demo", img);
        //在视频播放期间按键退出
        if (waitKey(33) >= 0)
            break;

    }

    //释放
    capture.release();
}

int test_romancia(void* parser, std::string romancia_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/weixin.jpg");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    capture >> img;

    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    jsonobj_param.clear();
    jsonobj_param = romancia_data;
    jsonobj_param["instance_guid"] = Json::Value(romancia_guid);
    jsonobj_param["format"] = Json::Int(1);
    jsonobj_param["height"] = Json::Int(img.rows);
    jsonobj_param["width"] = Json::Int(img.cols);

    //适用于128维特征提取（Gaius）的人脸对齐
        //char* result_str = parser_parse(parser, "Romancia.alignFace128", writer.write(jsonobj_param).c_str(),
        //	reinterpret_cast<char*>(img.data), 1ll * img.channels() * img.cols * img.rows, results, 100000);
        //人脸对齐
    char* result_str = parser_parse(parser, "Romancia.alignFace", writer.write(jsonobj_param).c_str(),
        reinterpret_cast<char*>(img.data), 1ll * img.channels() * img.cols * img.rows, results, 100000);
    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        aligned_images_data["aligned_images"] = jsonobj_result["aligned_images"];
        aligned_images_data["format"] = jsonobj_result["format"];
    }
    else
    {
        std::cout << jsonobj_result["status"]["message"].asString() << std::endl;
    }
    parser_free(result_str);
    result_str = nullptr;

    cv::imshow("video-demo", img);
    //在视频播放期间按键退出
    waitKey(0);
    //释放
    capture.release();
}

int test_damocles(void* parser, std::string damocles_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    //capture.open("D:/opencv/e8dda5b6c23915758ffcba00753ae2a9.mp4");
    //capture.open("D:/opencv/QQ20220809100238.mp4");
    capture.open("D:/opencv/49909ccdec3c04b95dddb5b9917b96ed.mp4");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    while (1)
    {
        capture >> img;
        if (img.empty())
            break;
        Json::Reader reader(Json::Features::strictMode());
        Json::Value jsonobj_param;
        Json::Value jsonobj_result;
        jsonobj_param.clear();
        jsonobj_param["instance_guid"] = Json::Value(damocles_guid);
        jsonobj_param["action_cmd"] = Json::Int(3);    //// { 0: 眨眼; 1 : 张嘴; 2 : 点头; 3 : 左摇头; 4 : 右摇头}
        jsonobj_param["format"] = Json::Int(1);
        jsonobj_param["height"] = Json::Int(img.rows);
        jsonobj_param["width"] = Json::Int(img.cols);
        jsonobj_param["facerect"]["x"] = Json::Int(0);
        jsonobj_param["facerect"]["y"] = Json::Int(0);
        jsonobj_param["facerect"]["width"] = Json::Int(img.cols);
        jsonobj_param["facerect"]["height"] = Json::Int(img.rows);
        char* result_str = parser_parse(parser, "Damocles.presentation_attack_detect", writer.write(jsonobj_param).c_str(),
            reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
        reader.parse(result_str, jsonobj_result);
        bool flag = 0;
        if (jsonobj_result["status"]["code"].asInt() == 0)
        {
            if (jsonobj_result["presentation_attack_result"].asBool() == true)
            {

                int x, y;
                x = jsonobj_param["facerect"]["x"].asInt() + 10;
                y = jsonobj_param["facerect"]["y"].asInt() + 10;
                int width, height;
                width = jsonobj_param["facerect"]["width"].asInt() - 50;
                height = jsonobj_param["facerect"]["height"].asInt() - 200;
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255, 255, 255), 2);
                flag = 1;
            }
        }
        else
        {
            printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
            continue;
        }
        parser_free(result_str);
        result_str = nullptr;


        cv::imshow("video-demo", img);
        //在视频播放期间按键退出
        if (flag && waitKey(500) >= 0)
            break;
        if (waitKey(33) >= 0)
            break;
    }

    //释放
    capture.release();
}

int test_mjollner(void* parser, std::string mjollner_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/QQ20220816171122.jpg");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    while (1)
    {
        capture >> img;
        if (img.empty())
            break;
        Json::Reader reader(Json::Features::strictMode());
        Json::Value jsonobj_param;
        Json::Value jsonobj_result;
        jsonobj_param.clear();
        jsonobj_param["instance_guid"] = Json::Value(mjollner_guid);
        jsonobj_param["format"] = Json::Int(1);
        jsonobj_param["height"] = Json::Int(img.rows);
        jsonobj_param["width"] = Json::Int(img.cols);

        //选择整张图作为检测区域
        jsonobj_param["roi"]["x"] = Json::Int(0);
        jsonobj_param["roi"]["y"] = Json::Int(0);
        jsonobj_param["roi"]["height"] = Json::Int(img.rows);
        jsonobj_param["roi"]["width"] = Json::Int(img.cols);
        char* result_str = parser_parse(parser, "Mjollner.detect", writer.write(jsonobj_param).c_str(),
            reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
        reader.parse(result_str, jsonobj_result);
        if (jsonobj_result["status"]["code"].asInt() == 0)
        {
            if (jsonobj_result["strinfo_list"].size())
            {
                int list_size = jsonobj_result["strinfo_list"].size();
                Json::Value jsonobj_mjollner = jsonobj_result["strinfo_list"];
                printf("new =  %s \n", writer.write(jsonobj_mjollner).c_str());
                printf("old =  %s \n", writer.write(jsonobj_result["strinfo_list"]).c_str());


                for (int i = 0; i < list_size; i++)
                {
                    int x1, y1;//右下角   
                    x1 = jsonobj_mjollner[i]["location"][0]["x"].asInt();
                    y1 = jsonobj_mjollner[i]["location"][0]["y"].asInt();
                    int x2, y2;//左上角 
                    x2 = jsonobj_mjollner[i]["location"][2]["x"].asInt();
                    y2 = jsonobj_mjollner[i]["location"][2]["y"].asInt();
                    rectangle(img, Point(x2, y2), Point(x1, y1), Scalar(128, 255, 150), 2);
                    std::cout << "string = " << jsonobj_mjollner[i]["strinfo"].asCString() << std::endl;
                    std::cout << "angle = " << jsonobj_mjollner[i]["angle"].asDouble() << std::endl;

                }
            }
        }
        else
        {
            printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
            continue;
        }
        parser_free(result_str);
        result_str = nullptr;


        cv::imshow("video-demo", img);
        //在视频播放期间按键退出
        waitKey(0);
    }

    //释放
    capture.release();
}

std::vector<Json::Value> test_gaius(void* parser, std::string gaius_guid, Json::FastWriter& writer)
{

    std::vector< Json::Value > ans;
    //videocapture结构创建一个catture视频对象

    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(gaius_guid);
    jsonobj_param["format"] = aligned_images_data["format"];
    jsonobj_param["aligned_images"] = aligned_images_data["aligned_images"];
    ///Gaius 128维特征提取
    //jsonobj_param["has_mask"] = Json::Value(false);
    //char* result_str = parser_parse(parser, "Gaius.forward", writer.write(jsonobj_param).c_str(),
    //	nullptr, 0, results, 100000);

    //Gaius 128维模拟口罩特征提取
    char* result_str = parser_parse(parser, "Gaius.make_mask_forward", writer.write(jsonobj_param).c_str(),
        nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        if (jsonobj_result["features"].size())
        {
            int list_size = jsonobj_result["features"].size();
            Json::Value jsonobj_gaius; jsonobj_gaius.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_gaius = jsonobj_result["features"][i];
                ans.push_back(jsonobj_gaius);

            }
        }
        printf("sonobj_result[\"features\"].size= %d\n", jsonobj_result["features"].size());

    }
    else
    {
        printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
    }
    parser_free(result_str);
    result_str = nullptr;

    return ans;
}

std::vector<Json::Value> test_selene(void* parser, std::string selene_guid, Json::FastWriter& writer)
{

    std::vector< Json::Value > ans;
    //videocapture结构创建一个catture视频对象

    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(selene_guid);
    jsonobj_param["format"] = aligned_images_data["format"];
    jsonobj_param["aligned_images"] = aligned_images_data["aligned_images"];

    //Selene 256维特征提取
    //char* result_str = parser_parse(parser, "Selene.forward", writer.write(jsonobj_param).c_str(),
    //	nullptr, 0, results, 100000);

    //Selene 256维模拟口罩特征提取
    char* result_str = parser_parse(parser, "Selene.make_mask_forward", writer.write(jsonobj_param).c_str(),
        nullptr, 0, results, 100000);

    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        if (jsonobj_result["features"].size())
        {
            int list_size = jsonobj_result["features"].size();
            Json::Value jsonobj_gaius; jsonobj_gaius.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_gaius = jsonobj_result["features"][i];
                ans.push_back(jsonobj_gaius);

            }
        }
        printf("sonobj_result[\"features\"].size= %d\n", jsonobj_result["features"].size());

    }
    else
    {
        printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
    }
    parser_free(result_str);
    result_str = nullptr;

    return ans;
}

std::vector<Json::Value> test_cassius(void* parser, std::string cassius_guid, Json::FastWriter& writer)
{

    std::vector< Json::Value > ans;
    //videocapture结构创建一个catture视频对象

    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(cassius_guid);
    jsonobj_param["format"] = aligned_images_data["format"];
    jsonobj_param["aligned_images"] = aligned_images_data["aligned_images"];

    char* result_str = parser_parse(parser, "Cassius.forward", writer.write(jsonobj_param).c_str(),
        nullptr, 0, results, 100000);

    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        if (jsonobj_result["features"].size())
        {
            int list_size = jsonobj_result["features"].size();
            Json::Value jsonobj_gaius; jsonobj_gaius.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_gaius = jsonobj_result["features"][i];
                ans.push_back(jsonobj_gaius);

            }
        }
        printf("sonobj_result[\"features\"].size= %d\n", jsonobj_result["features"].size());

    }
    else
    {
        printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
    }
    parser_free(result_str);
    result_str = nullptr;

    return ans;
}

Json::Value test_irisviel_load(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param)//人员库加载
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    char* result_str = parser_parse(parser, "Irisviel.load_databases", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_search(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param)//人员库搜索
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    jstr_param["top"] = Json::Value(3);
    //jstr_param["min_similarity"] = Json::Value(0.8);
    for (int i = 0; i < 128; ++i)
    {
        jstr_param["feature"][i] = Json::Value(0.1);
    }
    char* result_str = parser_parse(parser, "Irisviel.search", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_clear(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param)//人员库清除缓存  清内存
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    char* result_str = parser_parse(parser, "Irisviel.clear", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_removeAll(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param)//人员库清空  清内存和磁盘
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    char* result_str = parser_parse(parser, "Irisviel.remove_all", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_removeRecords(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, std::vector<std::string>keys)//人员库批量删除记录  
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    for (int i = 0; i < keys.size(); ++i)
        jstr_param["keys"][i] = Json::Value(keys[i]);
    char* result_str = parser_parse(parser, "Irisviel.remove_records", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_removeRecord(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, std::string key)//人员库删除记录  
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    jstr_param["key"] = Json::Value(key);
    char* result_str = parser_parse(parser, "Irisviel.remove_record", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_addRecords(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, std::vector<Json::Value > temp_features)//人员库批量添加记录 
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    for (int i = 0; i < temp_features.size(); i++)
    {
        jstr_param["data"][i]["feature"] = temp_features[i]["feature"];
        std::string temp = "name"; temp += 'a' + i;
        jstr_param["data"][i]["key"] = Json::Value(temp);
    }

    char* result_str = parser_parse(parser, "Irisviel.add_records", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_addRecord(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, Json::Value temp_features)//人员库添加记录 
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    jstr_param["data"]["feature"] = temp_features["feature"];
    jstr_param["data"]["key"] = Json::Value("777777777777777");

    char* result_str = parser_parse(parser, "Irisviel.add_record", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_updateRecords(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, std::vector<Json::Value> Json_keys)//人员库批量更新记录  
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    /// 日常
    for (int i = 0; i < Json_keys.size(); ++i)
        jstr_param["data"][i] = Json_keys[i];

    ///  临时
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 128; ++j)
        {
            jstr_param["data"][i]["feature"][j] = Json::Value(0.1);
        }
        jstr_param["data"][i]["key"] = Json::Value("");
    }

    char* result_str = parser_parse(parser, "Irisviel.update_records", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

Json::Value test_irisviel_updateRecord(void* parser, std::string irisviel_guid, Json::FastWriter& writer, Json::Value& jstr_param, Json::Value Json_key)//人员库更新记录  
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_result;
    jstr_param.clear();  // 
    jstr_param["instance_guid"] = Json::Value(irisviel_guid);
    /// 日常
    jstr_param["data"] = Json_key;

    /// 临时
    for (int j = 0; j < 128; ++j)
    {
        jstr_param["data"]["feature"][j] = Json::Value(0.1);
    }
    jstr_param["data"]["key"] = Json::Value("9527");

    char* result_str = parser_parse(parser, "Irisviel.update_record", writer.write(jstr_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    parser_free(result_str);
    result_str = nullptr;
    return jsonobj_result;
}

int test_valklyrs(void* parser, std::string valklyrs_guid, Json::FastWriter& writer)
{
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/e8dda5b6c23915758ffcba00753ae2a9.mp4");

    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    while (1)
    {
        capture >> img;
        if (img.empty())
            break;

        Json::Reader reader(Json::Features::strictMode());
        Json::Value jsonobj_param;
        Json::Value jsonobj_result;
        jsonobj_param.clear();
        jsonobj_param["instance_guid"] = Json::Value(valklyrs_guid);
        jsonobj_param["format"] = Json::Int(1);
        jsonobj_param["height"] = Json::Int(img.rows);
        jsonobj_param["width"] = Json::Int(img.cols);
        char* result_str = parser_parse(parser, "Valklyrs.detect", writer.write(jsonobj_param).c_str(),
            reinterpret_cast<char*>(img.data), img.channels() * img.cols * img.rows, results, 100000);
        reader.parse(result_str, jsonobj_result);
        if (jsonobj_result["status"]["code"].asInt() == 0)
        {
            int list_size = jsonobj_result["results"]["person_list"].size();
            Json::Value jsonobj_person;
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_person = jsonobj_result["results"]["person_list"][i];
                int x, y;
                x = jsonobj_person["coordinates"]["x"].asInt();
                y = jsonobj_person["coordinates"]["y"].asInt();
                int width, height;
                width = jsonobj_person["coordinates"]["width"].asInt();
                height = jsonobj_person["coordinates"]["height"].asInt();
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255, 255, 255), 2);

                std::vector<std::string>person_key = jsonobj_person["attributes"].getMemberNames();
                for (int j = 0; j < person_key.size(); j++)
                {
                    std::cout << person_key[j] << "=" << jsonobj_person["attributes"][person_key[j]] << std::endl;
                }
                std::cout << "++++++++++++++++++++  Gorgeous dividing line  +++++++++++++++++++++++" << std::endl << std::endl;
            }
            list_size = jsonobj_result["results"]["vehicle_list"].size();
            Json::Value jsonobj_vehicle;
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_vehicle = jsonobj_result["results"]["vehicle_list"][i];
                int x, y;
                x = jsonobj_vehicle["coordinates"]["x"].asInt();
                y = jsonobj_vehicle["coordinates"]["y"].asInt();
                int width, height;
                width = jsonobj_vehicle["coordinates"]["width"].asInt();
                height = jsonobj_vehicle["coordinates"]["height"].asInt();
                rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255, 255, 0), 2);

                std::vector<std::string>vehicle_key = jsonobj_person["attributes"].getMemberNames();
                for (int j = 0; j < vehicle_key.size(); j++)
                {
                    std::cout << vehicle_key[j] << "=" << jsonobj_person["attributes"][vehicle_key[j]] << std::endl;
                }
                std::cout << "++++++++++++++++++++  Gorgeous dividing line  +++++++++++++++++++++++" << std::endl << std::endl;
            }
        }
        else
        {
            printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
            continue;
        }
        parser_free(result_str);
        result_str = nullptr;

        cv::imshow("video-demo", img);
        //在视频播放期间按键退出
        if (waitKey(33) >= 0)
            break;

    }

    //释放
    capture.release();
}

//融合对齐128维特征提取
std::vector<Json::Value> test_romancia_gaius(void* parser, std::string romancia_guid, std::string gaius_guid, Json::FastWriter& writer)
{

    std::vector< Json::Value > ans;
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/more.jpg");
    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return ans;
    }
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    capture >> img;

    jsonobj_param.clear();
    jsonobj_param = romancia_data;
    jsonobj_param["romancia_instance_guid"] = Json::Value(romancia_guid);
    jsonobj_param["gaius_instance_guid"] = Json::Value(gaius_guid);
    jsonobj_param["format"] = Json::Int(1);
    jsonobj_param["height"] = Json::Int(img.rows);
    jsonobj_param["width"] = Json::Int(img.cols);
    jsonobj_param["has_mask"] = Json::Value(false);

    char* result_str = parser_parse(parser, "Fusion.Romancia.alignFace128.Gaius.forward", writer.write(jsonobj_param).c_str(),
        nullptr, 0, results, 100000);

    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        if (jsonobj_result["features"].size())
        {
            int list_size = jsonobj_result["features"].size();
            Json::Value jsonobj_gaius; jsonobj_gaius.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_gaius = jsonobj_result["features"][i];
                ans.push_back(jsonobj_gaius);
            }
        }
        printf("sonobj_result[\"features\"].size= %d\n", jsonobj_result["features"].size());
    }
    else
    {
        printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
    }
    parser_free(result_str);
    result_str = nullptr;

    cv::imshow("video-demo", img);
    //在视频播放期间按键退出
    waitKey(0);
    //释放
    capture.release();

    return ans;
}

//融合对齐256维特征提取
std::vector<Json::Value> test_romancia_selene(void* parser, std::string romancia_guid, std::string selene_guid, Json::FastWriter& writer)
{
    std::vector< Json::Value > ans;
    Json::Reader reader(Json::Features::strictMode());
    Json::Value jsonobj_param;
    Json::Value jsonobj_result;
    //videocapture结构创建一个catture视频对象
    VideoCapture capture;
    //连接视频
    capture.open("D:/opencv/test/more.jpg");
    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return ans;
    }
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 创建视频中每张图片对象
    Mat img;
    capture >> img;

    jsonobj_param.clear();
    jsonobj_param = romancia_data;
    jsonobj_param["romancia_instance_guid"] = Json::Value(romancia_guid);
    jsonobj_param["gaius_instance_guid"] = Json::Value(selene_guid);
    jsonobj_param["format"] = Json::Int(1);
    jsonobj_param["height"] = Json::Int(img.rows);
    jsonobj_param["width"] = Json::Int(img.cols);

    char* result_str = parser_parse(parser, "Fusion.Romancia.alignFace.Selene.forward", writer.write(jsonobj_param).c_str(),
        nullptr, 0, results, 100000);

    reader.parse(result_str, jsonobj_result);
    if (jsonobj_result["status"]["code"].asInt() == 0)
    {
        if (jsonobj_result["features"].size())
        {
            int list_size = jsonobj_result["features"].size();
            Json::Value jsonobj_selene; jsonobj_selene.clear();
            for (int i = 0; i < list_size; i++)
            {
                jsonobj_selene = jsonobj_result["features"][i];
                ans.push_back(jsonobj_selene);
            }
        }
        printf("sonobj_result[\"features\"].size= %d\n", jsonobj_result["features"].size());
    }
    else
    {
        printf("Error info : %s\n", jsonobj_result["status"]["message"].asString().c_str());
    }
    parser_free(result_str);
    result_str = nullptr;

    cv::imshow("video-demo", img);
    //在视频播放期间按键退出
    waitKey(0);
    //释放
    capture.release();

    return ans;
}

int main()
{
    results = malloc(100000);

    Json::Reader reader(Json::Features::strictMode());
    Json::FastWriter writer;
    Json::Value jsonobj_result;
    Json::Value jsonobj_param;

    const char config_file_path[] =
        "D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/configure_file/plugin_configure.json";
    void* parser = parser_new_instance();//++++++++++++++++++++++++++
    char* result_str = parser_init_plugin(parser, config_file_path, "");//++++++++++++++++++++++++++
    reader.parse(result_str, jsonobj_result);

    if (jsonobj_result["status"]["code"].asInt() == 0)
        printf("Successfully init sdk.\n");
    else
    {
        printf("Error info : %s \n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }

#pragma region
    // 创建Longinus实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    writer.write(jsonobj_param);
    result_str = parser_parse(parser, "Longinus.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string longinus_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        longinus_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    test_longinus(parser, longinus_guid, writer); // 人脸识别 + 追踪

    // 删除Longinus实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(longinus_guid);
    result_str = parser_parse(parser, "Longinus.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Longinus Demo

#pragma region
    // 创建Gungnir实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    result_str = parser_parse(parser, "Gungnir.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string gungnir_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        gungnir_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    //  test_gungnir(parser, gungnir_guid, writer); // 人头检测

    /// 删除Gungnir实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(gungnir_guid);
    result_str = parser_parse(parser, "Gungnir.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Gungnir Demo

#pragma region
    //创建Romancia实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    result_str = parser_parse(parser, "Romancia.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string romancia_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        romancia_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    test_romancia(parser, romancia_guid, writer); // 人脸对齐
    /// 删除Romancia实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(romancia_guid);
    result_str = parser_parse(parser, "Gungnir.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Romancia Demo

#pragma region
    // 创建Damocles实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["use_int8"] = Json::Value(false);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    result_str = parser_parse(parser, "Damocles.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string damocles_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        damocles_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    //test_damocles(parser, damocles_guid, writer); // 活体动作检测   （相当不准）

    /// 删除Damocles实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(damocles_guid);
    result_str = parser_parse(parser, "Damocles.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Damocles Demo

#pragma region
    // 创建Mjollner实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    result_str = parser_parse(parser, "Mjollner.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string mjollner_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        mjollner_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    //test_mjollner(parser, mjollner_guid, writer);//  文字检测   

    /// 删除Mjollner实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(mjollner_guid);
    result_str = parser_parse(parser, "Mjollner.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Mjollner Demo

#pragma region
    // 创建Gaius实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    jsonobj_param["use_int8"] = Json::Value(false);
    result_str = parser_parse(parser, "Gaius.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string gaius_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        gaius_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    std::vector<Json::Value > temp_features128;
    temp_features128 = test_gaius(parser, gaius_guid, writer);//  128特征提取   

    // 删除Gaius实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(gaius_guid);
    result_str = parser_parse(parser, "Gaius.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Gaius Demo

#pragma region
    // 创建Selene实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["model_type"] = Json::Int(2);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    jsonobj_param["use_int8"] = Json::Value(false);
    result_str = parser_parse(parser, "Selene.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string selene_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        selene_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    std::vector<Json::Value > temp_features256;
    temp_features256 = test_selene(parser, selene_guid, writer);//  256特征提取   

    // 删除Selene实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(selene_guid);
    result_str = parser_parse(parser, "Selene.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Selene Demo

#pragma region
    // 创建Cassius实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["model_type"] = Json::Int(0);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    jsonobj_param["use_int8"] = Json::Value(false);
    result_str = parser_parse(parser, "Cassius.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string cassius_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        cassius_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    std::vector<Json::Value > temp_features512;
    //temp_features512 = test_cassius(parser, cassius_guid, writer);//  512特征提取   

    // 删除Cassius实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(cassius_guid);
    result_str = parser_parse(parser, "Cassius.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Cassius Demo

#pragma region
    // 创建Irisviel实例
    jsonobj_param.clear();
    jsonobj_param["dimension"] = Json::Int(128);
    jsonobj_param["working_directory"] = Json::Value("D:/opencv/test");
    result_str = parser_parse(parser, "Irisviel.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string irisviel_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        irisviel_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;

    Json::Value jstr_param, jstr_result;
    int flag = -1;
    do {
        printf("Input Irisviel key (0-9), (-1 exit): ");
        scanf_s("%d", &flag);
        std::vector<std::string> keys;
        std::string key;
        std::vector<Json::Value> Json_keys;
        Json::Value Json_key;
        if (flag == 0)
            jstr_result = test_irisviel_load(parser, irisviel_guid, writer, jstr_param);//人员库加载
        else if (flag == 1)
            jstr_result = test_irisviel_search(parser, irisviel_guid, writer, jstr_param);//人员库搜索
        else if (flag == 2)
            jstr_result = test_irisviel_clear(parser, irisviel_guid, writer, jstr_param);//人员库清除缓存  清内存
        else if (flag == 3)
            jstr_result = test_irisviel_removeAll(parser, irisviel_guid, writer, jstr_param);//人员库清空  清内存和磁盘
        else if (flag == 4)
            jstr_result = test_irisviel_removeRecords(parser, irisviel_guid, writer, jstr_param, keys);//人员库批量删除记录 
        else if (flag == 5)
            jstr_result = test_irisviel_removeRecord(parser, irisviel_guid, writer, jstr_param, key);//人员库删除记录  
        else if (flag == 6)
            jstr_result = test_irisviel_addRecords(parser, irisviel_guid, writer, jstr_param, temp_features128);//人员库批量添加记录 
        else if (flag == 7)
            jstr_result = test_irisviel_addRecord(parser, irisviel_guid, writer, jstr_param, temp_features128[0]);//人员库添加记录  
        else if (flag == 8)
            jstr_result = test_irisviel_updateRecords(parser, irisviel_guid, writer, jstr_param, Json_keys);//人员库批量更新记录 
        else if (flag == 9)
            jstr_result = test_irisviel_updateRecord(parser, irisviel_guid, writer, jstr_param, Json_key);//人员库更新记录  
        else
            break;
    } while (flag != -1);



    // 删除Irisviel实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(irisviel_guid);
    result_str = parser_parse(parser, "Irisviel.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;

#pragma endregion Irisviel Demo

#pragma region
    // 创建Valklyrs实例
    jsonobj_param.clear();
    jsonobj_param["device"] = Json::Int(-1);
    jsonobj_param["models_directory"] = Json::Value("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/models");
    result_str = parser_parse(parser, "Valklyrs.new", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    reader.parse(result_str, jsonobj_result);
    std::string valklyrs_guid;
    if (jsonobj_result["status"]["code"].asInt() == 0)
        valklyrs_guid = jsonobj_result["instance_guid"].asString();
    else
    {
        printf("Error info :%s\n", jsonobj_result["status"]["message"].asString().c_str());
        exit(-1);
    }
    parser_free(result_str);
    result_str = nullptr;
    // test_valklyrs(parser, valklyrs_guid, writer); // 行人属性和车辆属性检测
    /// 删除Valklyrs实例
    jsonobj_param.clear();
    jsonobj_param["instance_guid"] = Json::Value(valklyrs_guid);
    result_str = parser_parse(parser, "Valklyrs.delete", writer.write(jsonobj_param).c_str(), nullptr, 0, results, 100000);
    parser_free(result_str);
    result_str = nullptr;
#pragma endregion Valklyrs Demo

    parser_release_instance(parser);//++++++++++++++++++++++++++
    parser = nullptr;
    //  free(results);
    return 0;
}

#pragma region
#pragma endregion end------------


