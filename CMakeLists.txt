# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("CMakeProject1")

# 将源代码添加到此项目的可执行文件。
add_executable (CMakeProject1  "src/main.cpp" "include/test.h")


#指定要引用的dll库的头文件所在路径
include_directories("include/opencv/build/include")
include_directories("include/opencv/build/include/opencv2")
#指定该dll库的lib所在路径
link_directories("lib/opencv")
#指定链接库的名字，opencv_world460d.lib 文件放在对应项目文件中
target_link_libraries(CMakeProject1 opencv_world460d)


#指定要引用的dll库的头文件所在路径
include_directories("include/jsoncpp-library/include")
#指定该dll库的lib所在路径
link_directories("lib/jsoncpp-library")
#指定链接库的名字，jsoncpp.lib 文件放在对应项目文件中
target_link_libraries(CMakeProject1 jsoncpp)


#指定要引用的dll库的头文件所在路径
include_directories("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/include")
#指定该dll库的lib所在路径
link_directories("D:/Code/Glasssix_CV_SDK_2.9.21.20220811_beta/lib/windows/x64/release")
#指定链接库的名字，libparser.lib 文件放在对应项目文件中
target_link_libraries(CMakeProject1 libparser)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject1 PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
