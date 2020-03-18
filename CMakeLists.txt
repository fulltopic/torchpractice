cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project (torchpractice)

set(CAFFE2_SRC_PATH /home/zf/workspaces/workspace_cpp/pytorch/)
set(CAFFE2_SRC_INC_PATH ${CAFFE2_SRC_PATH}/torch/include/)

set(CAFFE2_BLD_PATH /home/zf/workspaces/workspace_cpp/pytorch/build/)
set(CAFFE2_GEN_INC_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-2.7/torch/include/)
set(CAFFE2_GEN_LIB_PATH ${CAFFE2_BLD_PATH}/lib/)
set(CAFFE2_LIB_PATH /home/zf/workspaces/workspace_cpp/pytorch/build/lib.linux-x86_64-2.7/torch/lib/)
set(STATIC_LIB_PATH ${CAFFE2_GEN_LIB_PATH})

set(static_libs ${STATIC_LIB_PATH}/libprotoc.a 
	${STATIC_LIB_PATH}/libprotobuf.a 
	${STATIC_LIB_PATH}/libpthreadpool.a
	${STATIC_LIB_PATH}/libc10d.a)
set(shared_libs opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc torch glog gflags boost_system c10 rt pthread leveldb lmdb python2.7)
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_BUILD_TYPE Debug)

include_directories(include)
include_directories(${CAFFE2_SRC_INC_PATH})
include_directories(${CAFFE2_GEN_INC_PATH})
include_directories(/home/zf/workspaces/workspace_cpp/pytorch/torch/csrc/api/include)
include_directories(/home/zf/workspaces/workspace_cpp/matplotlib-cpp)
include_directories(/usr/include/python2.7)

add_subdirectory(src/pytools)
add_subdirectory(src/lmdbtools)
add_subdirectory(src/nets)

link_directories(${CAFFE2_LIB_PATH})

add_executable(mnist src/mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${shared_libs})

add_executable(lstmtest src/whatever.cpp)
target_compile_features(lstmtest PUBLIC cxx_range_for)
target_link_libraries(lstmtest ${shared_libs})

add_executable(deduce src/testtmpdeduce.cpp)
target_compile_features(deduce PUBLIC cxx_range_for)
target_link_libraries(deduce ${shared_libs})

add_executable(sintest src/sin.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(sintest PUBLIC cxx_range_for)
target_link_libraries(sintest ${shared_libs})

add_executable(shakespearetest src/shakespeare.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(shakespearetest PUBLIC cxx_range_for)
target_link_libraries(shakespearetest ${shared_libs})

add_executable(generator src/TextGenerator.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(generator PUBLIC cxx_range_for)
target_link_libraries(generator ${shared_libs})

add_executable(plottest src/pytools/testplotsin.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(plottest PUBLIC cxx_range_for)
target_link_libraries(plottest ${shared_libs})

add_executable(lmdbtest src/testmjdata.cpp)
target_compile_features(lmdbtest PUBLIC cxx_range_for)
target_link_libraries(lmdbtest ${shared_libs} ${static_libs})

add_executable(lmdbscenetest src/testmjlmdb.cpp $<TARGET_OBJECTS:lmdbtools>)
target_compile_features(lmdbscenetest PUBLIC cxx_range_for)
target_link_libraries(lmdbscenetest ${shared_libs} ${static_libs})

add_executable(convrnntest src/convrnntest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(convrnntest PUBLIC cxx_range_for)
target_link_libraries(convrnntest ${shared_libs} ${static_libs})

add_executable(testplot src/testplot.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(testplot PUBLIC cxx_range_for)
target_link_libraries(testplot ${shared_libs} ${static_libs})

add_executable(generateShuffleDb src/generateshufflelmdb.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:pytools>)
target_compile_features(generateShuffleDb PUBLIC cxx_range_for)
target_link_libraries(generateShuffleDb ${shared_libs} ${static_libs})

add_executable(cnndimtest src/testdbreader4cnn.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:pytools>)
target_compile_features(cnndimtest PUBLIC cxx_range_for)
target_link_libraries(cnndimtest ${shared_libs} ${static_libs})


add_executable(cnntest src/cnntest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(cnntest PUBLIC cxx_range_for)
target_link_libraries(cnntest ${shared_libs} ${static_libs})
