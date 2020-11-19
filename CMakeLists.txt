cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project (torchpractice)

set(CAFFE2_SRC_PATH /home/zf/workspaces/workspace_cpp/pytorch_1_3/pytorch)
set(CAFFE2_SRC_INC_PATH ${CAFFE2_SRC_PATH}/torch/include/)

set(CAFFE2_BLD_PATH ${CAFFE2_SRC_PATH}/build/)
set(CAFFE2_GEN_INC_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-3.7/torch/include/)
set(CAFFE2_GEN_LIB_PATH ${CAFFE2_BLD_PATH}/lib/)
set(CAFFE2_LIB_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-3.7/torch/lib/)
set(STATIC_LIB_PATH ${CAFFE2_GEN_LIB_PATH})

set(static_libs ${STATIC_LIB_PATH}/libprotoc.a 
	${STATIC_LIB_PATH}/libprotobuf.a 
	${STATIC_LIB_PATH}/libpthreadpool.a
	${STATIC_LIB_PATH}/libc10d.a)
set(shared_libs opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc torch torch_cpu glog gflags boost_system  boost_filesystem boost_thread c10 rt pthread leveldb lmdb python2.7)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)

set(PYTORCH_CPP_RL_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl)
set(PYTORCH_CPP_RL_LIB_PATH ${PYTORCH_CPP_RL_PATH}/example/lib)
set(pytorch_cpp_rl_lib ${PYTORCH_CPP_RL_PATH}/build/libcpprl.a ${PYTORCH_CPP_RL_PATH}/build/example/lib/libzmq/lib/libzmq.a)


include_directories(include)
include_directories(${CAFFE2_SRC_INC_PATH})
include_directories(${CAFFE2_GEN_INC_PATH})
include_directories(${CAFFE2_SRC_PATH}/torch/csrc/api/include)
include_directories(/home/zf/workspaces/workspace_cpp/matplotlib-cpp)
include_directories(/usr/include/python2.7)
include_directories(${PYTORCH_CPP_RL_PATH}/include)
include_directories(${PYTORCH_CPP_RL_PATH}/example)
include_directories(${PYTORCH_CPP_RL_LIB_PATH}/msgpack-c/include)

add_subdirectory(src/pytools)
add_subdirectory(src/lmdbtools)
add_subdirectory(src/nets)
add_subdirectory(src/nets/supervisednet)
add_subdirectory(src/tenhouclient)
add_subdirectory(src/policy)
add_subdirectory(src/gymtest)
add_subdirectory(src/rltest)
add_subdirectory(src/utils)
add_subdirectory(src/selfserver)

link_directories(${CAFFE2_LIB_PATH})

add_executable(mnist src/mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${shared_libs})

add_executable(deduce src/testtmpdeduce.cpp)
target_compile_features(deduce PUBLIC cxx_range_for)
target_link_libraries(deduce ${shared_libs})

add_executable(sintest src/sin.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(sintest PUBLIC cxx_range_for)
target_link_libraries(sintest ${shared_libs})

add_executable(shakespearetest src/shakespeare.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(shakespearetest PUBLIC cxx_range_for)
target_link_libraries(shakespearetest ${shared_libs})

#add_executable(generator src/TextGenerator.cpp $<TARGET_OBJECTS:pytools>)
#target_compile_features(generator PUBLIC cxx_range_for)
#target_link_libraries(generator ${shared_libs})

add_executable(plottest src/pytools/testplotsin.cpp $<TARGET_OBJECTS:pytools>)
target_compile_features(plottest PUBLIC cxx_range_for)
target_link_libraries(plottest ${shared_libs})

add_executable(lmdbtest src/testmjdata.cpp)
target_compile_features(lmdbtest PUBLIC cxx_range_for)
target_link_libraries(lmdbtest ${shared_libs} ${static_libs})

add_executable(lmdbscenetest src/testmjlmdb.cpp $<TARGET_OBJECTS:lmdbtools>)
target_compile_features(lmdbscenetest PUBLIC cxx_range_for)
target_link_libraries(lmdbscenetest ${shared_libs} ${static_libs})

#add_executable(convrnntest src/convrnntest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
#target_compile_features(convrnntest PUBLIC cxx_range_for)
#target_link_libraries(convrnntest ${shared_libs} ${static_libs})

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

add_executable(lstmtest src/lstmtest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(lstmtest PUBLIC cxx_range_for)
target_link_libraries(lstmtest ${shared_libs} ${static_libs})

add_executable(testbalance src/testbalance.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(testbalance PUBLIC cxx_range_for)
target_link_libraries(testbalance ${shared_libs} ${static_libs})

add_executable(cnntransfertest src/cnntransfertest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(cnntransfertest PUBLIC cxx_range_for)
target_link_libraries(cnntransfertest ${shared_libs} ${static_libs})

add_executable(testimplicit src/testimplicit.cpp)
target_compile_features(testimplicit PUBLIC cxx_range_for)
target_link_libraries(testimplicit ${shared_libs} )


add_executable(grusteptest src/testgrustep.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(grusteptest PUBLIC cxx_range_for)
target_link_libraries(grusteptest ${shared_libs} ${static_libs})

add_executable(testgrutransstep src/testgrutransstep.cpp  $<TARGET_OBJECTS:supervisednet> $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:pytools>)
target_compile_features(testgrutransstep PUBLIC cxx_range_for)
target_link_libraries(testgrutransstep ${shared_libs} ${static_libs})

add_executable(rnnlentest src/testrnnlen.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools>)
target_compile_features(rnnlentest PUBLIC cxx_range_for)
target_link_libraries(rnnlentest ${shared_libs} ${static_libs})

add_executable(testdebug src/testdebug.cpp)
target_compile_features(testdebug PUBLIC cxx_range_for)
target_link_libraries(testdebug ${shared_libs})

########################################################## TEST ####################################################
add_executable(parsertest src/tenhouclienttest/testparser.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(parsertest PUBLIC cxx_range_for)
target_link_libraries(parsertest ${shared_libs} ${static_libs})

add_executable(testtensor src/tenhouclienttest/testtensor.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(testtensor PUBLIC cxx_range_for)
target_link_libraries(testtensor ${shared_libs} ${static_libs})

add_executable(gentest src/tenhouclienttest/testgenerator.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(gentest PUBLIC cxx_range_for)
target_link_libraries(gentest ${shared_libs} ${static_libs})

add_executable(fileserver src/tenhouclienttest/testfileserver.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(fileserver PUBLIC cxx_range_for)
target_link_libraries(fileserver ${shared_libs} ${static_libs})

#add_executable(loggertest src/tenhouclienttest/testlogger.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:utils>)
#target_compile_features(loggertest PUBLIC cxx_range_for)
#target_link_libraries(loggertest ${shared_libs} ${static_libs})

add_executable(clienttest src/tenhouclienttest/tenhouclient.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(clienttest PUBLIC cxx_range_for)
target_link_libraries(clienttest ${shared_libs} ${static_libs})

add_executable(dirserver src/tenhouclienttest/testdirserver.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(dirserver PUBLIC cxx_range_for)
target_link_libraries(dirserver ${shared_libs} ${static_libs})

add_executable(dirclient src/tenhouclienttest/testdirclient.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(dirclient PUBLIC cxx_range_for)
target_link_libraries(dirclient ${shared_libs} ${static_libs})

add_executable(asioclient src/tenhouclienttest/testasioclient.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:supervisednet> $<TARGET_OBJECTS:utils>)
target_compile_features(asioclient PUBLIC cxx_range_for)
target_link_libraries(asioclient ${shared_libs} ${static_libs})

add_executable(asiodirclient src/tenhouclienttest/testdirasioclient.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(asiodirclient PUBLIC cxx_range_for)
target_link_libraries(asiodirclient ${shared_libs} ${static_libs})

add_executable(testmjclient src/tenhouclienttest/testmjclient.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:supervisednet> $<TARGET_OBJECTS:utils>)
target_compile_features(testmjclient PUBLIC cxx_range_for)
target_link_libraries(testmjclient ${shared_libs} ${static_libs})

#add_executable(testgrutransfer src/nettest/gru2523test.cpp  $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:supervisednet>)
add_executable(testgrutransfer src/nettest/gru2523test.cpp  $<TARGET_OBJECTS:supervisednet>)
target_compile_features(testgrutransfer PUBLIC cxx_range_for)
target_link_libraries(testgrutransfer ${shared_libs} ${static_libs})

################################################ Util test ###############################################
add_executable(testpad src/testmaskrnn.cpp $<TARGET_OBJECTS:lmdbtools>)
target_compile_features(testpad PUBLIC cxx_range_for)
target_link_libraries(testpad ${shared_libs} ${static_libs})


############################################### Gym test ##################################################
add_executable(lunartest src/gymtest/lunartest.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunartest PUBLIC cxx_range_for)
target_link_libraries(lunartest ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

add_executable(lunartest2 src/gymtest/lunartest2.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunartest2 PUBLIC cxx_range_for)
target_link_libraries(lunartest2 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

add_executable(lunalstm src/gymtest/lunarlstm.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm PUBLIC cxx_range_for)
target_link_libraries(lunalstm ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})



add_executable(lunalstm2 src/gymtest/lunarlstm2.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm2 PUBLIC cxx_range_for)
target_link_libraries(lunalstm2 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})


add_executable(lunalstm3 src/gymtest/lunarlstm3.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm3 PUBLIC cxx_range_for)
target_link_libraries(lunalstm3 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

add_executable(lunalstm4 src/gymtest/lunarlstm4.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm4 PUBLIC cxx_range_for)
target_link_libraries(lunalstm4 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

add_executable(lunalstm5 src/gymtest/lunarlstm5.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm5 PUBLIC cxx_range_for)
target_link_libraries(lunalstm5 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

add_executable(lunalstm6 src/gymtest/lunarlstm6.cpp  $<TARGET_OBJECTS:gymtest>)
target_compile_features(lunalstm6 PUBLIC cxx_range_for)
target_link_libraries(lunalstm6 ${shared_libs} ${static_libs} ${pytorch_cpp_rl_lib})

######################################### RL test ######################################################
add_executable(rll2transfertest src/rltest/loadl2modeltest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:utils>)
target_compile_features(rll2transfertest PUBLIC cxx_range_for)
target_link_libraries(rll2transfertest ${shared_libs} ${static_libs})


add_executable(rlbatchtransfertest src/rltest/loadmaskbatchmodeltest.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:utils>)
target_compile_features(rlbatchtransfertest PUBLIC cxx_range_for)
target_link_libraries(rlbatchtransfertest ${shared_libs} ${static_libs})

add_executable(testmaskbatchnet src/rltest/testmaskbatchnet.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:utils>)
target_compile_features(testmaskbatchnet PUBLIC cxx_range_for)
target_link_libraries(testmaskbatchnet ${shared_libs} ${static_libs})

add_executable(testmaskbatchclient src/rltest/testmaskbatchclient.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:utils>)
target_compile_features(testmaskbatchclient PUBLIC cxx_range_for)
target_link_libraries(testmaskbatchclient ${shared_libs} ${static_libs})

add_executable(testworker src/rltest/testtrainworker.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:utils>)
target_compile_features(testworker PUBLIC cxx_range_for)
target_link_libraries(testworker ${shared_libs} ${static_libs})

add_executable(testtool src/rltest/testtool.cpp $<TARGET_OBJECTS:lmdbtools> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:policy> $<TARGET_OBJECTS:pytools> $<TARGET_OBJECTS:utils>)
target_compile_features(testtool PUBLIC cxx_range_for)
target_link_libraries(testtool ${shared_libs} ${static_libs})

add_executable(teststrand src/rltest/teststrand.cpp)
target_compile_features(teststrand PUBLIC cxx_range_for)
target_link_libraries(teststrand ${shared_libs} ${static_libs})



########################################### Server test ########################################
add_executable(testserverutils src/selfserver/testutils.cpp $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:selfserver> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(testserverutils PUBLIC cxx_range_for)
target_link_libraries(testserverutils ${shared_libs} ${static_libs})

add_executable(testselfserver src/selfserver/testserver.cpp $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:selfserver> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(testselfserver PUBLIC cxx_range_for)
target_link_libraries(testselfserver ${shared_libs} ${static_libs})

add_executable(testreaderserver src/selfserver/readtestserver.cpp $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:selfserver> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(testreaderserver PUBLIC cxx_range_for)
target_link_libraries(testreaderserver ${shared_libs} ${static_libs})

add_executable(testreaderclient src/selfserver/readtestclient.cpp $<TARGET_OBJECTS:rltest> $<TARGET_OBJECTS:selfserver> $<TARGET_OBJECTS:tenhouclient> $<TARGET_OBJECTS:nets> $<TARGET_OBJECTS:utils>)
target_compile_features(testreaderclient PUBLIC cxx_range_for)
target_link_libraries(testreaderclient ${shared_libs} ${static_libs})
