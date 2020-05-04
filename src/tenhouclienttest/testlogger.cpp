/*
 * testlogger.cpp
 *
 *  Created on: Apr 20, 2020
 *      Author: zf
 */


#include "tenhouclient/logger.h"
#include "spdlog/spdlog-inl.h"
#include "spdlog/common.h"
#include <map>
#include <string>

void test() {
//	spdlog::set_pattern("[%D %H:%M:%S] [%t] [%@] %v", spdlog::pattern_time_type::local);
	auto logger = Logger::GetLogger();
//	logger->set_pattern("[%D %H:%M:%S][%t][%L][%s] %v", spdlog::pattern_time_type::local);
	logger->info("Test");
	logger->error("Test error");
	logger->debug("Test debug");

	std::string s0 ("<JOIN t=\"0,1,r\" />");
	std::string s1 ("<JOIN t=\"0,1,r\" />");
	logger->info(s0.compare(s1));
}

int main() {
	test();
}

