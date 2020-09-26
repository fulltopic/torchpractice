/*
 * logger.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: zf
 */


#include "../../include/utils/logger.h"

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

//TODO: Move settings out of static function
std::shared_ptr<spdlog::logger> Logger::GetLogger() {
	static auto console = spdlog::stdout_color_mt("console");
	static bool setPattern = false;
	static auto logger = spdlog::get("console");

	if (!setPattern) {
		setPattern = true;
		logger->set_pattern("[%t][%l]: %v");
		spdlog::set_level(spdlog::level::info);
	}

	return logger;
//	return spdlog::get("console");
}


