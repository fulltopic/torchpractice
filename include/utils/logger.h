/*
 * logger.h
 *
 *  Created on: Apr 18, 2020
 *      Author: zf
 */

#ifndef INCLUDE_UTILS_LOGGER_H_
#define INCLUDE_UTILS_LOGGER_H_

#include "spdlog/sinks/rotating_file_sink.h"

class Logger {
	Logger() = delete;
	~Logger() = delete;

public:
	static std::shared_ptr<spdlog::logger> GetLogger();
};



#endif /* INCLUDE_UTILS_LOGGER_H_ */
