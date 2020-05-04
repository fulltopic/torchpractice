/*
 * tenhouconn.h
 *
 *  Created on: Apr 17, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_TENHOUCONN_H_
#define INCLUDE_TENHOUCLIENT_TENHOUCONN_H_

#include <sys/socket.h>
#include <arpa/inet.h>

#include <string>

class TenhouTcpConn {
private:
	int sock;

//	static const std::string ServerIp = "133.242.10.78";
//	static const int ServerPort = 10080;
	static const std::string ServerIp;
//	static const int ServerPort = 12345;
	static const int BufferSize = 1024;

	char sBuffer[BufferSize];
	char rBuffer[BufferSize];
public:
	TenhouTcpConn();
	~TenhouTcpConn();

	bool connServer();
	void connSend(std::string msg);
	std::string connRcv();
};


#endif /* INCLUDE_TENHOUCLIENT_TENHOUCONN_H_ */
