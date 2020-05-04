/*
 * tenhouconn.cpp
 *
 *  Created on: Apr 17, 2020
 *      Author: zf
 */



#include "tenhouclient/tenhouconn.h"

#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <cerrno>
#include <time.h>


using namespace std;

const std::string TenhouTcpConn::ServerIp = "127.0.0.1";
static const int ServerPort = 26238;

TenhouTcpConn::TenhouTcpConn():
				sock(0) {

}

TenhouTcpConn::~TenhouTcpConn() {
	//TODO: To close sock
	close(sock);
}

bool TenhouTcpConn::connServer() {
	struct sockaddr_in servAddr;

	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		cout << "Failed to create sock" << endl;
		return false;
	}

	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(ServerPort);

	if(inet_pton(AF_INET, ServerIp.c_str(), &servAddr.sin_addr) < 0) {
		cout << "Failed to bind address " << endl;
		return false;
	}

	struct timeval timeout;
	timeout.tv_sec = 3;
	timeout.tv_usec = 0;
	if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
		cout << "Failed to set timeout option " << strerror(errno) << endl;
		return false;
	}

	if (connect(sock, (struct sockaddr*)&servAddr, sizeof(servAddr)) < 0) {
		cout << "Failed to connect to server " << strerror(errno) << endl;
		return false;
	}

	return true;
}

void TenhouTcpConn::connSend(string msg) {
	send(sock, msg.c_str(), msg.length(), MSG_DONTWAIT);
}

string TenhouTcpConn::connRcv() {
	int len = read(sock, rBuffer, BufferSize);

	if (len <= 0) {
		return "TIMEOUT";
	}
	//TODO: Check if its copy
	return string(rBuffer, rBuffer + len);
}
