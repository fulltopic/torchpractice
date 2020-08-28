/*
 * testfileserver.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: zf
 */



#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cerrno>

#include <boost/algorithm/string.hpp>


using namespace std;

int createServerSock() {
	int serverFd;
	struct sockaddr_in address;
	int opt = 1;

	serverFd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
            &opt, sizeof(opt));
    address.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
//    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(26238);


    if (bind(serverFd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    	cout << "Server bind failure " << endl;
    	cout << strerror(errno) << endl;
    	return -1;
    }

    if (listen(serverFd, 3) < 0) {
    	cout << "Listen accept failed " << endl;
    	return -1;
    }

    cout << "Listening on " << address.sin_port << endl;
    return serverFd;
}

int acceptClient(int serverFd) {
	struct sockaddr_in address;
	int addressLen = sizeof(address);
	int newSocket = accept(serverFd,
			(struct sockaddr*)&address, (socklen_t*)&addressLen);
	if (newSocket < 0) {
		cout << "Failed to accept connection " << strerror(errno) << endl;
		return -1;
	}

	cout << "Accept client connection " << endl;
	return newSocket;
}

struct Msg {
	int type; //send = 0, get = 1
	string msg;
};

vector<Msg> parseFile(string path) {
	ifstream inputFile(path);
	string line;
	vector<Msg> msgs;

	while (getline(inputFile, line)) {
		if (line.find("Send") != string::npos) {
			int sendPos = line.find("Send");
			int tokenPos = line.find(":", sendPos);
			string msg = line.substr(tokenPos + 2);
			msgs.push_back({0, msg});
			cout << "Send --> " << msg << endl;
		} else if (line.find("Get") != string::npos) {
			int getPos = line.find("Get");
			int tokenPos = line.find(":", getPos);
			string msg = line.substr(tokenPos + 2);

			boost::replace_all(msg, "> <", ">####<");
			vector<string> items;
			boost::split(items, msg, boost::is_any_of("####"), boost::token_compress_on);
			for (int i = 0; i < items.size(); i ++) {
				boost::trim(items[i]);
				msgs.push_back({1, items[i]});
				cout << "Get --> " << items[i] << endl;
			}

//			if (msg.find("> <") != string::npos) {
//				int tPos = msg.find("> <");
//				string msg0 = msg.substr(0, tPos + 1);
//				msgs.push_back({1, msg0});
//				cout << "Get --> " << msg0 << endl;
//
//				string msg1 = msg.substr(tPos + 2);
//				if (msg1.find("> <") != string::npos) {
//					boost::trim(msg1);
//					int t1Pos = msg1.find("> <");
//					string msg2 = msg1.substr(0, t1Pos + 1);
//					string msg3 = msg1.substr(t1Pos + 2);
//					msgs.push_back({1, msg2});
//					cout << "Get --> " << msg2 << endl;
//					msgs.push_back({1, msg3});
//					cout << "Get --> " << msg3 << endl;
//				} else {
//					msgs.push_back({1, msg1});
//					cout << "Get --> " << msg1 << endl;
//				}
//			} else {
//				msgs.push_back({1, msg});
//				cout << "Get --> " << msg << endl;
//			}
		} else {
//			cout << "Pass " << line << endl;
		}
	}

	return msgs;
}

static void service(int clientSock, vector<Msg>& msgs) {
	int index = 0;
	char buff[1024] = {0};
	int i = 0;
//	for (int i = 0; i < msgs.size(); i ++){
	while (i < msgs.size()) {
		if (msgs[i].type == 0) {
			if (msgs[i].msg.find("<Z") != string::npos) {
				cout << "Pass KA message " << endl;
				i ++;
				continue;
			}

			int len = read(clientSock, buff, 1024);
			string rcvMsg (buff, buff + len);
			boost::trim(rcvMsg);
			if (rcvMsg.length() == 0) {
				break;
			}

			cout << "Received raw message " << rcvMsg << endl;
			boost::replace_all(rcvMsg, "<", "####<");
			vector<string> rcvMsgs;
			boost::split(rcvMsgs, rcvMsg,
						boost::is_any_of("####"), boost::token_compress_on);

			for (int k = 0; k < rcvMsgs.size(); k ++) {
				boost::trim(rcvMsgs[k]);
//				cout << "Received message: " << rcvMsgs[k] << endl;
				if (rcvMsgs[k].length() == 0) {
					continue;
				}
				if (rcvMsgs[k].find("<Z") != string::npos) {
					continue;
				}

				auto bPos = rcvMsgs[k].find("<");
				auto ePos = rcvMsgs[k].find(">");
				cout << rcvMsgs[k] << " --> get poses " << bPos << ", " << ePos << endl;
				rcvMsgs[k] = rcvMsgs[k].substr(bPos, (ePos - bPos + 1));
				int diff = msgs[i].msg.compare(rcvMsgs[k]);
				if ( diff!= 0) {
					cout << "Failed to match: " << diff << endl
							<< msgs[i].msg << " --> " << endl
							<< rcvMsgs[k] << endl;
					cout << msgs[i].msg.length() << " --> " << rcvMsgs[k].length() << endl;
					for (int diffIndex = 0; diffIndex < min(msgs[i].msg.length(), rcvMsgs[k].length()); diffIndex ++) {
						if (rcvMsg[diffIndex] != msgs[i].msg[diffIndex]) {
							cout << "Diff " << msgs[i].msg[diffIndex] << " --> " << rcvMsgs[k][diffIndex] << endl;
						} else {
							cout << "Same " << msgs[i].msg[diffIndex] << " --> " << rcvMsgs[k][diffIndex] << endl;
						}
					}
				} else {
					cout << "Received expected message: " << msgs[i].msg << endl;
					i ++;
				}
			}
		} else {
			cout << "To send message: " << msgs[i].msg << endl;
			send(clientSock, msgs[i].msg.c_str(), msgs[i].msg.length(), 0);
			i ++;
			sleep(3);
		}

	}
}

void test(string path) {
//	string path = "/home/zf/workspaces/workspace_cpp/torchpractice/src/tenhouclienttest/reachtestlog.txt";
//	string path = "/home/zf/workspaces/workspace_cpp/torchpractice/build/tenhoulogs/tenhoulog38.txt";
	vector<Msg> msgs = parseFile(path);
	int serverFd = createServerSock();
	if (serverFd < 0) {
		return;
	}
	int clientFd = acceptClient(serverFd);


	service(clientFd, msgs);
}


int main(int argc, char** argv) {
	test(string(argv[1]));
}
