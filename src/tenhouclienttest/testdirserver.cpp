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
#include <filesystem>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>


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
//			cout << "Send --> " << msg << endl;
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
//				cout << "Get --> " << items[i] << endl;
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
				if (rcvMsgs[k].length() == 0) {
					continue;
				}
				if (rcvMsgs[k].find("<Z") != string::npos) {
					continue;
				}

				int diff = msgs[i].msg.compare(rcvMsgs[k]);
				if ( diff!= 0) {
					cout << "Failed to match: " << diff << endl
							<< msgs[i].msg << " --> " << endl
							<< rcvMsg << endl;
					cout << msgs[i].msg.length() << " --> " << rcvMsg.length() << endl;
					for (int diffIndex = 0; diffIndex < min(msgs[i].msg.length(), rcvMsg.length()); diffIndex ++) {
						if (rcvMsg[diffIndex] != msgs[i].msg[diffIndex]) {
							cout << "Diff " << msgs[i].msg[diffIndex] << " --> " << rcvMsg[diffIndex] << endl;
						} else {
							cout << "Same " << msgs[i].msg[diffIndex] << " --> " << rcvMsg[diffIndex] << endl;
						}
					}
					exit(-1);
				} else {
					cout << "Received expected message: " << msgs[i].msg << endl;
					i ++;
				}
			}
		} else {
			cout << "To send message: " << msgs[i].msg << endl;
			send(clientSock, msgs[i].msg.c_str(), msgs[i].msg.length(), 0);
			i ++;
			sleep(1);
		}

	}
}

void test(string dirPath) {
//	for (const auto &entry: std::filesystem::directory_iterator(dirPath)) {
//		cout << entry.path() << endl;
//	}
//
//
//	vector<Msg> msgs = parseFile(path);
//	int serverFd = createServerSock();
//	if (serverFd < 0) {
//		return;
//	}
//	int clientFd = acceptClient(serverFd);
//
//
//	service(clientFd, msgs);
}

void testDir(string dirPath) {
	cout << "testDir " << dirPath << endl;

	int serverFd = createServerSock();
	if (serverFd < 0) {
		return;
	}
	int clientFd = acceptClient(serverFd);
	cout << "Get client socket " << endl;

	boost::filesystem::path dir(dirPath);

	boost::filesystem::directory_iterator end_ite;
	for (boost::filesystem::directory_iterator ite(dir); ite != end_ite; ++ite) {
		cout << "To extract file " << ite->path().string() << endl;
		if (boost::filesystem::is_regular_file (ite->status())) {
			cout << "#####################################################################################" << endl;
			cout << ite->path().string() << endl;

			vector<Msg> msgs = parseFile(ite->path().string());
			if (msgs.size() == 0) {
				cout << "Not a valid file " << endl;
				continue;
			}

			service(clientFd, msgs);
		}
	}
}


int main(int argc, char** argv) {
	testDir(string(argv[1]));
}
