/*
 * asiotenhoufsm.h
 *
 *  Created on: May 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_
#define INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_

#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>

//#include "tenhoufsm.h"
#include "logger.h"
#include "netproxy.hpp"
#include "nets/grustep.h"

#include <iostream>
#include <string>
#include <vector>


#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

#include "tenhouclient/tenhoumsggenerator.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsgutils.h"

template<class NetType>
class asiotenhoufsm
	:public boost::enable_shared_from_this <asiotenhoufsm<NetType>> {
public:
	static const int GameEnd2NextTimeout;// = 2;
	static const int SceneEnd2StartTimeout;// = 5;
	static const int KATimeout;// = 15;
	static const int LNThreshold;
	static const int ReConnTimeout;

	//TODO: for instance
	const std::string name;

	typedef boost::shared_ptr<asiotenhoufsm<NetType>> pointer;
//	static pointer Create (boost::asio::io_context& io, NetProxy<RandomNet>& net);
	static pointer Create (boost::asio::io_context& io, NetProxy<NetType>& net, const std::string tenhouName);

	bool start();
	void reset();

	~asiotenhoufsm();

private:
	boost::asio::ip::tcp::resolver resolver;
	boost::asio::ip::tcp::endpoint serverP;
	boost::asio::ip::tcp::socket sock;
	boost::array<char, 512> rcvBuf;

//	NetProxy<RandomNet>& net;
	NetProxy<NetType> net;
	std::shared_ptr<spdlog::logger> logger;

	boost::asio::deadline_timer gameEnd2NextTimer;
	boost::asio::deadline_timer sceneEnd2StartTimer;
	boost::asio::deadline_timer kaTimer;
	boost::asio::deadline_timer reConnTimer;

	bool authenticated;
	bool gameBegin;
	int lnCount;

//	asiotenhoufsm(boost::asio::io_context& iio, NetProxy<RandomNet>& iNet);
	asiotenhoufsm(boost::asio::io_context& iio, NetProxy<NetType>& iNet, const std::string tenhouName);

	void send (std::string msg);
//	void handleSnd (StateType nextType, boost::system::error_code& e);

	void errState ();
	void heloState (const boost::system::error_code& e, std::size_t len);
	void authState (const boost::system::error_code& e, std::size_t len);
	void joinState (const boost::system::error_code& e, std::size_t len);
	void readyState (const boost::system::error_code& e, std::size_t len);
	void gameState (const boost::system::error_code& e, std::size_t len);
	void gameEndState (const boost::system::error_code& e, std::size_t len);
	void sceneEndState (const boost::system::error_code& e, std::size_t len);

	void gameEnd2NextTimerHandle(const boost::system::error_code& e);
	void sceneEnd2StartTimerHandle(const boost::system::error_code& e);
	void kaTimerHandle(const boost::system::error_code& e);
	void reConnTimerHandle(const boost::system::error_code& e);


	inline void logUnexpMsg(const std::string stateName, const std::string msg) {
		logger->error("{} received unexpected msg: {}", stateName, msg);
	}
	inline void logNetworkErr(const std::string stateName, const std::string msg) {
		logger->error("{} networking failure: {}", stateName, msg);
	}

	void processGameMsg(std::string msg);
	std::vector<std::string> splitRcvMsg(std::string msg);
};

using boost::asio::ip::tcp;

using G=TenhouMsgGenerator;
using P=TenhouMsgParser;
using U=TenhouMsgUtils;
using S=std::string;
using ErrorCode=boost::system::error_code;

template<class NetType>
const int asiotenhoufsm<NetType>::GameEnd2NextTimeout = 2;
template<class NetType>
const int asiotenhoufsm<NetType>::SceneEnd2StartTimeout = 5;
template<class NetType>
const int asiotenhoufsm<NetType>::KATimeout = 15;
template<class NetType>
const int asiotenhoufsm<NetType>::LNThreshold = 8;
template<class NetType>
const int asiotenhoufsm<NetType>::ReConnTimeout = 15;

//template<class NetType>
//const std::string asiotenhoufsm<NetType>::name = "ID5F706D6D-2WBML2Pe"; //testrl0

//static const std::string ServerIp = "127.0.0.1";
//static const int ServerPort = 26238;

static const std::string ServerIp = "133.242.10.78";
static const int ServerPort = 10080;
//static const std::string ServerIp = "127.0.0.1";
//static const int ServerPort = 26238;


#define RegRcv(handle) sock.async_receive(	\
boost::asio::buffer(rcvBuf),	\
boost::bind(&asiotenhoufsm::handle, this->shared_from_this(),	\
		boost::asio::placeholders::error,	\
		boost::asio::placeholders::bytes_transferred())	\
);

template<class NetType>
asiotenhoufsm<NetType>::asiotenhoufsm(boost::asio::io_context& iio, NetProxy<NetType>& iNet, const std::string tenhouName)
	: name(tenhouName),
	  resolver(iio),
	  serverP(boost::asio::ip::address::from_string(ServerIp), ServerPort),
	  sock(iio),
	  net(iNet),
	  logger(Logger::GetLogger()),
	  gameEnd2NextTimer(iio),
	  sceneEnd2StartTimer(iio),
	  kaTimer(iio),
	  reConnTimer(iio),
	  authenticated(false),
	  gameBegin(false),
	  lnCount(0)
{
}

template<class NetType>
asiotenhoufsm<NetType>::~asiotenhoufsm() {
	sock.close();
}

template<class NetType>
typename asiotenhoufsm<NetType>::pointer asiotenhoufsm<NetType>::Create(boost::asio::io_context& io, NetProxy<NetType>& net, const std::string tenhouName) {
	return pointer(new asiotenhoufsm<NetType>(io, net, tenhouName));
}

template<class NetType>
void asiotenhoufsm<NetType>::send (std::string msg) {
//	sock.async_send(boost::asio::buffer(msg),
//			boost::bind(&asiotenhoufsm::handleSnd, shared_from_this(),
//					nextState,
//					boost::asio::placeholders::error));

	if(msg.length() == 0) {
		return;
	}
	if (msg.compare(StateReturnType::Nothing) == 0) {
		return;
	}

//	logger->info("Test mutable buffer");
//	std::string testStr("<HELO name=\"NoName\" tid=\"f0\" sx=\"M\" />");
//	testStr.resize(testStr.length() + 1);
//	boost::asio::mutable_buffer testBuf((void*)testStr.data(), testStr.length());
////	logger->info("Generated mutable buffer {}: {}", (char*)testBuf.data(), testBuf.size());
//	logger->info("Extended mutable buffer {}: {}", (char*)testBuf.data(), testBuf.size());
//	auto testData = (char*)testBuf.data();
//	testData[testBuf.size() - 1] = 0;
//	logger->info("testbuf {}: {}", msg, testBuf.size());

//	std::string expStr("<HELO name=\"NoName\" tid=\"f0\" sx=\"M\" />\\0");
//	logger->info("Size of {} = {}", testStr, testStr.length());
//	logger->info("Size of {} = {}", expStr, expStr.length());
	if (msg.find(StateReturnType::SplitToken) != S::npos) {
		std::vector<S> items;
		boost::split(items, msg,
				boost::is_any_of(StateReturnType::SplitToken), boost::token_compress_on);
		logger->debug("To sent splitted msg {}", items.size());
		for (int i = 0; i < items.size(); i ++) {
			items[i].resize(items[i].length() + 1);
			sock.send(boost::asio::buffer(items[i].data(), items[i].length()));
			logger->debug("Sent {}", items[i]);
			sleep(1);
		}
	} else {
		msg.resize(msg.length() + 1);
		logger->debug("Sent {}, {}", msg, msg.length());
//		sock.send(boost::asio::buffer(msg.data(), msg.length() + 1));
		sock.send(boost::asio::buffer(msg.data(), msg.length()));
		sleep(1);
	}

//	sock.send(boost::asio::buffer(msg));
}

template<class NetType>
std::vector<S> asiotenhoufsm<NetType>::splitRcvMsg(S msg) {
	if (msg.find(">") == S::npos) {
		logUnexpMsg("gameState", msg);
		return {};
	}

	std::vector<S> rc;
	int lastIndex = 0;
	int index = msg.find(">", lastIndex);

	while (index != S::npos) {
		S subMsg = msg.substr(lastIndex, (index - lastIndex + 1));
		boost::trim(subMsg);
		rc.push_back(subMsg);

		lastIndex = index + 1;
		index = msg.find(">", lastIndex);
	}

	return rc;
}

//void asiotenhoufsm::handleSnd(StateType nextType, boost::system::error_code& e) {
//
//}

template<class NetType>
void asiotenhoufsm<NetType>::heloState(const boost::system::error_code& e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("helo received msg: {}", msg);

		if (msg.find("HELO") != S::npos) {
			auto parts = P::ParseHeloReply(msg);
			S authMsg = G::GenAuthReply(parts);
			S pxrMsg = G::GenPxrMsg();

			send (authMsg + StateReturnType::SplitToken + pxrMsg);

			RegRcv(authState);

//			sock.async_receive(
//					boost::asio::buffer(rcvBuf),
//					boost::bind(&asiotenhoufsm::authState, shared_from_this(),
//							boost::asio::placeholders::error,
//							boost::asio::placeholders::bytes_transferred())
//			);
		} else if (msg.find("GO") != S::npos) {
			//TODO: It is reconnect
			send(G::GenGoMsg());
			send(G::GenNextReadyMsg());
			send(G::GenNextReadyMsg());
			RegRcv(joinState);
		} else {
			this->logUnexpMsg("heloState", msg);
		}
	} else {
		this->logNetworkErr("heloState", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::authState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("auth received msg: {}", msg);

		if (msg.find("LN") != S::npos) {
			send (G::GenJoinMsg());
			RegRcv(joinState);
		} else if (msg.find("REJOIN") != S::npos) {
			send (G::GenRejoinMsg(msg));
			RegRcv(joinState);
		} else if (msg.find("RANKING") != S::npos) {
			RegRcv(authState);
		} else if (msg.find("GO") != S::npos) {
			//TODO: It is reconnect
			send(G::GenGoMsg());
			send(G::GenNextReadyMsg());
			send(G::GenNextReadyMsg());
			RegRcv(joinState);
		}
		else {
			logger->error("Received unexpected msg: {}", msg);
			RegRcv(authState);
		}
	} else {
		logger->error("auth networking failure: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::reConnTimerHandle(const boost::system::error_code& e) {
	logger->error("Too many LN received, to restart client");
	reset();
}


template<class NetType>
void asiotenhoufsm<NetType>::joinState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("join received msg: {}", msg);

		bool toReset = false;
		auto msgs = splitRcvMsg(msg);
		for (int i = 0; i < msgs.size(); i ++) {
			if (msgs[i].find("GO") != S::npos) {
				send(G::GenGoMsg());
				send(G::GenNextReadyMsg());
				send(G::GenNextReadyMsg());
			} else if (msgs[i].find("UN") != S::npos) {
				//nothing
			} else if (msgs[i].find("REJOIN") != S::npos) {
//				send("<JOIN t=\"0,1,r\" />");
				send(G::GenRejoinMsg(msgs[i]));
			} else if (msgs[i].find("LN") != S::npos) {
				send(G::GenPxrMsg());
				lnCount ++;
				if (lnCount > LNThreshold) {
					toReset = true;
				}
			} else if (msgs[i].find("TAIKYOKU") != S::npos) {
				gameBegin = true;
			} else {
				logUnexpMsg("joinState", msgs[i]);
			}
		}

		if (toReset) {
			kaTimer.cancel();
			reConnTimer.expires_from_now(boost::posix_time::seconds(ReConnTimeout));
			reConnTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::reConnTimerHandle, this->shared_from_this(),
								boost::asio::placeholders::error));
		} else {
			if (gameBegin) {
				RegRcv(readyState);
				gameBegin = false;
				lnCount = 0;
			} else {
				RegRcv(joinState);
			}
		}



//		if (msgs.size() == 1) {
//			if (msg.find("GO") != S::npos) {
//				RegRcv(joinState);
//			} else if (msg.find("LN") != S::npos) {
//				send(G::GenPxrMsg());
//				RegRcv(joinState);
//			} else if (msg.find("UN") != S::npos) {
//				RegRcv(joinState);
//			} else if (msg.find("TAIKYOKU") != S::npos) {
//				send(G::GenGoMsg());
//				send(G::GenNextReadyMsg());
//				//+ StateReturnType::SplitToken + G::GenNextReadyMsg() + StateReturnType::SplitToken + G::GenNextReadyMsg());
//				RegRcv(readyState);
//			} else if (msg.find("REJOIN") != S::npos) {
//				send(G::GenRejoinMsg(msg));
//				RegRcv(joinState);
//			} else {
//				logUnexpMsg(msg);
//			}
//		} else {
//			for (int i = 0; i < msgs.size(); i ++) {
//				if (msgs[i].find("TAIKYOKU") != S::npos) {
////					send(G::GenGoMsg());
////					send(G::GenNextReadyMsg());
////					send(G::GenNextReadyMsg());
//					// + StateReturnType::SplitToken + G::GenNextReadyMsg() + StateReturnType::SplitToken + G::GenNextReadyMsg());
//					RegRcv(readyState);
//					return;
//				}
//
//
//
//				if (msgs[i].find("LN") != S::npos) {
//					send(G::GenPxrMsg());
//				} else if(msgs[i].find("UN") != S::npos) {
////					send(G::GenNextReadyMsg());
//				} else if (msgs[i].find("GO") != S::npos) {
////					send(G::GenGoMsg());
//					send(G::GenGoMsg());
//					send(G::GenNextReadyMsg());
//					send(G::GenNextReadyMsg());
//				} else if (msgs[i].find("REJOIN") != S::npos) {
//					send (G::GenRejoinMsg(msg));
//				} else {
//					logger->info("Pass msg: {}", msgs[i]);
//				}
//			}
//			RegRcv(joinState);
//		}
	} else {
		logNetworkErr("joinState", e.message());
		if (e == boost::asio::error::eof) {
			RegRcv(joinState);
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::readyState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("ready received msg: {}", msg);

		if (msg.find("INIT") != S::npos) {
//			reset();
			processGameMsg(msg);

			RegRcv(gameState);
		} else if (msg.find("PROF") != S::npos) {
			RegRcv(sceneEndState);
		} else if (P::IsGameEnd(msg)) {
			processGameMsg(msg);

			RegRcv(readyState);
		}else {
			logUnexpMsg("readyState", msg);
		}
	} else {
		logNetworkErr("readyState", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameEndState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("gameEnd received msg: {}", msg);

		if (msg.find("PROF") != S::npos) {
			gameEnd2NextTimer.cancel();

			sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
			sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::sceneEnd2StartTimerHandle, this->shared_from_this(),
					boost::asio::placeholders::error));
			RegRcv(sceneEndState);
		} else if (P::IsGameEnd(msg)) {
			gameEnd2NextTimer.cancel();

			processGameMsg(msg);
			send(G::GenNextReadyMsg());

			RegRcv(gameEndState);
			gameEnd2NextTimer.expires_from_now(boost::posix_time::seconds(GameEnd2NextTimeout));
			gameEnd2NextTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::gameEnd2NextTimerHandle, this->shared_from_this(),
					boost::asio::placeholders::error));
		} else {
			logUnexpMsg("gameEnd", msg);
		}
	} else {
		logNetworkErr("gameEnd", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameEnd2NextTimerHandle(const boost::system::error_code& e) {
	std::cout << "Test e " << e <<  e.message() << std::endl;
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("The gameEnd2NextTimer had been cancelled ");
	} else if (!e) {
		logger->warn("gameEnd2NextTimer expiring");
		sock.cancel();

		send(G::GenNoopMsg());
		RegRcv(readyState);
	} else {
		logger->error("gameEnd2NextTimer interrupted as: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("game received msg: {}", msg);

		if (msg.find("PROF") != S::npos) {
			auto msgs = splitRcvMsg(msg);

			if (msgs.size() == 1) {
				RegRcv(sceneEndState);
				return;
			} else {
				for (int i = 0; i < msgs.size(); i ++) {
					if (U::IsTerminalMsg(msgs[i])) {
						processGameMsg(msgs[i]);
					}
				}

				sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
				sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm::sceneEnd2StartTimerHandle, this->shared_from_this(),
									boost::asio::placeholders::error));
				RegRcv(sceneEndState);
			}
		}
		if (!U::IsGameMsg(msg)) {
			logger->warn("Received unexpected msg: {}", msg);
			RegRcv(gameState);
			return;
		}

		bool isTerminal = U::IsTerminalMsg(msg);
		processGameMsg(msg);

		if(isTerminal) {
			send(G::GenNextReadyMsg());
			RegRcv(readyState);
		} else {
			RegRcv(gameState);
		}
	} else {
		logNetworkErr("gameState", e.message());
		if (e == boost::asio::error::eof) {
			RegRcv(gameState);
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::sceneEndState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("sceneEnd received msg: {}", msg);

		if (U::IsTerminalMsg(msg)) {
			sceneEnd2StartTimer.cancel();

			processGameMsg(msg);
			send(G::GenNextReadyMsg());

			sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
			sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm::sceneEnd2StartTimerHandle, this->shared_from_this(),
					boost::asio::placeholders::error));
			RegRcv(sceneEndState);

		} else {
			logUnexpMsg("sceneEndState", msg);
		}
	} else {
		logNetworkErr("sceneEndState", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::sceneEnd2StartTimerHandle(const ErrorCode &e) {
	logger->info("sceneEnd2StartTimer e = {}: {} ",e.value(), e.message());
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("The sceneEnd2StartTimer had been cancelled ");
	} else if (!e) {
		logger->warn("sceneEnd2StartTimer expiring");
		sock.cancel();

		send(G::GenByeMsg());
		sleep(5);

		send(G::GenHeloMsg(name));
		RegRcv(heloState);
	} else {
		logger->error("sceneEnd2StartTimer interrupted as: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::kaTimerHandle(const boost::system::error_code& e) {
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("Katimer cancelled");
	} else if(!e) {
		send(G::GenKAMsg());

		kaTimer.expires_from_now(boost::posix_time::seconds(KATimeout));
		kaTimer.async_wait(boost::bind(&asiotenhoufsm::kaTimerHandle, this->shared_from_this(),
						boost::asio::placeholders::error));
	} else {
		logger->error("Katimer interrupted as: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::processGameMsg(S msg) {
	int lastIndex = 0;
	int index = msg.find(">", lastIndex);


	while (index != S::npos) {
		S subMsg = msg.substr(lastIndex, (index - lastIndex + 1));
		boost::trim(subMsg);
		logger->debug("To process msg {}", subMsg);

		S rc = net.processMsg(subMsg);
		send(rc);

		lastIndex = index + 1;
		index = msg.find(">", lastIndex);
	}
}


template<class NetType>
bool asiotenhoufsm<NetType>::start() {
	try {
		sock.open(tcp::v4());
		sock.connect(serverP);
		logger->info("Connected to {}, {}", ServerIp, ServerPort);

//		send (G::GenHeloMsg("NoName"));
		send(G::GenHeloMsg(name));
		RegRcv(heloState);
	} catch (std::exception& e) {
		logger->error("Failed to start connection: {}", e.what());
		return false;
	}

	kaTimer.expires_from_now(boost::posix_time::seconds(KATimeout));
	kaTimer.async_wait(boost::bind(&asiotenhoufsm::kaTimerHandle, this->shared_from_this(),
						boost::asio::placeholders::error));

	return true;
}

template<class NetType>
void asiotenhoufsm<NetType>::reset() {
	sock.cancel();
	gameEnd2NextTimer.cancel();
	sceneEnd2StartTimer.cancel();
	kaTimer.cancel();

	authenticated = false;
	gameBegin = false;
	lnCount = 0;

	sock.close();
	net.reset();

	start();
}





#endif /* INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_ */
