/*
 * asiotenhoufsm.h
 *
 *  Created on: May 7, 2020
 *      Author: zf
 */

#ifndef INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_
#define INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_


//#include "tenhoufsm.h"
#include "netproxy.hpp"
#include "nets/grustep.h"

#include <iostream>
#include <string>
#include <vector>
#include <atomic>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

#include "../utils/logger.h"
#include "tenhouclient/tenhoumsggenerator.h"
#include "tenhouclient/tenhoumsgparser.h"
#include "tenhouclient/tenhoumsgutils.h"

template<class NetType>
class asiotenhoufsm
	:public boost::enable_shared_from_this <asiotenhoufsm<NetType>> {
public:
	enum ResetStatus {
		NotReset = 0,
		Resetting = 1,
		Resetted = 2,
	};

	static const int GameEnd2NextTimeout;// = 2;
	static const int SceneEnd2StartTimeout;// = 5;
	static const int DetectNetUpdateTimeout;
	static const int KATimeout;// = 15;
	static const int LNThreshold; // = 8
	static const int ReConnTimeout; // = 15
	static const int ReConnTrial; // = 10
	static const int ReConnThreshold; // = 60 (second)

	const std::string name;

	typedef boost::shared_ptr<asiotenhoufsm<NetType>> pointer;
	static pointer Create (boost::asio::io_context& io, std::shared_ptr<NetProxy<NetType>> net,
			const std::string serverIp, const int serverPort, const std::string tenhouName, bool isPrivate = false);

	bool start();
	void reset();
	//TODO: Add close state

	~asiotenhoufsm();

private:
	boost::asio::ip::tcp::resolver resolver;
	boost::asio::ip::tcp::endpoint serverP;
	boost::asio::ip::tcp::socket sock;
	boost::array<char, 512> rcvBuf;

	std::shared_ptr<NetProxy<NetType>> net;
	std::shared_ptr<spdlog::logger> logger;

	boost::asio::deadline_timer gameEnd2NextTimer;
	boost::asio::deadline_timer sceneEnd2StartTimer;
	boost::asio::deadline_timer kaTimer;
	boost::asio::deadline_timer reConnTimer;

	bool authenticated;
	bool gameBegin;
	int lnCount;
	int restartCount;
	bool isValid;
	bool isPrivate;

	std::atomic<int> resetState;

//	asiotenhoufsm(boost::asio::io_context& iio, NetProxy<RandomNet>& iNet);
	asiotenhoufsm(boost::asio::io_context& iio, std::shared_ptr<NetProxy<NetType>> iNet,
			const std::string ServerIp, const int serverPort, const std::string tenhouName, bool privateLobby = false);

	bool send (std::string msg);
//	void handleSnd (StateType nextType, boost::system::error_code& e);

	void errState ();
	void heloState (const boost::system::error_code& e, std::size_t len);
	void authState (const boost::system::error_code& e, std::size_t len);
	void heloLobbyState (const boost::system::error_code& e, std::size_t len);
	void joinState (const boost::system::error_code& e, std::size_t len);
	void readyState (const boost::system::error_code& e, std::size_t len);
	void gameState (const boost::system::error_code& e, std::size_t len);
	void gameEndState (const boost::system::error_code& e, std::size_t len);
	void sceneEndState (const boost::system::error_code& e, std::size_t len);

	void gameEnd2NextTimerHandle(const boost::system::error_code& e);
	void sceneEnd2StartTimerHandle(const boost::system::error_code& e);
	void kaTimerHandle(const boost::system::error_code& e);
	void reConnTimerHandle(const boost::system::error_code& e);
	void toReset (int n = 1, bool validFsm = true);  //To break too long command line

	inline void logUnexpMsg(const std::string stateName, const std::string msg) {
		logger->error("{} --> {} received unexpected msg: {}", net->getName(), stateName, msg);
	}

	//TODO: To recover and reconnect
	inline void logNetworkErr(const std::string stateName, const std::string msg) {
		logger->error("{} networking failure: {}", stateName, msg);
	}

	inline bool isWorking() { return isValid; }

	void processGameMsg(std::string msg);
	std::vector<std::string> splitRcvMsg(std::string msg);

	void close(bool validFsm = true);
};

using boost::asio::ip::tcp;

using G=TenhouMsgGenerator;
using P=TenhouMsgParser;
using U=TenhouMsgUtils;
using S=std::string;
using ErrorCode=boost::system::error_code;

//TODO: Move into setting?
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
template<class NetType>
const int asiotenhoufsm<NetType>::ReConnTrial = 30;
template<class NetType>
const int asiotenhoufsm<NetType>::ReConnThreshold = 10;
template<class NetType>
const int asiotenhoufsm<NetType>::DetectNetUpdateTimeout = 30;


#define RegRcv(handle) sock.async_receive(	\
boost::asio::buffer(rcvBuf),	\
boost::bind(&asiotenhoufsm::handle, this->shared_from_this(),	\
		boost::asio::placeholders::error,	\
		boost::asio::placeholders::bytes_transferred())	\
);

template<class NetType>
asiotenhoufsm<NetType>::asiotenhoufsm(boost::asio::io_context& iio, std::shared_ptr<NetProxy<NetType>> iNet,
		const std::string serverIp, const int serverPort, const std::string tenhouName, bool privateLobby)
	: name(tenhouName),
	  resolver(iio),
	  serverP(boost::asio::ip::address::from_string(serverIp), serverPort),
	  sock(iio),
	  net(iNet),
	  logger(Logger::GetLogger()),
	  gameEnd2NextTimer(iio),
	  sceneEnd2StartTimer(iio),
	  kaTimer(iio),
	  reConnTimer(iio),
	  authenticated(false),
	  gameBegin(false),
	  lnCount(0),
	  restartCount(0),
	  isValid(true),
	  isPrivate(privateLobby),
	  resetState(NotReset)
{
}

template<class NetType>
asiotenhoufsm<NetType>::~asiotenhoufsm() {
	sock.close();
}

template<class NetType>
typename asiotenhoufsm<NetType>::pointer asiotenhoufsm<NetType>::Create(boost::asio::io_context& io, std::shared_ptr<NetProxy<NetType>> net,
		const std::string serverIp, const int serverPort, const std::string tenhouName, bool isPrivate) {
	return pointer(new asiotenhoufsm<NetType>(io, net, serverIp, serverPort, tenhouName, isPrivate));
}

//TODO: Ensure only one reset invoked once, and no reset in normal running
//atomic status to ensure, and close immediately
template <class NetType>
void asiotenhoufsm<NetType>::toReset(int n, bool validFsm) {
	int tmpResetState = resetState;
	if (tmpResetState != NotReset) {
		return;
	}

	if (!resetState.compare_exchange_weak(tmpResetState, Resetting)) {
		return;
	}

	close(validFsm);

	if (validFsm) {
		logger->warn("To reset connection after {} seconds ", n * ReConnTimeout);
		reConnTimer.expires_from_now(boost::posix_time::seconds(n * ReConnTimeout));
		reConnTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::reConnTimerHandle, this->shared_from_this(),
						boost::asio::placeholders::error));
	} else {
		logger->error("Invalid client id {}, end of fsm ", net->getName());
	}
}

template <class NetType>
void asiotenhoufsm<NetType>::close(bool validFsm) {
	isValid = false;

	sock.cancel();
	gameEnd2NextTimer.cancel();
	sceneEnd2StartTimer.cancel();
	kaTimer.cancel();
	reConnTimer.cancel();

	authenticated = false;
	gameBegin = false;
	lnCount = 0;

	sock.close();

	net->setGameEnd(validFsm);
//	if (validFsm) {
//		if (!net->isRunning()) {
//			logger->error("Set net end in fsm close");
//			if (net->setDetected(validFsm)) {
//				net->setGameEnd();
//			}
//		}
//	} else {
//		while (net->isRunning()) {
//			logger->warn("Waiting for network updating to set fsm invalid");
//			sleep(60);
//		}
//		logger->error("Set net end in fsm close");
//		if (net->setDetected(validFsm)) {
//			net->setGameEnd();
//		}
//	}

	logger->error("fsm terminates");
}

template<class NetType>
bool asiotenhoufsm<NetType>::send (std::string msg) {
//	sock.async_send(boost::asio::buffer(msg),
//			boost::bind(&asiotenhoufsm::handleSnd, shared_from_this(),
//					nextState,
//					boost::asio::placeholders::error));

	if(msg.length() == 0) {
		return true;
	}
	if (msg.compare(StateReturnType::Nothing) == 0) {
		return true;
	}

	if (msg.find(StateReturnType::SplitToken) != S::npos) {
		std::vector<S> items;
		boost::split(items, msg,
				boost::is_any_of(StateReturnType::SplitToken), boost::token_compress_on);
		logger->debug("To sent splitted msg {}", items.size());
		for (int i = 0; i < items.size(); i ++) {
			items[i].resize(items[i].length() + 1);
			try {
				sock.send(boost::asio::buffer(items[i].data(), items[i].length()));
				logger->debug("Sent {}", items[i]);
				sleep(1);
			} catch (boost::system::system_error& e) {
				logger->error ("Send error {}: {}", net->getName(), e.what());
				logger->info("To reset");
				toReset();
				return false;
			}
		}
	} else {
		msg.resize(msg.length() + 1);
		try {
			logger->debug("Sent {}, {}", msg, msg.length());
			sock.send(boost::asio::buffer(msg.data(), msg.length()));
			sleep(1);
		} catch (boost::system::system_error& e) {
			logger->error ("Send error {}: {}", net->getName(), e.what());
			logger->info("To reset");
			toReset();
			return false;
		}
	}

	return true;
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

			if (send (authMsg + StateReturnType::SplitToken + pxrMsg)) {
				RegRcv(authState);
			}
		} else if (msg.find("GO") != S::npos) {
			logger->info("{} ---> log file: {}", net->getName(), msg);

			if (send(G::GenGoMsg())) {
				if (send(G::GenNextReadyMsg())) {
					if (send(G::GenNextReadyMsg())) {
						RegRcv(joinState);
					}
				}
			}
		} else if (msg.find("ERR") != S::npos) {
			//TODO: To deal with the error, how if one client invalid in process running
			logger->error ("{} --> Invalid client id: {}", net->getName(), msg);
//			close();
			toReset(5, false); //Retry, set end of game at reset(), distinguish cancel and normal ending
		} else if (msg.find("SAIKAI") != S::npos) {
			RegRcv(readyState);
			//TODO: deal with saikai in helo
		} else {
			this->logUnexpMsg("heloState", msg);
			RegRcv(heloState);
		}
	} else {
		logNetworkErr("heloState", e.message());
		toReset(); //If heloState failed because of reset(), then reset would be cancelled as the reConnTimer canceled by reset()
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::heloLobbyState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("heloLobby received msg: {}", msg);

		if (msg.find("HELO") != S::npos) {
			auto parts = P::ParseHeloReply(msg);
			S authMsg = G::GenAuthReply(parts);
			S pxrMsg = G::GenPxrMsg();

			if (send (authMsg + StateReturnType::SplitToken + pxrMsg)) {
				RegRcv(heloLobbyState);
			}
		} else if (msg.find("LN") != S::npos) {
//			S chatMsg = G::GenLobbyChatMsg();
			S pxrMsg = G::GenLobbyPxrMsg();
			S joinMsg = G::GenLobbyJoinMsg();

			if (send( pxrMsg + StateReturnType::SplitToken + joinMsg)) {
				RegRcv(authState);
			}
		} else if (msg.find("SAIKAI") != S::npos) {
			RegRcv(readyState);
		} else if (msg.find("REJOIN") != S::npos) {
			if (send(G::GenLobbyRejoinMsg())) {
				RegRcv(joinState);
			}
		}
		else {
			logUnexpMsg("heloLobbyState", msg);
			RegRcv(heloLobbyState);
		}
	} else {
		logNetworkErr ("heloLobbyState", e.message());
		toReset();
	}
}


template<class NetType>
void asiotenhoufsm<NetType>::authState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("auth received msg: {}", msg);

		if (msg.find("LN") != S::npos) {
			if  (send (G::GenJoinMsg())) {
				RegRcv(joinState);
			}
		} else if (msg.find("REJOIN") != S::npos) {
			if (isPrivate) {
				if (send(G::GenLobbyRejoinMsg())) {
					RegRcv(joinState);
				}
			} else {
				if (send (G::GenRejoinMsg(msg))) {
					RegRcv(joinState);
				}
			}
		} else if (msg.find("RANKING") != S::npos) {
			RegRcv(authState);
		} else if (msg.find("GO") != S::npos) {
			logger->info("Reinit msg: {}", msg);
			//TODO: It is reinit
			if (send(G::GenGoMsg())) {
				if (send(G::GenNextReadyMsg())) {
					if (send(G::GenNextReadyMsg())) {
						RegRcv(joinState);
					}
				}
			}
		} else if (msg.find("REINIT") != S::npos) {
			logger->debug("Reinit received in authState for private lobby player");
			processGameMsg(msg);
			RegRcv(gameState);
		}
		else {
			logger->error("Received unexpected msg: {}", msg);
			RegRcv(authState);
		}
	} else {
		logger->error("auth networking failure: {}", e.message());
		toReset();
	}
}

//TODO: Make thread_safe
template<class NetType>
void asiotenhoufsm<NetType>::reConnTimerHandle(const boost::system::error_code& e) {
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("reConnTimer cancelled");
	} else if(!e) {
		logger->error("Network failure, to restart client {}", net->getName());
		start();
	} else {
		logger->error("reConnTimer interrupted as: {}", e.message());
//		reset();
	}

}


template<class NetType>
void asiotenhoufsm<NetType>::joinState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("join received msg: {}", msg);

		bool toResetGame = false;
		auto msgs = splitRcvMsg(msg);
		for (int i = 0; i < msgs.size(); i ++) {
			if (msgs[i].find("dan") != S::npos) {
				logger->info("Log file: {}", msgs[i]);
			}

			if (msgs[i].find("GO") != S::npos) {
				if (send(G::GenGoMsg())) {
					if (send(G::GenNextReadyMsg())) {
						send(G::GenNextReadyMsg());
					}
				}
			} else if (msgs[i].find("UN") != S::npos) {
				//nothing
			} else if (msgs[i].find("REJOIN") != S::npos) {
//				send("<JOIN t=\"0,1,r\" />");
				send(G::GenRejoinMsg(msgs[i]));
			} else if (msgs[i].find("LN") != S::npos) {
				if (send(G::GenPxrMsg())) {
					lnCount ++;
					if (lnCount > LNThreshold) {
						toResetGame = true;
						break;
					}
				}
			} else if (msgs[i].find("TAIKYOKU") != S::npos) {
				gameBegin = true;
			} else if (msgs[i].find("SAIKAI") != S::npos) {
				gameBegin = true;
			}
			else if (msgs[i].find("REINIT") != S::npos) {
				gameBegin = false;
				lnCount = 0;
				for (int j = i; j < msgs.size(); j ++) {
					processGameMsg(msgs[j]); //TODO: REINIT should be the last message
				}
				RegRcv(gameState);
				return;
			}else if (msgs[i].find("INIT") != S::npos) { //Reconn edge case
				gameBegin = false;
				lnCount = 0;
				for (int j = i; j < msgs.size(); j ++) {
					processGameMsg(msgs[j]);
				}
				RegRcv(gameState);
				return;
			}
			else if (msgs[i].find("RANKING") != S::npos) {
				RegRcv(joinState);
			} else {
				logUnexpMsg("joinState", msgs[i]);
			}
		}

		if (toResetGame) {
			lnCount = 0;
//			kaTimer.cancel();
			toReset();
//			reConnTimer.expires_from_now(boost::posix_time::seconds(ReConnTimeout));
//			reConnTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::reConnTimerHandle, this->shared_from_this(),
//								boost::asio::placeholders::error));
		} else {
			if (gameBegin) {
				RegRcv(readyState);
				gameBegin = false;
				lnCount = 0;
			} else {
				RegRcv(joinState);
			}
		}
	} else {
		logNetworkErr("joinState", e.message());
		if (e == boost::asio::error::eof) {
//			RegRcv(joinState);
//			reset();
			toReset();
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::readyState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("ready received msg: {}", msg);

		if (msg.find("INIT") != S::npos) {
			processGameMsg(msg);

			RegRcv(gameState);
		} else if (msg.find("PROF") != S::npos) {
			RegRcv(sceneEndState);
		} else if (P::IsGameEnd(msg)) {
			//Sometimes, sceneEndMsg --> readyState --> gameEndMsg
			processGameMsg(msg);
			RegRcv(readyState);
		}else {
			logUnexpMsg("readyState", msg);
			RegRcv(readyState);
		}
	} else {
		logNetworkErr("readyState", e.message());
//		reset();
		toReset();
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameEndState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		gameEnd2NextTimer.cancel(); //TODO: Check cancel  result, the first gameend in gameState set the timer

		S msg (rcvBuf.data(), len);
		logger->debug("gameEnd received msg: {}, timer cancelled", msg);

		auto msgs = splitRcvMsg(msg);
		bool isSceneEndMsg = false;
		bool isGameEndMsg = false;

		for (int i = 0; i < msgs.size(); i ++) {
			if (msgs[i].find("PROF") != S::npos) {
				isSceneEndMsg = true;
			} else if (P::IsGameEnd(msgs[i])) {
				isGameEndMsg = true;
				processGameMsg(msgs[i]);
			}
		}

		if (isSceneEndMsg) {
			sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
			sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::sceneEnd2StartTimerHandle, this->shared_from_this(),
					boost::asio::placeholders::error));
			RegRcv(sceneEndState);
		} else if (isGameEndMsg) {
//			send(G::GenNextReadyMsg()); //TODO: Send message in timeout

			gameEnd2NextTimer.expires_from_now(boost::posix_time::seconds(GameEnd2NextTimeout));
			gameEnd2NextTimer.async_wait(boost::bind(&asiotenhoufsm<NetType>::gameEnd2NextTimerHandle, this->shared_from_this(),
					boost::asio::placeholders::error));
			RegRcv(gameEndState);
		} else {
			logUnexpMsg("gameEnd", msg);
			RegRcv(gameEndState);
		}
	}
	else {
		logNetworkErr("gameEnd", e.message());
		if (e && (e == boost::asio::error::operation_aborted)) {
			logger->info("canceled by timer");
		} else {
//			reset();
			toReset();
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameEnd2NextTimerHandle(const boost::system::error_code& e) {
//	std::cout << "Test e " << e <<  e.message() << std::endl;
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("The gameEnd2NextTimer had been cancelled ");
	} else if (!e) {
		logger->warn("gameEnd2NextTimer expiring {}", net->getName());
		net->setGameEnd();

		sock.cancel();

		if (send(G::GenNextReadyMsg())) {
//			if (send(G::GenNoopMsg())) {
				RegRcv(readyState);
//			}
		}
	} else {
		logger->error("gameEnd2NextTimer interrupted as: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::gameState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		S msg (rcvBuf.data(), len);
		logger->debug("game received msg: {} ---> {}", len, msg);

		if (!U::IsGameMsg(msg)) {
			logger->warn("Received unexpected msg: {}", msg);
			RegRcv(gameState);
			return;
		}

		if (msg.find("PROF") != S::npos) {
			logger->info("PROF msg {}: {}", net->getName(), msg);
			auto msgs = splitRcvMsg(msg);

			for (int i = 0; i < msgs.size(); i ++) {
				if (U::IsTerminalMsg(msgs[i])) {
					processGameMsg(msgs[i]);
				}
			}
			sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
			sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm::sceneEnd2StartTimerHandle, this->shared_from_this(),
								boost::asio::placeholders::error));
			RegRcv(sceneEndState);

			return;
		} else {
			bool isTerminal = U::IsTerminalMsg(msg);
			processGameMsg(msg);

			if(isTerminal) {
				logger->info("GameEnd msg {}: {}", net->getName(), msg);
				gameEnd2NextTimer.expires_from_now(boost::posix_time::seconds(GameEnd2NextTimeout));
				gameEnd2NextTimer.async_wait(boost::bind(&asiotenhoufsm::gameEnd2NextTimerHandle, this->shared_from_this(),
								boost::asio::placeholders::error));
				RegRcv(gameEndState);
			} else {
				RegRcv(gameState);
			}
		}
	}
	else {
		logNetworkErr("gameState", e.message());
		if (e == boost::asio::error::eof) {
//			reset();
			toReset();
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::sceneEndState(const ErrorCode &e, std::size_t len) {
	if ((!e) || (e == boost::asio::error::message_size)) {
		sceneEnd2StartTimer.cancel();

		S msg (rcvBuf.data(), len);
		logger->debug("sceneEnd received msg: {}", msg);

		auto msgs = splitRcvMsg(msg);
		bool isGameEndMsg = false;
		for (int i = 0; i < msgs.size(); i ++) {
			if (P::IsGameEnd(msgs[i])) {
				isGameEndMsg = true;
				processGameMsg(msgs[i]);
			} else {
				logUnexpMsg("sceneEndState", msgs[i]);
			}
		}

		if (isGameEndMsg) {
			send(G::GenNextReadyMsg());
		}
		sceneEnd2StartTimer.expires_from_now(boost::posix_time::seconds(SceneEnd2StartTimeout));
		sceneEnd2StartTimer.async_wait(boost::bind(&asiotenhoufsm::sceneEnd2StartTimerHandle, this->shared_from_this(),
				boost::asio::placeholders::error));
		RegRcv(sceneEndState);
	} else {
		logNetworkErr("sceneEndState", e.message());
		if (e && (e == boost::asio::error::operation_aborted)) {
			logger->info("canceled by timer");
		} else {
//			reset();
			toReset();
		}
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::sceneEnd2StartTimerHandle(const ErrorCode &e) {
	logger->info("sceneEnd2StartTimer e = {}: {} ",e.value(), e.message());
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("The sceneEnd2StartTimer had been cancelled ");
	} else if (!e) {
		logger->warn("sceneEnd2StartTimer expiring");

		net->setGameEnd(); //setGameEnd so that need not to wait for next INIT message
		try {
			sock.cancel();
			send(G::GenByeMsg());
		} catch(std::exception& e) {
			logger->error("Failed to cancel sock: {}", e.what());
		}
		toReset();
	} else {
		logger->error("sceneEnd2StartTimer interrupted as: {}", e.message());
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::kaTimerHandle(const boost::system::error_code& e) {
	if (e && (e == boost::asio::error::operation_aborted)) {
		logger->warn("Katimer cancelled");
	} else if(!e) {
		if (send(G::GenKAMsg())) {
			kaTimer.expires_from_now(boost::posix_time::seconds(KATimeout));
			kaTimer.async_wait(boost::bind(&asiotenhoufsm::kaTimerHandle, this->shared_from_this(),
						boost::asio::placeholders::error));
		}
	} else {
		logger->error("Katimer interrupted as: {}", e.message());
//		reset();
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

		S rc = net->processMsg(subMsg);

		if (send(rc)) {
			lastIndex = index + 1;
			index = msg.find(">", lastIndex);
		} else {
			break;
		}
	}
}


template<class NetType>
bool asiotenhoufsm<NetType>::start() {
	resetState = NotReset;

	if (isPrivate) {
		int seconds = rand() % 10;
		sleep(seconds);
	}
	try {
		sock.open(tcp::v4());
		sock.connect(serverP);
		logger->info("Connected to {}, {}: {}", serverP.address().to_string(), serverP.port(), net->getName());
//		send(G::GenHeloMsg(name));
	} catch (std::exception& e) {
		logger->error("Failed to start connection: {}", e.what());
		restartCount ++;
		if (restartCount >= ReConnTrial) {
			logger->error("Broken network");
			throw e;
		} else {
			int sleepTime = rand() % ReConnThreshold;
			sleep(sleepTime);
			logger->warn("To restart");
			toReset();
		}

		return false;
	}
//	sock.open(tcp::v4());
//	sock.connect(serverP);
//	logger->info("Connected to {}, {}", serverP.address().to_string(), serverP.port());


	if (send(G::GenHeloMsg(name))) { //send treat exception itself
		logger->info("{} sent helo ", net->getName());
		if (!isPrivate) {
			RegRcv(heloState);
		} else {
			RegRcv(heloLobbyState);
		}
		kaTimer.expires_from_now(boost::posix_time::seconds(KATimeout));
		kaTimer.async_wait(boost::bind(&asiotenhoufsm::kaTimerHandle, this->shared_from_this(),
						boost::asio::placeholders::error));
		restartCount = 0;

		return true;
	} else {
		return false;
	}
}

template<class NetType>
void asiotenhoufsm<NetType>::reset() {
	sock.cancel();
	gameEnd2NextTimer.cancel();
	sceneEnd2StartTimer.cancel();
	kaTimer.cancel();
	reConnTimer.cancel();

	authenticated = false;
	gameBegin = false;
	lnCount = 0;

	sock.close();
	net->reset();

	start();
}

#endif /* INCLUDE_TENHOUCLIENT_ASIOTENHOUFSM_HPP_ */
