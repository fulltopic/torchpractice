/*
 * selfserver.cpp
 *
 *  Created on: Oct 28, 2020
 *      Author: zf
 */

#include "selfserver/selfserver.h"
#include <boost/bind/bind.hpp>
#include <boost/asio.hpp>

#include "utils/logger.h"

namespace {
	auto logger = Logger::GetLogger();
}
//#include "selfserver/clientconn.h"

SelfServer::SelfServer(int iPort, boost::asio::io_context& service)
	: roomSeq(0),
	  port(iPort),
	  curClientIndex(0),
	  ioService(service),
	  acceptor(service,	boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
	  pollTimer(service)
{
	std::cout << "Server constructed " << std::endl;
	logger->info("Server constructed ");
	curRoom = Room::Create(roomSeq);
	rooms[roomSeq] = curRoom;
	std::cout << "End of server construction" << std::endl;
	logger->info("End of server construction");
}

void SelfServer::start() {
	logger->debug("To start pollTimer");
	pollTimer.expires_from_now(boost::posix_time::seconds((int)CleanUpTimeout));
	pollTimer.async_wait(boost::bind(&SelfServer::handleRoomPoll, this->shared_from_this(),
							boost::asio::placeholders::error));
	logger->debug("pollTimer started ");

	accept();
}

std::pair<std::shared_ptr<Room>, int> SelfServer::getRoomIndex() {
	std::unique_lock<std::mutex> lock(roomMutex);
	int index = -1;

	if (curClientIndex >= 4) {
		roomSeq ++;
		curRoom = Room::Create(roomSeq);
		logger->info("Room {} created", roomSeq);
		rooms[roomSeq] = curRoom;
		curClientIndex = 0;
//		index = 0;
	}
//	} else {
//		index = curClientIndex;
//		curClientIndex  ++;
//	}
	index = curClientIndex;
	curClientIndex ++;

	return std::make_pair(rooms[roomSeq], index); //How if return curRoom directly?
}

void SelfServer::accept() {
	logger->debug("accept");
//	auto client = ClientConn::Create
	auto roomIndex = getRoomIndex();

	std::shared_ptr<ClientConn> client = ClientConn::Create(std::get<1>(roomIndex), ioService, std::get<0>(roomIndex));
	logger->debug("client created {}:{}", std::get<0>(roomIndex)->seq, std::get<1>(roomIndex));

	acceptor.async_accept(client->getSocket(),
	        boost::bind(&SelfServer::handleAccept, shared_from_this(), client, //TODO: Why not shared_from_this? shared_from_this returns weak_ptr
	          boost::asio::placeholders::error));
	logger->debug("Ready to accept");
}


//TODO: add the client into room
void SelfServer::handleAccept(std::shared_ptr<ClientConn> client, const boost::system::error_code& error) {
	if (!error) {
		client->start();
		accept();
	} else {
		logger->error("===================================================================> \n Failed to accept in server: {}", error.message());
	}
}

void SelfServer::handleRoomPoll(const boost::system::error_code& error) {
	logger->warn("handleRoomPoll");
	if (!error) {
		for (auto ite = rooms.begin(); ite != rooms.end(); ) {
			if (!ite->second->isWorking()) {
				logger->warn("To remove room {}", ite->second->seq);
				ite = rooms.erase(ite);
			} else {
				ite ++;
			}
		}

		pollTimer.expires_from_now(boost::posix_time::seconds((int)CleanUpTimeout));
		pollTimer.async_wait(boost::bind(&SelfServer::handleRoomPoll, this->shared_from_this(),
								boost::asio::placeholders::error));
	}
}
