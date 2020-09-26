/*
 * teststrand.cpp
 *
 *  Created on: Sep 21, 2020
 *      Author: zf
 */

#include <ctime>
#include <iostream>
#include <string>
#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/array.hpp>
#include <boost/date_time.hpp>

#include <boost/thread/thread.hpp>


using boost::asio::ip::tcp;
const std::string portStr = "33333";
const int portNum = 33333;

/************************************************* Server **************************************************/
static std::string make_daytime_string() {
	using namespace std;
	time_t now = time(0);
	return ctime(&now);
}

class tcp_connection
	: public boost::enable_shared_from_this<tcp_connection>
{
public:
	typedef boost::shared_ptr<tcp_connection> pointer;

	static pointer Create(boost::asio::io_context& io) {
		return pointer(new tcp_connection(io));
	}

	tcp::socket& socket() {
		return sock;
	}

	void start() {
		msg = make_daytime_string();

		boost::asio::async_write(sock, boost::asio::buffer(msg),
				boost::bind(&tcp_connection::handle_write, shared_from_this(),
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred)
		);
	}

	~tcp_connection() {
		std::cout << "End of a session " << std::endl;
	}
private:
	tcp::socket sock;
	std::string msg;
	boost::array<char, 4> rcvBuf;

	tcp_connection(boost::asio::io_context& io): sock(io) {}

	void handle_rcv(const boost::system::error_code &e, std::size_t len) {
		if (!e) {
			sleep(5);

			msg = make_daytime_string();

			boost::asio::async_write(sock, boost::asio::buffer(msg),
				boost::bind(&tcp_connection::handle_write, shared_from_this(),
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred)
			);
		} else {
			std::cout << "Rcv failure: " << e.message() << std::endl;
		}
	}

	void handle_write (const boost::system::error_code& e, size_t) {
//		msg = make_daytime_string();
//
		if (!e) {
			sock.async_receive(boost::asio::buffer(rcvBuf),
				boost::bind(&tcp_connection::handle_rcv, shared_from_this(),
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred)
			);
		} else {
			std::cout << "Send failure " << e.message() <<  std::endl;
		}
	}
};

class tcp_server {
private :
	tcp::acceptor acp;

	void start_accept() {
		tcp_connection::pointer newConn =
				tcp_connection::Create(static_cast<boost::asio::io_context&>(acp.get_executor().context()));

		acp.async_accept(newConn->socket(),
				boost::bind(&tcp_server::handleAccept, this, newConn, boost::asio::placeholders::error));
	}

	void handleAccept (tcp_connection::pointer newConn,
			const boost::system::error_code& error) {
		if (!error) {
			newConn->start();
		}

		start_accept();
	}

public:
	tcp_server (boost::asio::io_context& io):
		acp (io, tcp::endpoint(tcp::v4(), portNum)) {
		start_accept();
	}
};


static void asyncservice() {
	try {
		boost::asio::io_context io;
		tcp_server server(io);
		io.run();
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
}


/****************************************** Client ***************************************************/
const int timerSecond = 10;
const int timerKickTh = 10;
class tcpclient :
		public boost::enable_shared_from_this<tcpclient> {
private:
	tcp::resolver resolver;
	tcp::endpoint serverP;
	tcp::socket sock;
	boost::array<char, 128> rcvBuf;
	boost::asio::deadline_timer timer;

	boost::asio::deadline_timer sndCancelTimer;
	boost::asio::io_context::strand timerStrand;

	volatile int timerKick;

	void printLocalTime(std::string name) {
		auto localTime = boost::posix_time::second_clock::local_time();
		std::cout << name << ": " << localTime << std::endl;
	}

	void startTimer() {
		timer.expires_from_now(boost::posix_time::seconds(timerSecond));
		sndCancelTimer.expires_from_now(boost::posix_time::seconds(timerSecond));

		timer.async_wait(boost::bind(&tcpclient::handleTimer, this, boost::asio::placeholders::error));
//		timer.async_wait(timerStrand.wrap(boost::bind(&tcpclient::handleTimer, this, boost::asio::placeholders::error)));
		std::cout << "Timer expries at: " << timer.expires_at() << std::endl;

		sndCancelTimer.async_wait(boost::bind(&tcpclient::handleSndTimer, this, boost::asio::placeholders::error));
//		sndCancelTimer.async_wait(timerStrand.wrap(boost::bind(&tcpclient::handleSndTimer, this, boost::asio::placeholders::error)));
		std::cout << "sndCancelTimer expires at: " << sndCancelTimer.expires_at() << std::endl;
	}

	void handleTimer(const boost::system::error_code& error) {
		if (error) {
			std::cout << "handleTimer error: " << error.message() << std::endl;
			return;
		}

		printLocalTime("handleTimer");
		if (sndCancelTimer.cancel() > 0) {
			std::cout << "Send timer timeout, snd cancel timer canceled " << std::endl;
		} else {
			std::cout << "Too late to cancel sndCancelTimer" << std::endl;
		}

		boost::array<char, 1> sndBuf = {{0}};

		sock.async_send(
				boost::asio::buffer(sndBuf),
				boost::bind(&tcpclient::handleSnd, this, //TODO: lifetime of sndbuf
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred));
	}


	void handleSndTimer(const boost::system::error_code& error) {
		if (error) {
			std::cout << "handleSndTimer error: " << error.message() << std::endl;
			return;
		}

		printLocalTime("handleSndTimer");
		if (timer.cancel() > 0) {
			std::cout << "Send cancel timer timeout, timer cancel " << std::endl;
		} else {
			std::cout << "Too late to cancel timer " << std::endl;
		}
		timerKick ++;
		std::cout << "timerKick: " << timerKick << std::endl;
		if (timerKick >= timerKickTh) {
			return;
		}

		startTimer();
	}

	void handleSnd (const boost::system::error_code&, std::size_t) {
		std::cout << "Sent indicator " << std::endl;

		sock.async_receive(
				boost::asio::buffer(rcvBuf),
				boost::bind(&tcpclient::handleRcv, this,
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred())
		);
	}

	void handleRcv (const boost::system::error_code &err, std::size_t len) {
		if ((!err) || (err == boost::asio::error::message_size)) {
			std::cout << "The message " << std::string(rcvBuf.data(), len) << std::endl;

			timerKick ++;
			std::cout << "timerKick: " << timerKick << std::endl;
			if (timerKick >= timerKickTh) {
				return;
			}

			startTimer();
		} else {
			std::cout << "Error " << err << std::endl;
		}
	}

public:
	void start() {
		try {
			sock.open(tcp::v4());
			std::cout << "Open sock for v4" << std::endl;
//			boost::asio::connect (sock, serverP);
			sock.connect(serverP);
			std::cout << "Connect to server " << serverP.address() << std::endl;

			boost::array<char, 1> sndBuf = {{0}};
//			boost::asio::async_write(sock, boost::asio::buffer(sndBuf),
			sock.async_send(boost::asio::buffer(sndBuf),
					boost::bind(&tcpclient::handleSnd, this,
							boost::asio::placeholders::error,
							boost::asio::placeholders::bytes_transferred
							));
			std::cout << "Bind handle in start " << std::endl;
		} catch (std::exception &e) {
			std::cout << "Failed to start " << std::endl;
			std::cout << e.what() << std::endl;
		}
	}

	tcpclient(boost::asio::io_context& io)
		:resolver(io),
		 sock(io),
		 timer(io),
		 sndCancelTimer(io),
		 timerStrand(io),
		 timerKick(0)
	{
		serverP = *(resolver.resolve(tcp::v4(), "localhost", portStr).begin());
	}
};

boost::asio::io_context io;
boost::thread_group threadPool;

static void runService() {
	io.run();
}

static void asyncclientservice() {
	tcpclient client(io);
	client.start();

	for (int i = 0; i < 1; i ++) {
		threadPool.create_thread(runService);
	}

	threadPool.join_all();
//	io.run();
}

/********************************* Test others *******************************************/
static void printFunc(std::vector<int>& datas) {
	for (auto data: datas) {
		std::cout << data << std::endl;
		sleep(1);
	}
}

static void printFunc1 () {
	std::vector<int> datas {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	printFunc(datas);
}

static void printFunc2() {
	std::vector<int> datas {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
	printFunc(datas);
}

static void printTest() {
	threadPool.create_thread(printFunc1);
	threadPool.create_thread(printFunc2);

	threadPool.join_all();
}

int main(int argc, char** argv) {
//	int type = atoi(argv[1]);
//
//	if (type == 0) {
//		asyncservice();
//	} else {
//		asyncclientservice();
//	}

	printTest();
}



