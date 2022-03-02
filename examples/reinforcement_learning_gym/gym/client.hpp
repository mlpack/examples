/**
 * @file client.hpp
 * @author Marcus Edel
 *
 * Miscellaneous client routines.
 */
#ifndef GYM_CLIENT_HPP
#define GYM_CLIENT_HPP

#include <string>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/copy.hpp>

namespace gym {

using boost::asio::ip::tcp;
using boost::asio::deadline_timer;
using boost::lambda::_1;
using boost::lambda::var;
using boost::lambda::bind;

/**
 * Implementation of the Client.
 */
class Client
{
 public:
  static void async_read_handler(const boost::system::error_code& err,
                                 boost::system::error_code* err_out,
                                 std::size_t bytes_transferred,
                                 std::size_t* bytes_out)
  {
    *err_out = err;
    *bytes_out = bytes_transferred;
  }

  /**
   * Create the Client object using the given host and port.
   *
   * @param host The hostname to connect.
   * @param port The port used for the connection.
   */
  Client() :
      s(io_service),
      deadline(io_service),
      compressionLevel(0)
  {
    deadline.expires_at(boost::posix_time::pos_infin);

    // Start the persistent actor that checks for deadline expiry.
    check_deadline();
  }

  ~Client()
  {
    // The socket is closed so that any outstanding asynchronous operations are
    // cancelled. This allows the blocked connect(), read_line() or write_line()
    // functions to return.
    boost::system::error_code ignored_ec;
    s.close(ignored_ec);
  }

  void connect(const std::string& host, const std::string& port)
  {
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(tcp::v4(), host, port);
    tcp::resolver::iterator iterator = resolver.resolve(query);

    // Set a deadline for the asynchronous operation.
    deadline.expires_from_now(boost::posix_time::seconds(10));

    // Set up the variable that receives the result of the asynchronous
    // operation.
    boost::system::error_code ec = boost::asio::error::would_block;

    // Start the asynchronous operation.
    boost::asio::async_connect(s, iterator, var(ec) = _1);

    // Block until the asynchronous operation has completed.
    do io_service.run_one(); while (ec == boost::asio::error::would_block);

    // Determine whether a connection was successfully established.
    if (ec || !s.is_open())
    {
      throw boost::system::system_error(
          ec ? ec : boost::asio::error::operation_aborted);
    }
  }

  /**
   * Receive a message using the currently open socket.
   *
   * @param data The received data.
   */
  void receive(std::string& data)
  {
    // Set a deadline for the asynchronous operation.
    deadline.expires_from_now(boost::posix_time::seconds(10));

    // Set up the variable that receives the result of the asynchronous
    // operation.
    boost::system::error_code ec = boost::asio::error::would_block;

    boost::asio::streambuf response;
    size_t reply_length;

    boost::asio::async_read_until(s, response, "\r\n\r\n",
        boost::bind(async_read_handler, boost::asio::placeholders::error, &ec,
        boost::asio::placeholders::bytes_transferred, &reply_length));

    // Block until the asynchronous operation has completed.
    do io_service.run_one(); while (ec == boost::asio::error::would_block);

    if (ec)
    {
      throw boost::system::system_error(ec);
    }

    data = std::string(
        boost::asio::buffers_begin(response.data()),
        boost::asio::buffers_begin(response.data()) + reply_length);

    if (compressionLevel > 0)
    {
      boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
      in.push( boost::iostreams::zlib_decompressor());
      in.push(boost::iostreams::array_source(data.data(), data.size()));

      std::stringstream ss;
      boost::iostreams::copy(in, ss);
      data = ss.str();
    }
  }

  /**
   * Send a message using the currently open socket.
   *
   * @param data The data to be send.
   */
  void send(const std::string& data)
  {
    // Set a deadline for the asynchronous operation.
    deadline.expires_from_now(boost::posix_time::seconds(10));

    // Set up the variable that receives the result of the asynchronous
    // operation.
    boost::system::error_code ec = boost::asio::error::would_block;

    std::string d = data  + "\r\n";

    boost::asio::async_write(s, boost::asio::buffer(d),
        boost::asio::transfer_exactly(d.size()), var(ec) = _1);

    // Block until the asynchronous operation has completed.
    do io_service.run_one(); while (ec == boost::asio::error::would_block);

    if (ec)
    {
      throw boost::system::system_error(ec);
    }
  }

  /*
   * The compression level in range [0, 9] where 0 means no compression used for
   * receiving data.
   *
   * @param compression The compression level.
   */
  void compression(const size_t compression)
  {
    compressionLevel = compression;
  }

 private:
  void check_deadline()
  {
    // Check whether the deadline has passed. We compare the deadline against
    // the current time since a new asynchronous operation may have moved the
    // deadline before this actor had a chance to run.
    if (deadline.expires_at() <= deadline_timer::traits_type::now())
    {
      // There is no longer an active deadline. The expiry is set to positive
      // infinity so that the actor takes no action until a new deadline is set.
      deadline.expires_at(boost::posix_time::pos_infin);
    }

    // Put the actor back to sleep.
    deadline.async_wait(bind(&Client::check_deadline, this));
  }

  //! Locally stored io service.
  boost::asio::io_service io_service;

  //! Locally stored socket object.
  tcp::socket s;

  //! Object to control connection timeouts.
  deadline_timer deadline;

  //! Locally-stored compression parameter.
  size_t compressionLevel;
}; // class Client

} // namespace gym

#endif
