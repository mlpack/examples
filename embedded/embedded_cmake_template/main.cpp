/**
 * This is a super simple example using random forest. The idea is show how we
 * can cross compile this binary and use it on an embedded Linux device.
 *
 * It is up to the user to built something interesting out of this example, the
 * following is just a starting point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Omar Shrit
 */

#include <mlpack.hpp>

using namespace mlpack;

int main(int argc, char** argv)
{
  Log::Info << "Welcome to an mlpack crosscompiled binary." << std::endl;
}
