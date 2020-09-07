/*
 * testhasfunction.cpp
 *
 *  Created on: Sep 4, 2020
 *      Author: zf
 */




#include <string>
#include <iostream>

template <typename T>
struct HasCreateHState {
	template<typename U, int (U::*)()> struct SFINEA {};
	template<typename U> static char Test(SFINEA<U, &U::createHState>*);
	template<typename U> static int Test(...);
	static const bool Has = (sizeof(Test<T>(0)) == sizeof(char));
};

template <typename T>
void hasFuncDefined(std::true_type) {
	std::cout << "defined " << std::endl;
}

template<typename T>
void hasFuncDefined(std::false_type) {
	std::cout << "not defined " << std::endl;
}

struct NotDefined {
	void test() {}
};

struct Defined {
	int createHState() {
		return 0;
	}
};

int main() {
	std::cout << HasCreateHState<Defined>::Has << std::endl;
	hasFuncDefined<Defined>(std::integral_constant<bool, HasCreateHState<Defined>::Has>());
	hasFuncDefined<NotDefined>(std::integral_constant<bool, HasCreateHState<NotDefined>::Has>());
}
