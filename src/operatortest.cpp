#include <iostream>
#include <string>

using namespace std;

namespace {
template<typename Target>
class Test {
private:
	Target* dataPtr;

	template<typename FromTarget>
	friend class Test;
public:
	Test(Target* t): dataPtr(t) {
		std::cout << "constructor" << std::endl;
	}

//	void setTarget(Target* t) {
//		dataPtr = t;
//	}

	template<typename FromTarget>
	Test& operator=(Test<FromTarget>& other) {
		std::cout << "type = operator " << endl;
		static_assert(std::is_convertible<FromTarget*, Target*>::value, "Not convertible");
		Test tmp = other; //which copy constructor invoked?
		swap(tmp.dataPtr, other.dataPtr);

		return *this;
	}

	Test& operator=(Test& other) {
		std::cout << "= operator" << endl;
		return operator=<Target>(other);
	}

//	Test(Test& other) = delete;

	void printData() {
		std::cout << *dataPtr << std::endl;
	}
};
}

void opTest() {
	int* data = new int(8);
	Test<int> testData(data);

	int* copyData = new int(9);
	Test<int> copied(copyData);

	copied = testData;

	copied.printData();
}


int main() {
	opTest();
}
