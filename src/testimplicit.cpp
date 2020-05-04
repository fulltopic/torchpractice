/*
 * testimplicit.cpp
 *
 *  Created on: Apr 8, 2020
 *      Author: zf
 */

#include <iostream>
#include <string>

using namespace std;

struct foo
{
  foo(int x) {};              // 1
  foo(char* s, int x = 0) {}; // 2
  foo(float f, int x) {};     // 3
  explicit foo(char x) {};    // 4
};

foo testReturnFoo() {
	return {1.0f, 5};
}

struct ObjOptions {
private:
	const int a;
	const int b;

public:
	ObjOptions(const int aa, const int bb): a(aa), b(bb)  {}
};

struct ObjDef {
	ObjDef(ObjOptions options) { cout << "obj of options " << endl; }
	ObjDef(const int aa, const int bb): ObjDef(ObjOptions(aa, bb)) { cout << "obj from raws " << endl; }
};


void test() {
	ObjDef a(1, 2);
}

ObjOptions testReturn() {
	return {1, 2};
}

int main() {
	test();
	testReturnFoo();
}
