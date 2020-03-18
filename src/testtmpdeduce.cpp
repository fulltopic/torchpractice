#include "templatededuce.h"
#include <vector>

using namespace c10::detail;

void testPrim() {
	auto r = c10::detail::assert_is_valid_input_type<int64_t, true>();
}

void testOption() {
	auto r = c10::detail::assert_is_valid_input_type<c10::optional<std::string>, false>();

	auto r1 = assert_is_valid_input_type<c10::optional<std::vector<int64_t> >, false>();

	//Error, no int type supported
//	auto r2 = assert_is_valid_input_type<c10::optional<std::vector<int> >, false>();
}

void testVector() {
	auto r = assert_is_valid_input_type<std::vector<int64_t>, false>();

	//Error, no scalar vector
//	auto r1 = assert_is_valid_input_type<std::vector<at::Scalar>, false>();
}

void testMap() {
	auto r = assert_is_valid_input_type<std::unordered_map<std::string, std::string>, true>();

//	auto r1 = assert_is_valid_input_type<std::unordered_map<std::string, std::string>, false>();

//	auto r2 = assert_is_valid_input_type<std::unordered_map<char, std::string>, true>();
}



int main() {
	testPrim();
	testOption();
	testMap();

	return 0;
}
