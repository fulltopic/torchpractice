/*
 * tenhournnstate.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: zf
 */


//#include "tenhouclient/tenhournnstate.h"
//
//using namespace std;
//using namespace torch;
//
////Deprecated, hidden state in proxy
//
//GRUState::GRUState(int ww, int hh): BaseState(ww, hh), step(0) {}
//
//GRUState::~GRUState() {}
//
//vector<Tensor> GRUState::getState(int indType) {
//	auto output = BaseState::getState(indType);
//	return {output, hiddenState};
//}
//
////TODO: updateState at first to make hiddenState valid
////TODO: Seemed state should not be bound to board state?
//void GRUState::updateState(vector<Tensor> newStates) {
//	hiddenState = newStates[0];
//	step ++;
//}
//
////TODO: hiddenState to be reseted
//void GRUState::reset() {
//	BaseState::reset();
//
//	step = 0;
//}

