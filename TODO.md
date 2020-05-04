* boost.intrusive_ptr
* Memory management
* The framework that enables the fluent NN coding
* The RNN example
* My computing graph
* customized dataset
* How net backward knows the structure
* std::dynamic_pointer_cast<>
* std::move
* dropout
* SFINAE: Substitution failure is not an error
* std::is_base_of
	[void_t](https://www.fluentcpp.com/2017/06/02/write-template-metaprogramming-expressively/),
	[void_t stack_overflow](https://stackoverflow.com/questions/27687389/how-does-void-t-work),
	[test](https://gist.github.com/jefftrull/ff6083e2e92fdabb62f6),
	[paper](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf)
* RAII: Resource Acquisition Is Initialization	

* RNN param initialization
*torch/csrc/api/src/nn/modules/rnn.cpp*:
RNNImplBase<Derived>::RNNImplBase -->
void RNNImplBase<Derived>::reset() -->
void RNNImplBase<Derived>::reset_parameters()

* Some torch functions defined in nn/init.cpp
