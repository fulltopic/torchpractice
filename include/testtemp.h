#include <type_traits>
namespace details {
	// 2 return value types
    template <typename Base> std::true_type is_base_of_test_func(const volatile Base*);
    template <typename Base> std::false_type is_base_of_test_func(const volatile void*);
    // Decide if Derived* could be deducted into Base type.
    // Yes --> is_base_of_test_func(const volatile Base*)
    // No --> invalid type
    //TODO: Why volatile?
    template <typename Base, typename Derived>
    using pre_is_base_of = decltype(is_base_of_test_func<Base>(std::declval<Derived*>()));

    // with <experimental/type_traits>:
    // template <typename Base, typename Derived>
    // using pre_is_base_of2 = std::experimental::detected_or_t<std::true_type, pre_is_base_of, Base, Derived>;


    // Why default type get true_type?
    template <typename Base, typename Derived, typename = void>
    struct pre_is_base_of2 : public std::true_type { };
    // note std::void_t is a C++17 feature

    // TODO: Why derive pre_is_base_of?
    // If it is OK, return pre_is_base_of<Base, Derived>
    // Else, return above type (typename = void)
    template <typename Base, typename Derived>
    struct pre_is_base_of2<Base, Derived, std::void_t<pre_is_base_of<Base, Derived>>> :
        public pre_is_base_of<Base, Derived> { };
}

template <typename Base, typename Derived>
struct is_base_of :
    public std::conditional_t<
        std::is_class<Base>::value && std::is_class<Derived>::value,
        details::pre_is_base_of2<Base, Derived>,
        std::false_type
    > { };
