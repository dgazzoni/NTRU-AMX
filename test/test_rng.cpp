#include "gtest/gtest.h"
#include "test.h"

extern "C" void nist_randombytes_init(unsigned char *entropy_input, unsigned char *personalization_string,
                                      int security_strength);
extern "C" int nist_randombytes(unsigned char *x, unsigned long long xlen);
extern "C" void opt_randombytes_init(unsigned char *entropy_input, unsigned char *personalization_string,
                                     int security_strength);
extern "C" int opt_randombytes(unsigned char *x, unsigned long long xlen);

// https://stackoverflow.com/a/48733150/523079
template <typename T>
class rng : public ::testing::Test {};

using test_types = ::testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 15>,
                                    std::integral_constant<std::size_t, 16>, std::integral_constant<std::size_t, 17>,
                                    std::integral_constant<std::size_t, 31>, std::integral_constant<std::size_t, 32>,
                                    std::integral_constant<std::size_t, 33>, std::integral_constant<std::size_t, 63>,
                                    std::integral_constant<std::size_t, 64>, std::integral_constant<std::size_t, 65>,
                                    std::integral_constant<std::size_t, 127>, std::integral_constant<std::size_t, 128>,
                                    std::integral_constant<std::size_t, 129>, std::integral_constant<std::size_t, 191>,
                                    std::integral_constant<std::size_t, 192>, std::integral_constant<std::size_t, 193>,
                                    std::integral_constant<std::size_t, 1048576> >;

class test_type_names {
   public:
    template <typename T>
    static std::string GetName(int) {
        return std::to_string(T::value);
    }
};

TYPED_TEST_SUITE(rng, test_types, test_type_names);

TYPED_TEST(rng, ref_matches_opt_1call) {
    static constexpr std::size_t len = TypeParam::value;
    unsigned char entropy_input[48] = {0};
    unsigned char xref[len], xopt[len];

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    nist_randombytes_init(entropy_input, NULL, 256);
    opt_randombytes_init(entropy_input, NULL, 256);

    nist_randombytes(xref, len);
    opt_randombytes(xopt, len);

    ASSERT_TRUE(ArraysMatch(xref, xopt));
}

TYPED_TEST(rng, ref_matches_opt_2calls) {
    static constexpr std::size_t len = TypeParam::value;
    unsigned char entropy_input[48] = {0};
    unsigned char xref[2 * len], xopt[2 * len];

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    nist_randombytes_init(entropy_input, NULL, 256);
    opt_randombytes_init(entropy_input, NULL, 256);

    nist_randombytes(xref, len);
    nist_randombytes(&xref[len], len);
    opt_randombytes(xopt, len);
    opt_randombytes(&xopt[len], len);

    ASSERT_TRUE(ArraysMatch(xref, xopt));
}
