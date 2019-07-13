#include <gtest/gtest.h>
#include <chrono>
#include <codecvt>
#include <fstream>
#include <locale>
#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "experimental/AlignUtils.h"
using namespace w2l;
namespace {} // namespace

TEST(AlignTest, getAlignedWords) {
  std::vector<std::string> ltrs = {"|", "|", "|", "h", "h", "e", "e", "l", "l",
                                   "l", "l", "1", "o", "o", "o", "|", "|", "w",
                                   "o", "o", "r", "l", "l", "d", "|"};

  auto words = getAlignedWords(ltrs, 2);
  ASSERT_EQ(words[0].word, "$");
  ASSERT_EQ(words[1].word, "hello");
  ASSERT_EQ(words[2].word, "$");
  ASSERT_EQ(words[3].word, "world");
  ASSERT_EQ(words[4].word, "$");
}

TEST(AlignTest, getAlignedWordSplits1) {
  std::vector<std::string> ltrs = {"|", "|", "|", "h", "h", "e", "e", "l", "l",
                                   "l", "l", "1", "1", "1", "o", "o", "o", "|",
                                   "|", "w", "o", "o", "r", "l", "l", "d", "|"};

  auto words = getAlignedWords(ltrs, 2);
  ASSERT_EQ(words[0].word, "$");
  ASSERT_EQ(words[1].word, "hello");
  ASSERT_EQ(words[2].word, "$");
  ASSERT_EQ(words[3].word, "world");
  ASSERT_EQ(words[4].word, "$");
}

TEST(AlignTest, getAlignedWordSplits2) {
  std::vector<std::string> ltrs = {"|", "|", "|", "h", "h", "e", "e", "l", "l",
                                   "l", "l", "2", "2", "2", "o", "o", "o", "|",
                                   "|", "w", "o", "o", "r", "l", "l", "d", "|"};

  auto words = getAlignedWords(ltrs, 2);
  ASSERT_EQ(words[0].word, "$");
  ASSERT_EQ(words[1].word, "helllo");
  ASSERT_EQ(words[2].word, "$");
  ASSERT_EQ(words[3].word, "world");
  ASSERT_EQ(words[4].word, "$");
}

TEST(AlignTest, trailingWord) {
  std::vector<std::string> ltrs = {"|", "|", "|", "h", "h", "e", "e", "l", "l",
                                   "l", "l", "2", "2", "2", "o", "o", "o", "|",
                                   "|", "w", "o", "o", "r", "l", "l", "d"};

  auto words = getAlignedWords(ltrs, 2);
  ASSERT_EQ(words[0].word, "$");
  ASSERT_EQ(words[1].word, "helllo");
  ASSERT_EQ(words[2].word, "$");
  ASSERT_EQ(words[3].word, "world");
}

TEST(AlignTest, noBeginningSilence) {
  std::vector<std::string> ltrs = {"h", "h", "e", "e", "l", "l", "l", "l",
                                   "2", "2", "2", "o", "o", "o", "|", "|",
                                   "w", "o", "o", "r", "l", "l", "d"};

  auto words = getAlignedWords(ltrs, 2);
  ASSERT_EQ(words[0].word, "helllo");
  ASSERT_EQ(words[1].word, "$");
  ASSERT_EQ(words[2].word, "world");
}
