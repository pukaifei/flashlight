#include <codecvt>
#include <locale>
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"

const double msPerFrame = 20;

struct AlignedWord {
  std::string word;
  double startTimeMs;
  double endTimeMs;
};

void remapUTFWord(std::u16string& input, int replabel) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;

  // dedup labels
  auto it = std::unique(input.begin(), input.end());
  input.resize(std::distance(input.begin(), it));

  // map of replabels
  std::unordered_map<char16_t, int64_t> replabelMap;
  for (int64_t i = 1; i <= replabel; ++i) {
    replabelMap[utf16conv.from_bytes(std::to_string(i))[0]] = i;
  }

  std::u16string output;
  output += input[0];
  for (size_t i = 1; i < input.size(); ++i) {
    auto repCount = replabelMap.find(input[i]);
    if (repCount != replabelMap.end()) {
      for (auto j = 0; j < repCount->second; j++) {
        output += input[i - 1];
      }
    } else {
      output += input[i];
    }
  }

  std::swap(input, output);
}

// Converts aligned letter sequence into vector of AlignedWords
std::vector<AlignedWord> getAlignedWords(
    const std::vector<std::string>& ltrs,
    int replabel) {
  // correctly handling character strings for languages needing utf-8 / 16
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
  std::vector<std::u16string> utf16Ltrs;
  for (auto& l : ltrs) {
    utf16Ltrs.emplace_back(utf16conv.from_bytes(l));
  }

  std::vector<AlignedWord> alignedWords;
  std::u16string currWord;
  int stFrame = 0;
  bool inWord = false;
  int silStart = 0;
  for (int i = 0; i < utf16Ltrs.size(); i++) {
    if (utf16Ltrs[i] == utf16conv.from_bytes(w2l::kSilToken)) {
      if (inWord) { // found end of word, insert
        auto endTimeMs = msPerFrame * i;
        auto startTimeMs = msPerFrame * stFrame;
        remapUTFWord(currWord, replabel);
        alignedWords.emplace_back(
            AlignedWord{utf16conv.to_bytes(currWord), startTimeMs, endTimeMs});
        inWord = false;
        silStart = i;
      }
    } else if (!inWord) { // starting new word
      stFrame = i;
      currWord = utf16Ltrs[i];
      inWord = true;
      // Also insert silence
      if (silStart < i - 1) {
        alignedWords.emplace_back(
            AlignedWord{"$", msPerFrame * silStart, msPerFrame * (i)});
      }
    } else { // continue in same word
      currWord += utf16Ltrs[i];
    }
  }

  // Take care of trailing silence or trailing word
  auto endTimeMs = msPerFrame * (utf16Ltrs.size());
  if (inWord) {
    // we may encounter trailing word only
    // if we train without -surround='|'
    currWord += utf16Ltrs[utf16Ltrs.size() - 1];
    remapUTFWord(currWord, replabel);
    alignedWords.emplace_back(AlignedWord{
        utf16conv.to_bytes(currWord), msPerFrame * stFrame, endTimeMs});
  } else {
    alignedWords.emplace_back(
        AlignedWord{"$", msPerFrame * silStart, endTimeMs});
  }
  return alignedWords;
}

// Utility function which converts the aligned words
// into CTM format which is compatible with AML's alignment output.
// this format can be used in several workflows later, including
// segmentation workflow.
std::string getCTMFormat(std::vector<AlignedWord> alignedWords) {
  std::stringstream ctmString;
  int i = 0;
  for (auto& alignedWord : alignedWords) {
    double stTimeSec = alignedWord.startTimeMs / 1000.0;
    double durationSec =
        (alignedWord.endTimeMs - alignedWord.startTimeMs) / 1000.0;
    ctmString << "ID A " << stTimeSec << " " << durationSec << " "
              << alignedWord.word;
    if (i < alignedWords.size() - 1) {
      ctmString << "\\n";
    }
    i++;
  }
  return ctmString.str();
}
