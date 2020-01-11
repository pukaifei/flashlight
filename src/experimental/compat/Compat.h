#include "experimental/compat/TransformerLayer.h"

namespace w2l {

DECLARE_int64(decoder_layers);
DECLARE_double(decoder_dropout);
DECLARE_double(decoder_layerdrop);

void initCompat();

} // namespace w2l
