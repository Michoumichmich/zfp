#pragma once
//#define IMPLICIT_MEMORY_COPY
#define VERBOSE_SYCL
#define SYCL_ZFP_RATE_PRINT 1

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
size_t sycl_compress(zfp_stream *stream, const zfp_field *field, int variable_rate);
void sycl_decompress(zfp_stream *stream, zfp_field *field);
#ifdef __cplusplus
}
#endif
