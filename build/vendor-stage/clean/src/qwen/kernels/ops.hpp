#ifndef GUFO_MODELS_QWEN_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "qwen/kernels/ops/attention.hpp"
#include "qwen/kernels/ops/gemm.hpp"
#include "qwen/kernels/ops/norm_residual.hpp"
#include "qwen/kernels/ops/ssm.hpp"
#include "qwen/kernels/ops/swiglu.hpp"
#include "qwen/kernels/ops/token.hpp"

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HPP_
