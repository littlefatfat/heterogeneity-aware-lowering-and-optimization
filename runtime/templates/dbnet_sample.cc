#include "ODLA/odla.h"

__attribute__((annotate("halo_build_computation")))
odla_computation static build_model();

static odla_context ctx;
static odla_computation comp;
static odla_device dev;

static odla_uint32 nr_inputs;
static odla_uint32 nr_outputs;
static odla_value first_input;
static odla_value first_output;

static odla_value input0_value;
static odla_uint32 input0_idx = 0;
static odla_value out1_value;
static odla_uint32 output1_idx = 0;

static bool is_dynamic_shape = true;

// min max opt
odla_value_shape input0_min_shape{.size = 4, .dims = {1, 3, 1, 1}};
odla_value_shape input0_max_shape{.size = 4, .dims = {1, 3, 1000, 2000}};
odla_value_shape input0_opt_shape{.size = 4, .dims = {1, 3, 960, 1280}};

// real shape
odla_value_shape input0_real_shape{.size = 4, .dims = {1, 3, 960, 1280}};
odla_value_shape out1_real_shape{.size = 4, .dims = {1, 1, 960, 1280}};

static int init_model() {
  if (comp == NULL) {
    comp = build_model();
  }

  // get value from idx
  odla_GetArgFromComputationByIdx(comp, input0_idx, &input0_value);
  odla_GetOutputFromComputationByIdx(comp, output1_idx, &out1_value);

  // get number of args/outputs
  odla_GetNumOfArgsFromComputation(comp, &nr_inputs);
  odla_GetNumOfOutputsFromComputation(comp, &nr_outputs);

  // set computation item
  odla_SetComputationItem(comp, ODLA_DYNAMIC_SHAPE,
                          (odla_item_value)&is_dynamic_shape);
  odla_SetComputationItem(comp, ODLA_DYNAMIC_SHAPE_INPUTS,
                          (odla_item_value)&input0_value);
  odla_SetComputationItem(comp, ODLA_DYNAMIC_SHAPE_OUTPUTS,
                          (odla_item_value)&out1_value);

  // set shapeinfo
  odla_SetValueShapeInfo(input0_value, ODLA_MIN_SHAPE, input0_min_shape);
  odla_SetValueShapeInfo(input0_value, ODLA_MAX_SHAPE, input0_max_shape);
  odla_SetValueShapeInfo(input0_value, ODLA_OPT_SHAPE, input0_opt_shape);

  return 0;
}

static int init_context() {
  if (ctx == NULL) {
    odla_CreateContext(&ctx);
  }

  // set runtime shape
  odla_SetRuntimeShape(ctx, input0_value, input0_real_shape);
  odla_SetRuntimeShape(ctx, out1_value, out1_real_shape);
}

#ifdef __cplusplus
extern "C" {
#endif

int dbnet(const void* const inputs[], void* outputs[]) {
  init_model();
  init_context();

  // bind args/outputs
  odla_GetArgFromComputationByIdx(comp, 0, &first_input);
  odla_BindToArgument(first_input, inputs, ctx);
  odla_GetOutputFromComputationByIdx(comp, 0, &first_output);
  odla_BindToOutput(first_output, outputs, ctx);

  for (int idx = 1; (nr_inputs > 1) && (idx < nr_inputs); ++idx) {
    odla_value val;
    odla_GetArgFromComputationByIdx(comp, idx, &val);
    odla_BindToArgument(val, inputs[idx], ctx);
  }
  for (int idx = 1; (nr_outputs > 1) && (idx < nr_outputs); ++idx) {
    odla_value val;
    odla_GetOutputFromComputationByIdx(comp, idx, &val);
    odla_BindToOutput(val, outputs[idx], ctx);
  }
  odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, dev);
}

#ifdef __cplusplus
}
#endif
