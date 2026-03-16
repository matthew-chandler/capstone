/*
 * cnn.c
 *
 *  Created on: Mar 7, 2026
 *      Author: parth
 */
#include "cnn.h"
#include "dsp_pipeline.h"
#include <math.h>
#include "arm_math.h"
#include <stdio.h>
// xcubeai headers
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

// 1. The main handle for the neural network
ai_handle network = AI_HANDLE_NULL;

// 2. The input and output memory buffers
ai_buffer *ai_input;
ai_buffer *ai_output;

// 3. The array to hold the final probabilities [Yes, No, Unknown]
volatile float32_t ai_out_data[3];

// allocate data for calculations for the cnn
AI_ALIGNED(32) static ai_u8 activations_buffer[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

// (Make sure your cnn_input_tensor from earlier is also declared here)
extern float32_t cnn_input_tensor[NUM_FRAMES * NUM_MEL_BINS];
ai_error err;

void AI_Init(void) {
    ai_error err;

    // 1. Put our RAM pointer into an array format (which the ST helper function requires)
    const ai_handle act_addr[] = { activations_buffer };

    // 2. The ST Magic Helper Function
    // This handles network creation, internal parameter binding, and initialization all at once!
    // Passing NULL for the weights tells it to just use the default weights stored in Flash.
    err = ai_network_create_and_init(&network, act_addr, NULL);

    if (err.type != AI_ERROR_NONE) {
        printf("AI Init FAILED! Error Type: 0x%02X, Error Code: 0x%02X\r\n", err.type, err.code);
        __NOP();
        while(1);
    }

    // 3. Fetch the internal buffer structures dynamically
    ai_input = ai_network_inputs_get(network, NULL);
    ai_output = ai_network_outputs_get(network, NULL);

    printf("✅ AI Model Initialized!\r\n");
}


void Run_AI_Inference(void) {
    ai_i32 batch;

    // 1. Point the AI Input buffer to your DSP output array
    ai_input[0].data = AI_HANDLE_PTR(cnn_input_tensor);

    // 2. Point the AI Output buffer to our empty result array
    ai_output[0].data = AI_HANDLE_PTR(ai_out_data);

    // 3. Run the Neural Network!
    batch = ai_network_run(network, ai_input, ai_output); // Pass pointers directly

    if (batch != 1) {
//        printf("Error: AI run failed.\r\n");
        return;
    }

    // 4. Decode the results
    float32_t conf_yes = ai_out_data[0] * 100.0f;
    float32_t conf_no  = ai_out_data[1] * 100.0f;
    float32_t conf_unk = ai_out_data[2] * 100.0f;

    // 5. Print the prediction
//    printf("--- INFERENCE RESULT ---\r\n");
//    printf("YES:     %5.1f%%\r\n", conf_yes);
//    printf("NO:      %5.1f%%\r\n", conf_no);
//    printf("UNKNOWN: %5.1f%%\r\n\n", conf_unk);
}
