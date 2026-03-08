/*
 * dsp_pipeline.c
 *
 *  Created on: Mar 7, 2026
 *      Author: parth
 */

#include "dsp_pipeline.h"
#include "arm_math.h"
#include "mel_constants.h"
#include <math.h>

// --- GLOBAL BUFFERS ---
// The raw input (simulating your microphone)
float32_t full_audio[AUDIO_LENGTH];

// The final output image (49 * 40 = 1960 floats) - This goes straight to the CNN
float32_t cnn_input_tensor[NUM_FRAMES * NUM_MEL_BINS];

// Working buffers (re-used for every frame to save RAM)
float32_t audio_frame[FRAME_LENGTH];
float32_t fft_buffer[1024];
float32_t fft_output[1024];
float32_t magnitudes[513];
float32_t mel_energies[NUM_MEL_BINS];

arm_rfft_fast_instance_f32 fft_handler;
uint8_t fft_handler_initialized = 0;


void Init_Audio_Pipeline_Full(void) {
	if (fft_handler_initialized) {
		return;
	}
	fft_handler_initialized = 1;

	arm_status fft_status = arm_rfft_fast_init_f32(&fft_handler, 1024);
	if (fft_status != ARM_MATH_SUCCESS) {
		Error_Handler();
	}
}

void Process_Full_Audio(void) {

    // LOOP: Slide the 640-sample window across the 16,000-sample audio
    for (int frame = 0; frame < NUM_FRAMES; frame++) {

        // Calculate where this frame starts in the main audio buffer
        int start_idx = frame * FRAME_STEP;

        // 1. Copy the 640 samples for this specific frame
        for (int i = 0; i < FRAME_LENGTH; i++) {
            audio_frame[i] = full_audio[start_idx + i];
        }

        // 2. Apply Hann Window
        arm_mult_f32(audio_frame, HANN_WINDOW, audio_frame, FRAME_LENGTH);

        // 3. Pad to 1024 for the FFT
        for(int i = 0; i < FRAME_LENGTH; i++) fft_buffer[i] = audio_frame[i];
        for(int i = FRAME_LENGTH; i < 1024; i++) fft_buffer[i] = 0.0f;

        // 4. Execute FFT
        arm_rfft_fast_f32(&fft_handler, fft_buffer, fft_output, 0);

        // 5. Calculate Magnitudes (Unpack CMSIS-DSP format)
        magnitudes[0] = fabsf(fft_output[0]);
        arm_cmplx_mag_f32(&fft_output[2], &magnitudes[1], 511);
        magnitudes[512] = fabsf(fft_output[1]);

        // 6. Mel Matrix Multiplication
        arm_matrix_instance_f32 mat_A;
        arm_matrix_instance_f32 mat_B;
        arm_matrix_instance_f32 mat_C;

        arm_mat_init_f32(&mat_A, 1, 513, magnitudes);
        arm_mat_init_f32(&mat_B, 513, NUM_MEL_BINS, (float32_t*)MEL_MATRIX);
        arm_mat_init_f32(&mat_C, 1, NUM_MEL_BINS, mel_energies);

        arm_mat_mult_f32(&mat_A, &mat_B, &mat_C);

        // 7. Logarithm & Store in the final CNN Tensor
        for(int i = 0; i < NUM_MEL_BINS; i++) {
            // We calculate the flat index: (current_row * columns) + current_column
            int flat_index = (frame * NUM_MEL_BINS) + i;

            cnn_input_tensor[flat_index] = logf(mel_energies[i] + 1e-6f);
        }
    }

    // At this point, cnn_input_tensor contains the exact 2D image required by X-CUBE-AI!
}
