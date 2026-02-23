#include "test_dsp.h"
#include "arm_math.h"
#include "constants.h"
#include <math.h>

float32_t audio_frame[640];
float32_t fft_buffer[1024];
float32_t fft_output[1024];
float32_t magnitudes[513];
float32_t mel_energies[40];
float32_t log_mel[40];

arm_rfft_fast_instance_f32 fft_handler;

// Call this ONCE in your main() before the while(1) loop
void Init_Audio_Pipeline(void) {
    arm_rfft_fast_init_f32(&fft_handler, 1024);
}

// Call this to test the pipeline
void Test_DSP_Pipeline(void) {
    // 1. Generate the exact same dummy signal (Linear ramp -1.0 to 1.0)
    for (int i = 0; i < 640; i++) {
        audio_frame[i] = -1.0f + (2.0f * (float32_t)i / 639.0f);
    }

    // 2. Apply Hann Window
    arm_mult_f32(audio_frame, HANN_WINDOW, audio_frame, 640);

    // 3. Pad to 1024
    for(int i = 0; i < 640; i++) fft_buffer[i] = audio_frame[i];
    for(int i = 640; i < 1024; i++) fft_buffer[i] = 0.0f;

    // 4. Execute FFT
    arm_rfft_fast_f32(&fft_handler, fft_buffer, fft_output, 0);

    // 5. Calculate Magnitudes (Unpacking the weird CMSIS-DSP format)
    // fft_output[0] = DC, fft_output[1] = Nyquist
    magnitudes[0] = fabsf(fft_output[0]);

    // Calculate complex magnitudes for bins 1 through 511
    arm_cmplx_mag_f32(&fft_output[2], &magnitudes[1], 511);

    magnitudes[512] = fabsf(fft_output[1]);

    // 6. Mel Matrix Multiplication
    // We are doing: magnitudes (1x513) * MEL_MATRIX (513x40) = mel_energies (1x40)
    arm_matrix_instance_f32 mat_A;
    arm_matrix_instance_f32 mat_B;
    arm_matrix_instance_f32 mat_C;

    arm_mat_init_f32(&mat_A, 1, 513, magnitudes);
    arm_mat_init_f32(&mat_B, 513, 40, (float32_t*)MEL_MATRIX);
    arm_mat_init_f32(&mat_C, 1, 40, mel_energies);

    arm_mat_mult_f32(&mat_A, &mat_B, &mat_C);

    // 7. Logarithm
    for(int i = 0; i < 40; i++) {
        log_mel[i] = logf(mel_energies[i] + 1e-6f);
    }
}
