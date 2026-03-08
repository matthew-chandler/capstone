/*
 * dsp_pipeline.h
 *
 *  Created on: Mar 7, 2026
 *      Author: parth
 */

#ifndef INC_DSP_PIPELINE_H_
#define INC_DSP_PIPELINE_H_

#include "main.h"

// --- PIPELINE CONSTANTS ---
#define AUDIO_LENGTH 16000
#define FRAME_LENGTH 640
#define FRAME_STEP   320
#define NUM_FRAMES   49
#define NUM_MEL_BINS 40

// --- GLOBAL FUNCTIONS ---
void Process_Full_Audio(void);
void Init_Audio_Pipeline_Full(void);


#endif /* INC_DSP_PIPELINE_H_ */
