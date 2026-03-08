/**
  ******************************************************************************
  * @file    attempt1_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-07T15:40:32-0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ATTEMPT1_DATA_PARAMS_H
#define ATTEMPT1_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ATTEMPT1_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_attempt1_data_weights_params[1]))
*/

#define AI_ATTEMPT1_DATA_CONFIG               (NULL)


#define AI_ATTEMPT1_DATA_ACTIVATIONS_SIZES \
  { 119332, }
#define AI_ATTEMPT1_DATA_ACTIVATIONS_SIZE     (119332)
#define AI_ATTEMPT1_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ATTEMPT1_DATA_ACTIVATION_1_SIZE    (119332)



#define AI_ATTEMPT1_DATA_WEIGHTS_SIZES \
  { 1256972, }
#define AI_ATTEMPT1_DATA_WEIGHTS_SIZE         (1256972)
#define AI_ATTEMPT1_DATA_WEIGHTS_COUNT        (1)
#define AI_ATTEMPT1_DATA_WEIGHT_1_SIZE        (1256972)



#define AI_ATTEMPT1_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_attempt1_activations_table[1])

extern ai_handle g_attempt1_activations_table[1 + 2];



#define AI_ATTEMPT1_DATA_WEIGHTS_TABLE_GET() \
  (&g_attempt1_weights_table[1])

extern ai_handle g_attempt1_weights_table[1 + 2];


#endif    /* ATTEMPT1_DATA_PARAMS_H */
