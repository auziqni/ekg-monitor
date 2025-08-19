/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#define BLOCK_SAMPLES 128
#define NUM_CHANNELS   9
#define ADC_BUF_LEN   (BLOCK_SAMPLES * NUM_CHANNELS)

static volatile uint16_t adc_buf[ADC_BUF_LEN];
static volatile uint8_t dma_half_ready = 0;
static volatile uint8_t dma_full_ready = 0;
static uint32_t sample_idx = 0; // +1 per sample-pair (2 ms)

// Debug SPS variables
volatile uint32_t target_sps = 400; // configurable target sampling rate (sample-pair per second)
static volatile uint32_t sps_counter = 0; // counts sent sample-pairs in the last 1s window
static uint32_t sps_last_ms = 0;          // timestamp of last SPS report

// Debug toggle
volatile bool debug_enabled = true; // set true to enable periodic debug info

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_ADC1_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buf, ADC_BUF_LEN);
  HAL_TIM_Base_Start(&htim2);

  // Initialize SPS timing window
  sps_last_ms = HAL_GetTick();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    char line[64];

if (dma_half_ready) {
  dma_half_ready = 0;
  for (uint32_t i = 0; i < ADC_BUF_LEN/2; i += NUM_CHANNELS) {
    uint16_t v_pb1 = adc_buf[i + 0];
    uint16_t v_pb0 = adc_buf[i + 1];
    uint16_t v_a7  = adc_buf[i + 2];
    uint16_t v_a6  = adc_buf[i + 3];
    uint16_t v_a5  = adc_buf[i + 4];
    uint16_t v_a4  = adc_buf[i + 5];
    uint16_t v_a3  = adc_buf[i + 6];
    uint16_t v_a2  = adc_buf[i + 7];
    uint16_t v_a1  = adc_buf[i + 8];
    uint32_t t_ms = sample_idx * 2U;
    sample_idx++;
    int n = snprintf(line, sizeof(line),
                     "%06lX,%03X,%03X,%03X,%03X,%03X,%03X,%03X,%03X,%03X;",
                     (unsigned long)t_ms,
                     (unsigned int)v_pb1, (unsigned int)v_pb0,
                     (unsigned int)v_a7,  (unsigned int)v_a6,
                     (unsigned int)v_a5,  (unsigned int)v_a4,
                     (unsigned int)v_a3,  (unsigned int)v_a2,
                     (unsigned int)v_a1);
    if (n > 0) HAL_UART_Transmit(&huart1, (uint8_t*)line, (uint16_t)n, 100);
    sps_counter++;
  }
}

if (dma_full_ready) {
  dma_full_ready = 0;
  for (uint32_t i = ADC_BUF_LEN/2; i < ADC_BUF_LEN; i += NUM_CHANNELS) {
    uint16_t v_pb1 = adc_buf[i + 0];
    uint16_t v_pb0 = adc_buf[i + 1];
    uint16_t v_a7  = adc_buf[i + 2];
    uint16_t v_a6  = adc_buf[i + 3];
    uint16_t v_a5  = adc_buf[i + 4];
    uint16_t v_a4  = adc_buf[i + 5];
    uint16_t v_a3  = adc_buf[i + 6];
    uint16_t v_a2  = adc_buf[i + 7]; 
    uint16_t v_a1  = adc_buf[i + 8];
    uint32_t t_ms = sample_idx * 2U;
    sample_idx++;
    int n = snprintf(line, sizeof(line),
                     "%06lX,%03X,%03X,%03X,%03X,%03X,%03X,%03X,%03X,%03X;",
                     (unsigned long)t_ms,
                     (unsigned int)v_pb1, (unsigned int)v_pb0,
                     (unsigned int)v_a7,  (unsigned int)v_a6,
                     (unsigned int)v_a5,  (unsigned int)v_a4,
                     (unsigned int)v_a3,  (unsigned int)v_a2,
                     (unsigned int)v_a1);
    if (n > 0) HAL_UART_Transmit(&huart1, (uint8_t*)line, (uint16_t)n, 100);
    sps_counter++;
  }

  HAL_GPIO_TogglePin(LED_BUILTIN_GPIO_Port, LED_BUILTIN_Pin);
}

// Periodic SPS debug report every 1 second
uint32_t now = HAL_GetTick();
if ((now - sps_last_ms) >= 1000U) {
  uint32_t sps = sps_counter;
  sps_counter = 0;
  sps_last_ms += 1000U;

  if (debug_enabled) {
    uint32_t eff = (target_sps > 0U) ? (sps * 100U) / target_sps : 0U;
    int n = snprintf(line, sizeof(line),
                     "\n# debug_info -> sps: %lu, target: %lu, efficiency: %lu%%\n",
                     sps, target_sps, eff);
    if (n > 0) {
      HAL_UART_Transmit(&huart1, (uint8_t*)line, (uint16_t)n, 100);
    }
  }
}
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc) {
  if (hadc->Instance == ADC1) dma_half_ready = 1;
}
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
  if (hadc->Instance == ADC1) dma_full_ready = 1;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
