#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "arm_neon.h"


#define TEST_M 256
#define TEST_K 256
#define TEST_N 256

void gemm_fp16(float16_t *a, int sa, float16_t *b, int sb, float16_t *c, int sc)
{
	float16x8_t vb[8];
	for(int y = 0; y < 8; y++){
		vb[y] = vld1q_f16(b + sb*y);
	}

	for(int y = 0; y < 8; y++){
		float16x8_t vc = vld1q_f16(c + sc*y);
		for(int x = 0; x < 8; x++){
			vc = vaddq_f16(vc, vmulq_n_f16(vb[x], *(a+sa*y+x)));
		}
		vst1q_f16(c + sc*y, vc);
	}
}


int main(void)
{
	float16_t* ma = (float16_t*)malloc(sizeof(float16_t)*TEST_K*TEST_M);
	float16_t* mb = (float16_t*)malloc(sizeof(float16_t)*TEST_N*TEST_K);
	float16_t* mc = (float16_t*)malloc(sizeof(float16_t)*TEST_N*TEST_M);
	float16_t* chk = (float16_t*)malloc(sizeof(float16_t)*TEST_N*TEST_M);

	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_K; x++){
			ma[y*TEST_K + x] = (float16_t)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_K; y++){
		for(int x = 0; x < TEST_N; x++){
			mb[y*TEST_N + x] = (float16_t)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_N; x++){
			mc[y*TEST_N + x] = (float16_t)0.0;
			chk[y*TEST_N + x] = (float16_t)0.0;
		}
	}

	struct timeval stime, etime;
	gettimeofday(&stime, NULL);
	for(int m = 0; m < TEST_M; m+=8){
		for(int n = 0; n < TEST_N; n+=8){
			for(int k = 0; k < TEST_K; k+=8){
				gemm_fp16(
							ma + m*TEST_K + k, TEST_K,
							mb + k*TEST_N + n, TEST_N,
							mc + m*TEST_N + n, TEST_N
						);
			}
		}
	}
	gettimeofday(&etime, NULL);
	printf("FP16 NEON: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	gettimeofday(&stime, NULL);
	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float16_t val = 0.0;
			for(int k = 0; k < TEST_K; k++){
				val += ma[m*TEST_K + k]*mb[k*TEST_N+n];
			}
			chk[m*TEST_N + n] = val;
		}
	}
	gettimeofday(&etime, NULL);
	printf("NAIVE: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float16_t val = chk[m*TEST_N + n] - mc[m*TEST_N + n];
			if( val > 0.2 || val < -0.2){
				printf("(%d,%d), %f %f\n", m, n, chk[m*TEST_N + n], mc[m*TEST_N + n]);
			}
		}
	}

	printf("DONE!\n");
	return 0;
}