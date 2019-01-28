
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>
#include "main.h"
#include "sha256.cuh"

#define SHOW_INTERVAL_MS 500
#define BLOCK_SIZE 256
//#define BLOCK_SIZE 512
#define SHA_PER_ITERATIONS 8'388'608L
//#define SHA_PER_ITERATIONS 1'048'576L
#define NUMBLOCKS (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE

// Wondering if allowing multiple iterations within kernel will improve performance
// Yes, boosts performance from 1.35 MH to 1.5 MH
#define KLOOPS 16
//#define KLOOPS 1

// First timestamp when program starts
static std::chrono::high_resolution_clock::time_point t1;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;

// Does the same as sprintf(char*, "%d%s", int, const char*) but a bit faster
__device__ size_t nonce_to_str(uint64_t nonce, char* out) {
	uint64_t result = nonce;
	uint8_t remainder;
	size_t nonce_size = nonce == 0 ? 1 : floor(log10((double)nonce)) + 1;
	size_t i = nonce_size;
	while (result >= 10) {
		remainder = result % 10;
		result /= 10;
		out[--i] = remainder + '0';
	}

	out[0] = result + '0';
	i = nonce_size;
	//out[i] = 0;
	return i;
}

// Wondering if constexpr would have worked here
__device__ const char hex_lookup[] = {
	'0', '1', '2', '3', '4', '5', '6', '7',
	'8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

__device__ void nonce_to_hex_str(uint64_t nonce, char* out) {
	out += 16;
	while (nonce)
	{
		*(--out) = hex_lookup[nonce & 0xF];
		nonce >>= 4; 
	}
	#if 0
	// Avoiding conditionals not always the best option
#pragma unroll 16
	for (int i = 0; i < 16; i++)
	{
		out[15 - i] = hex_lookup[nonce & 0xF];
		nonce >>= 4; 
	}
	#endif
}

__device__ void print_sha(uint64_t *sha)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 7; j >= 0; j--)
		{
			printf("%02x", ((uint8_t*)(sha + i))[j]);
		}
	}
	printf("\n");
}


//#define SINGLE
// Shared between cores on a SM, but each thread indexs to separate location
extern __shared__ char array[];
__global__ void sha256_kernel(int *out_found, int difficulty, uint64_t nonce_offset) {

	#ifdef SINGLE
	if (threadIdx.x != 0)
	{
		return;
	}
	printf("in thread 0\n");
	#endif

	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	//uint64_t nonce = KLOOPS * KLOOPS * idx + nonce_offset;
	uint64_t nonce = KLOOPS * idx + nonce_offset;
	
	SHA256_CTX ctx = 
	{
		//.data = "aaku8856-mifr0750-10000000000000000",
		// Includes initial string and terminating characters.
		// Not maintainable, but doesn't require trusting compiler
		.data = {
			0x61, 0x61, 0x6b, 0x75, 0x38, 0x38, 0x35, 0x36,
			0x2d, 0x6d, 0x69, 0x66, 0x72, 0x30, 0x37, 0x35,
			0x30, 0x2d, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30,
			0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
			0x30, 0x30, 0x30, 0x80, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x18},
		.datalen = 35,
		.bitlen = 0,
		// common state initialization value
		// Possibly faster to setup here than call sha256_init,
		// but could double-check assembly output to confirm.
		.state = { 
			0xbb67ae85,
			0x6a09e667,
			0xa54ff53a,
			0x3c6ef372,
			0x9b05688c,
			0x510e527f,
			0x5be0cd19,
			0x1f83d9ab
		}
	};

	//nonce_to_str(nonce, (char*)ctx.data + 20);
	nonce_to_hex_str(nonce, (char*)ctx.data + 19);
	uint64_t *sha64 = (uint64_t*)ctx.state;


	// unroll surprisingly worse
	//#pragma unroll

	// Could add another layer, but this doesn't improve performance
	/*
	for (int j = 0; j < KLOOPS; j++)
	{
		ctx.data[19 + 14] = hex_lookup[j];
	*/

	for (int i = 0; i < KLOOPS; i++)
	{
		#if KLOOPS != 1
		ctx.data[19 + 15] = hex_lookup[i];

		ctx.state[0] = 0xbb67ae85;
		ctx.state[1] = 0x6a09e667;
		ctx.state[2] = 0xa54ff53a;
		ctx.state[3] = 0x3c6ef372;
		ctx.state[4] = 0x9b05688c;
		ctx.state[5] = 0x510e527f;
		ctx.state[6] = 0x5be0cd19;
		ctx.state[7] = 0x1f83d9ab;
		#endif

		sha256_transform(&ctx, ctx.data);

		#ifdef SINGLE
		printf("%s ", ctx.data);
		print_sha(sha64);
		#endif

		if (__clzll(*sha64) >= difficulty && atomicExch(out_found, 1) == 0) {
			/*
			Slow printf, but this is fine for when a match is found.
			Possible interleaving print issue if another thread finds match
			at exact same iteration, but this is super rare.
			Otherwise other threads will wait until printing completes within conditional.
			*/

			int leading = __clzll(*sha64);

			printf("%d %.35s ", leading, ctx.data);
			print_sha(sha64);
		}
	}

}

void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

int main() {

	int difficulty = 1;

	int *g_found = nullptr;

	uint64_t nonce = 0;
	uint64_t user_nonce = 0;

	cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	t1 = std::chrono::high_resolution_clock::now();
	t_last_updated = std::chrono::high_resolution_clock::now();

	//user_nonce = 17050179084464;
	user_nonce = 0;
	//difficulty = 8; // finds in 15 seconds
	//difficulty = 9; // finds in a few minutes
	//difficulty = 12;
	difficulty = 28;

	cudaMallocManaged(&g_found, sizeof(int));
	*g_found = 0;

	nonce += user_nonce;

	pre_sha256();

	for (;;) {
#ifdef SINGLE
		sha256_kernel <<< 1, 32 >>> (g_found, difficulty, nonce);
#else
		sha256_kernel <<< NUMBLOCKS, BLOCK_SIZE >>> (g_found, difficulty, nonce);
#endif

		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error("Device error");
		}
#ifdef SINGLE
		break;
#endif

		//nonce += NUMBLOCKS * BLOCK_SIZE * KLOOPS * KLOOPS;
		nonce += NUMBLOCKS * BLOCK_SIZE * KLOOPS;

		if (*g_found) {
			difficulty++;
			*g_found = 0;

			#if 0
			// Enable profiling by breaking once certain difficulty reached
			if (difficulty == 34)
				break;
			#endif
		}

		#if 1
		// Print benchmarking info
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;
		if (last_show_interval.count() > SHOW_INTERVAL_MS) {
			t_last_updated = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> span = t2 - t1;
			float seconds = span.count() / 1000;
			uint64_t hashrate = static_cast<uint64_t>((nonce - user_nonce) / seconds);
			//std::cout << hashrate << " " << difficulty << " " << nonce << std::endl;
			printf("%lu %d 0x%lX\n", hashrate, difficulty, nonce);
			//std::cout << hashrate << " " << difficulty << " " << nonce << std::endl;
		}
		#endif
	}

	cudaFree(g_found);

	cudaDeviceReset();

	system("pause");

	return 0;
}