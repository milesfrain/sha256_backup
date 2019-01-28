
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
#define SHA_PER_ITERATIONS 8'388'608
#define NUMBLOCKS (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE

static size_t difficulty = 1;

// Output string by the device read by host
char *g_out = nullptr;
unsigned char *g_hash_out = nullptr;
int *g_found = nullptr;

static uint64_t nonce = 0;
static uint64_t user_nonce = 0;

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
	out[i] = 0;
	return i;
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

// Shared between cores on a SM, but each thread indexs to separate location
extern __shared__ char array[];
__global__ void sha256_kernel(int *out_found, size_t difficulty, uint64_t nonce_offset) {
//__global__ void sha256_kernel(char* out_input_string_nonce, unsigned char* out_found_hash, int *out_found, const char* in_input_string, size_t in_input_string_size, size_t difficulty, uint64_t nonce_offset) {

	#if 0
	if (threadIdx.x != 0)
	{
		return;
	}
	printf("in thread 0\n");
	#endif

	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t nonce = idx + nonce_offset;
	
	const char* prefix = "aaku8856-mifr0750-";
	// There should be a way to get string literal length at compile time.
	//const size_t prefix_len = strlen(prefix); // "host" code cannot run on device
	const size_t prefix_len = 19; // hardcoding with terminating char included
	char nonce_str[30];

	size_t nonce_str_len = nonce_to_str(nonce, nonce_str);

	/* Todo - could likely improve hash performance by 
	ensuring input is already appropriately padded.
	So only transform needs to be called.
	Could also check if compiler is making best use of 64-bit words.
	*/
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char*)prefix, prefix_len - 1);
	sha256_update(&ctx, (unsigned char*)nonce_str, nonce_str_len);
	//sha256_update(&ctx, (unsigned char*)"abc123", 6);
	//sha256_update(&ctx, (unsigned char*)"aaku8856-mifr0750-1", 18);
	sha256_final(&ctx);
	// Todo - should ideally treat sha as 64-bit number for counting leading zeros

	uint64_t *sha64 = (uint64_t*)ctx.state;

	//printf("%s%llu ", prefix, nonce);
	//print_sha(sha64);

	//if (checkZeroPadding(sha, difficulty) && atomicExch(out_found, 1) == 0) {
	if (__clzll(*sha64) >= difficulty && atomicExch(out_found, 1) == 0) {
		/*
		Slow printf, but this is fine for when a match is found.
		Possible interleaving print issue if another thread finds match
		at exact same iteration, but this is super rare.
		Otherwise other threads will wait until printing completes within conditional.
		*/

		int leading = __clzll(*sha64);

		printf("%d %s%llu ", leading, prefix, nonce);
		print_sha(sha64);
	}
}

void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

int main() {

	cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	t1 = std::chrono::high_resolution_clock::now();
	t_last_updated = std::chrono::high_resolution_clock::now();

	//user_nonce = 17050179084464;
	user_nonce = 1;
	//difficulty = 8; // finds in 15 seconds
	//difficulty = 9; // finds in a few minutes
	//difficulty = 12;
	difficulty = 28;

	cudaMallocManaged(&g_found, sizeof(int));
	*g_found = 0;

	nonce += user_nonce;

	pre_sha256();

	for (;;) {
		//sha256_kernel <<< NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size >>> (g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);
		sha256_kernel <<< NUMBLOCKS, BLOCK_SIZE >>> (g_found, difficulty, nonce);
		//sha256_kernel <<< 1, 32 >>> (g_found, difficulty, nonce);

		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error("Device error");
		}
		//break;

		nonce += NUMBLOCKS * BLOCK_SIZE;

		if (*g_found) {
			difficulty++;
			*g_found = 0;
		}

		// Print benchmarking info
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;
		if (last_show_interval.count() > SHOW_INTERVAL_MS) {
			t_last_updated = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> span = t2 - t1;
			float ratio = span.count() / 1000;
			uint64_t hashrate = static_cast<uint64_t>((nonce - user_nonce) / ratio);
			std::cout << hashrate << " " << difficulty << " " << nonce << std::endl;
		}
	}

	cudaFree(g_found);

	cudaDeviceReset();

	system("pause");

	return 0;
}