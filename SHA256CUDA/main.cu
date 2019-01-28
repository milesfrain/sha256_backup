
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
__device__ size_t nonce_to_str(uint64_t nonce, unsigned char* out) {
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
//__global__ void sha256_kernel(int *out_found, const char* in_input_string, size_t in_input_string_size, size_t difficulty, uint64_t nonce_offset) {
__global__ void sha256_kernel(char* out_input_string_nonce, unsigned char* out_found_hash, int *out_found, const char* in_input_string, size_t in_input_string_size, size_t difficulty, uint64_t nonce_offset) {

	// Todo - would be more efficient if could use static input string and skip next instructions
	
	// If this is the first thread of the block, init the input string in shared memory
	char* in = (char*) &array[0];
	if (threadIdx.x == 0) {
		memcpy(in, in_input_string, in_input_string_size + 1);
	}

	__syncthreads(); // Ensure the input string has been written in SMEM

	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t nonce = idx + nonce_offset;

	// Todo - ensure this is not recalculated with float division every time

	// The first byte we can write because there is the input string at the begining	
	// Respects the memory padding of 8 bit (char).
	size_t const minArray = static_cast<size_t>(ceil((in_input_string_size + 1) / 8.f) * 8);
	
	uintptr_t sha_addr = threadIdx.x * (64) + minArray;
	uintptr_t nonce_addr = sha_addr + 32;

	// Changing to sha stored in register vs shared mem increased hash rate from 400 to 445
	//unsigned char* sha = (unsigned char*)&array[sha_addr];
	unsigned char sha[32];
	unsigned char* out = (unsigned char*)&array[nonce_addr];
	// Todo - shouldn't need to memset
	memset(out, 0, 32);

	size_t size = nonce_to_str(nonce, out);

	assert(size <= 32);

	/* Todo - could likely improve hash performance by 
	ensuring input is already appropriately padded.
	So only transform needs to be called.
	Could also check if compiler is making best use of 64-bit words.
	*/
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *)in, in_input_string_size);
	sha256_update(&ctx, out, size);
	sha256_final(&ctx, sha);
	// Todo - should ideally treat sha as 64-bit number for counting leading zeros

	uint64_t *sha64 = (uint64_t*)ctx.state;
	//if (checkZeroPadding(sha, difficulty) && atomicExch(out_found, 1) == 0) {
	if (__clzll(*sha64) >= difficulty && atomicExch(out_found, 1) == 0) {
		/*
		Slow printf, but this is fine for when a match is found.
		Possible interleaving print issue if another thread finds match
		at exact same iteration, but this is super rare.
		Otherwise other threads will wait until printing completes within conditional.
		*/

		int leading = __clzll(*sha64);

		printf("%d %s%llu ", leading, in, nonce);
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

	std::string in;
	
	/*
	std::cout << "Prefix : ";
	std::cin >> in;


	std::cout << "Nonce : ";
	std::cin >> user_nonce;

	std::cout << "Num hex zeros: ";
	std::cin >> difficulty;
	std::cout << std::endl;
	*/
	in = "aaku8856-mifr0750-";
	//user_nonce = 17050179084464;
	user_nonce = 1;
	//difficulty = 8; // finds in 15 seconds
	//difficulty = 9; // finds in a few minutes
	//difficulty = 12;
	difficulty = 28;

	const size_t input_size = in.size();

	// Input string for the device
	char *d_in = nullptr;

	// Create the input string for the device
	cudaMalloc(&d_in, input_size + 1);
	cudaMemcpy(d_in, in.c_str(), input_size + 1, cudaMemcpyHostToDevice);

	cudaMallocManaged(&g_out, input_size + 32 + 1);
	cudaMallocManaged(&g_hash_out, 32);
	cudaMallocManaged(&g_found, sizeof(int));
	*g_found = 0;

	nonce += user_nonce;

	pre_sha256();

	size_t dynamic_shared_size = (ceil((input_size + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);

	std::cout << "Shared memory is " << dynamic_shared_size << "B" << std::endl;

	for (;;) {
		sha256_kernel <<< NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size >>> (g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);
		//sha256_kernel <<< NUMBLOCKS, BLOCK_SIZE >>> (g_found, nonce);

		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error("Device error");
		}

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


	cudaFree(g_out);
	cudaFree(g_hash_out);
	cudaFree(g_found);

	cudaFree(d_in);

	cudaDeviceReset();

	system("pause");

	return 0;
}