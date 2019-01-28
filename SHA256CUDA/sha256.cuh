#ifndef SHA256_H
#define SHA256_H

#include <cstdint>


/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        printf("GPU: cudaError %d (%s)\n", err, cudaGetErrorString(err)); \
}
/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef uint32_t  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;

__constant__ WORD dev_k[64];

static const WORD host_k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DECLARATIONS **********************/
#if 0
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void sha256_final(SHA256_CTX *ctx);
#endif

__device__ void mycpy12(uint32_t *d, const uint32_t *s) {
#pragma unroll 3
	for (int k = 0; k < 3; k++) d[k] = s[k];
}

__device__ void mycpy16(uint32_t *d, const uint32_t *s) {
#pragma unroll 4
	for (int k = 0; k < 4; k++) d[k] = s[k];
}

__device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
	for (int k = 0; k < 8; k++) d[k] = s[k];
}

__device__ void mycpy44(uint32_t *d, const uint32_t *s) {
#pragma unroll 11
	for (int k = 0; k < 11; k++) d[k] = s[k];
}

__device__ void mycpy48(uint32_t *d, const uint32_t *s) {
#pragma unroll 12
	for (int k = 0; k < 12; k++) d[k] = s[k];
}

__device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
	for (int k = 0; k < 16; k++) d[k] = s[k];
}

__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

#pragma unroll 16
	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);

#pragma unroll 64
	for (; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	// State indicies reordered (sequential pairs swapped) to allow 64-bit clz

	a = ctx->state[1];
	b = ctx->state[0];
	c = ctx->state[3];
	d = ctx->state[2];
	e = ctx->state[5];
	f = ctx->state[4];
	g = ctx->state[7];
	h = ctx->state[6];

#pragma unroll 64
	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e, f, g) + dev_k[i] + m[i];
		t2 = EP0(a) + MAJ(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[1] += a;
	ctx->state[0] += b;
	ctx->state[3] += c;
	ctx->state[2] += d;
	ctx->state[5] += e;
	ctx->state[4] += f;
	ctx->state[7] += g;
	ctx->state[6] += h;
}

#if 0
__device__ void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[1] = 0x6a09e667;
	ctx->state[0] = 0xbb67ae85;
	ctx->state[3] = 0x3c6ef372;
	ctx->state[2] = 0xa54ff53a;
	ctx->state[5] = 0x510e527f;
	ctx->state[4] = 0x9b05688c;
	ctx->state[7] = 0x1f83d9ab;
	ctx->state[6] = 0x5be0cd19;
}

// Assuming single block with all fields populated
__device__ void sha256_single_block_complete(SHA256_CTX *ctx)
{

}


__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	//printf("sha_update len %d, %s\n", len, data);

	// Transforms every 64 bytes (512 bits)

	// for each byte in message
	for (i = 0; i < len; ++i) {
		// ctx->data == message 512 bit chunk
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void sha256_final(SHA256_CTX *ctx)
{
	WORD i;

	i = ctx->datalen;
	ctx->data[i++] = 0x80;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	// Todo, can the below just be a 64-bit assignment?
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;

	#if 0
	for (i = 0; i < 64; i++)
	{
		printf("0x%02x, ", ctx->data[i]);
	}
	printf("\n");
	#endif
	
	sha256_transform(ctx, ctx->data);
}
#endif

#endif   // SHA256_H