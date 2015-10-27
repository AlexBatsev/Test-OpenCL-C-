#define n 4
#define N 256
#define N0 256
#define N1 64
#define N2 16
#define N3 4
#define N_4_0 0u
#define N_4_1 64u
#define N_4_2 64u * 2
#define N_4_3 64u * 3
#define N_3_0 0u
#define N_3_1 16u
#define N_3_2 16u * 2
#define N_3_3 16u * 3

__attribute__((always_inline))
float2 iMult(float2 x)
{
	return (float2)(-x.y, x.x);
}

__attribute__((always_inline))
float2 twiddle(uint ik, uint N)
{
	float angle = (M_PI * 2.0f * ik) / N;
	return (float2)(native_cos(angle), native_sin(angle));

}

kernel void fft256(global read_only float2 in[N], global write_only float2 out[N]) 
{
	size_t j = get_local_id(0);
	float2 x0 = in[j + N_4_0];
	float2 x2 = in[j + N_4_2];
	float2 x1 = in[j + N_4_1];
	float2 x3 = in[j + N_4_3];

	float2 a0 = x0 + x2;
	float2 b0 = x0 - x2;
	float2 a1 = x1 + x3;
	float2 b1 = iMult(x1 - x3);

	x0 = (a0 + a1) * twiddle(j * 0, N0);
	x1 = (b0 + b1) * twiddle(j * 1, N0);
	x2 = (a0 - a1) * twiddle(j * 2, N0);
	x3 = (b0 - b1) * twiddle(j * 3, N0);

	local float2 loc[N_4_1 * 5];
	local float2* loc_in = loc + (j >> 2) * 5 + j & 3u;

	uint indOut = j;
	uint stride = 0u;
	local float2* loc_out = 
	loc[indOut * 5 + 0] = z0;
	loc[indOut * 5 + 1] = z1;
	loc[indOut * 5 + 2] = z1;
	loc[indOut * 5 + 3] = z1;

	//////////    stage 1 ////////////
	j >>= 2;
	x0 = *(loc_in + N_3_0 * 5);
	x1 = *(loc_in + N_3_1 * 5);
	x2 = *(loc_in + N_3_2 * 5);
	x3 = *(loc_in + N_3_3 * 5);

	float2 a0 = x0 + x2;
	float2 b0 = x0 - x2;
	float2 a1 = x1 + x3;
	float2 b1 = iMult(x1 - x3);

	x0 = (a0 + a1) * twiddle(j * 0, N1);
	x1 = (b0 + b1) * twiddle(j * 1, N1);
	x2 = (a0 - a1) * twiddle(j * 2, N1);
	x3 = (b0 - b1) * twiddle(j * 3, N1);
	
	indOut = 

}