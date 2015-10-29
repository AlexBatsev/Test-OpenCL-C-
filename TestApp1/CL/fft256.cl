#define n 4
#define N0 256
#define N1 64
#define N2 16
#define N3 4
#define NS_0 1
#define NS_1 4
#define NS_2 16
#define NS_3 64
#define loc_in_ind_0 16u * 5u * 0u
#define loc_in_ind_1 16u * 5u * 1u
#define loc_in_ind_2 16u * 5u * 2u
#define loc_in_ind_3 16u * 5u * 3u
#define s0 0
#define s1 2
#define s2 4
#define s3 6

__attribute__((always_inline))
inline float2 iMult(float2 x)
{
	return (float2)(-x.y, x.x);
}

__attribute__((always_inline))
inline float2 twiddle(uint ik, uint N)
{
	float angle = (M_PI * 2.0f * ik) / N;
	return (float2)(native_cos(angle), native_sin(angle));

}

__attribute__((always_inline))
inline uint loc_ind(uint x)
{ 
	return (x >> 2) * 5 + (x & 3u);
}

__attribute__((always_inline))
inline float2 complex_mult(float2 a, float2 b)
{ 
	return (float2)(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}



kernel void fft256(global read_only float2 in[N0], global write_only float2 out[N0]) 
{
	size_t j = get_local_id(0);

	//////////    stage 0 ////////////
	float2 x0 = in[j + 64 * 0];
	float2 x2 = in[j + 64 * 2];
	float2 x1 = in[j + 64 * 1];
	float2 x3 = in[j + 64 * 3];

	float2 a0 = x0 + x2;
	float2 b0 = x0 - x2;
	float2 a1 = x1 + x3;
	float2 b1 = iMult(x1 - x3);
	
	float2 twid1 = twiddle(j, N0);
	float2 twid2 = complex_mult(twid1, twid1);

	x0 = a0 + a1;
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));

	local float2 loc[N1 * 5];

	uint stride = 1u;
	local float2* loc_out = loc + j * 5;
	loc_out[0 * stride] = x0;
	loc_out[1 * stride] = x1;
	loc_out[2 * stride] = x2;
	loc_out[3 * stride] = x3;

	local float2* loc_in = loc_out;

	//////////    stage 1 ////////////
	barrier(CLK_LOCAL_MEM_FENCE);
	x0 = loc_in[loc_in_ind_0];
	x1 = loc_in[loc_in_ind_1];
	x2 = loc_in[loc_in_ind_2];
	x3 = loc_in[loc_in_ind_3];

	a0 = x0 + x2;
	b0 = x0 - x2;
	a1 = x1 + x3;
	b1 = iMult(x1 - x3);

	uint j_hi = j >> s1;

	twid1 = twiddle(j_hi, N1);
	twid2 = complex_mult(twid1, twid1);

	x0 = (a0 + a1);
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));
	
	uint j_lo = j & (NS_1 - 1);
	loc_out = loc + loc_ind((j_hi << (s1 + 2)) + j_lo);
	stride = loc_ind(NS_1);
	
	loc_out[0 * stride] = x0;
	loc_out[1 * stride] = x1;
	loc_out[2 * stride] = x2;
	loc_out[3 * stride] = x3;

	//////////    stage 2 ////////////
	barrier(CLK_LOCAL_MEM_FENCE);
	x0 = loc_in[loc_in_ind_0];
	x1 = loc_in[loc_in_ind_1];
	x2 = loc_in[loc_in_ind_2];
	x3 = loc_in[loc_in_ind_3];

	a0 = x0 + x2;
	b0 = x0 - x2;
	a1 = x1 + x3;
	b1 = iMult(x1 - x3);

	j_hi = j >> s2;

	twid1 = twiddle(j_hi, N2);
	twid2 = complex_mult(twid1, twid1);

	x0 = (a0 + a1);
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));
	
	j_lo = j & (NS_2 - 1);
	loc_out = loc + loc_ind((j_hi << (s2 + 2)) + j_lo);
	stride = loc_ind(NS_2);
	
	loc_out[0 * stride] = x0;
	loc_out[1 * stride] = x1;
	loc_out[2 * stride] = x2;
	loc_out[3 * stride] = x3;

	//////////    stage 3 ////////////
	barrier(CLK_LOCAL_MEM_FENCE);
	x0 = loc_in[loc_in_ind_0];
	x1 = loc_in[loc_in_ind_1];
	x2 = loc_in[loc_in_ind_2];
	x3 = loc_in[loc_in_ind_3];

	a0 = x0 + x2;
	b0 = x0 - x2;
	a1 = x1 + x3;
	b1 = iMult(x1 - x3);

	j_hi = j >> s3;

	twid1 = twiddle(j_hi, N3);
	twid2 = complex_mult(twid1, twid1);

	x0 = (a0 + a1);
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));
	
/*
	j_lo = j & (NS_3 - 1);
	loc_out = loc + loc_ind((j_hi << (s3 + 2)) + j_lo);
	stride = loc_ind(NS_3);
	
	loc_out[0 * stride] = x0;
	loc_out[1 * stride] = x1;
	loc_out[2 * stride] = x2;
	loc_out[3 * stride] = x3;
*/
	out[j + N1 * 0] = x0;
	out[j + N1 * 1] = x1;
	out[j + N1 * 2] = x2;
	out[j + N1 * 3] = x3;
}