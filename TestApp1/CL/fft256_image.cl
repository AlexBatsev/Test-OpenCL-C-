#define n 4
#define N4 256
#define N3 64
#define N2 16
#define N1 4
#define N0 1
#define loc_in_ind_0 16u * 5u * 0u
#define loc_in_ind_1 16u * 5u * 1u
#define loc_in_ind_2 16u * 5u * 2u
#define loc_in_ind_3 16u * 5u * 3u
#define s0 0
#define s1 2
#define s2 4
#define s3 6
#define stride0 1
#define stride1 5
#define stride2 20
#define rem0 0
#define rem1 3
#define rem2 15
#define rem3 63

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


__attribute__((work_group_size_hint(N3, 1, 1)))
__attribute__((reqd_work_group_size(N3, 1, 1)))

kernel void fft256(global read_only float2 in[][N4], write_only image2d_t out)
{
	size_t j = get_local_id(0);
	uint j_hi = get_global_id(1); //temporary usage of j_hi as "i"
	local float2 loc[N3 * 5];

	//////////    stage 0 ////////////
	float2 x0 = in[j_hi][j + 64 * 0];
	float2 x2 = in[j_hi][j + 64 * 2];
	float2 x1 = in[j_hi][j + 64 * 1];
	float2 x3 = in[j_hi][j + 64 * 3];

	float2 a0 = x0 + x2;
	float2 b0 = x0 - x2;
	float2 a1 = x1 + x3;
	float2 b1 = iMult(x1 - x3);

	float2 twid1 = twiddle(j, N4);
	float2 twid2 = complex_mult(twid1, twid1);

	x0 = a0 + a1;
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));

	local float2* loc_out = loc + j * 5;
	loc_out[0 * stride0] = x0;
	loc_out[1 * stride0] = x1;
	loc_out[2 * stride0] = x2;
	loc_out[3 * stride0] = x3;

	//////////    stage 1 ////////////
	local float2* loc_in = loc + loc_ind(j);

	barrier(CLK_LOCAL_MEM_FENCE);
	x0 = loc_in[loc_in_ind_0];
	x1 = loc_in[loc_in_ind_1];
	x2 = loc_in[loc_in_ind_2];
	x3 = loc_in[loc_in_ind_3];

	a0 = x0 + x2;
	b0 = x0 - x2;
	a1 = x1 + x3;
	b1 = iMult(x1 - x3);

	j_hi = j >> s1;

	twid1 = twiddle(j_hi, N3);
	twid2 = complex_mult(twid1, twid1);

	x0 = (a0 + a1);
	x1 = complex_mult(b0 + b1, twid1);
	x2 = complex_mult(a0 - a1, twid2);
	x3 = complex_mult(b0 - b1, complex_mult(twid1, twid2));

	loc_out = loc + loc_ind((j_hi << (s1 + 2)) + (j & rem1));

	loc_out[0 * stride1] = x0;
	loc_out[1 * stride1] = x1;
	loc_out[2 * stride1] = x2;
	loc_out[3 * stride1] = x3;

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

	loc_out = loc + loc_ind((j_hi << (s2 + 2)) + (j & rem2));

	loc_out[0 * stride2] = x0;
	loc_out[1 * stride2] = x1;
	loc_out[2 * stride2] = x2;
	loc_out[3 * stride2] = x3;

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

	x0 = a0 + a1;
	x1 = b0 + b1;
	x2 = a0 - a1;
	x3 = b0 - b1;

	j_hi = get_global_id(1); //temporary usage of j_hi as "i"

	write_imagef(out, (int2)(j_hi, j + N3 * 0), (float4)(x0, 0.0f, 0.0f));
	write_imagef(out, (int2)(j_hi, j + N3 * 1), (float4)(x1, 0.0f, 0.0f));
	write_imagef(out, (int2)(j_hi, j + N3 * 2), (float4)(x2, 0.0f, 0.0f));
	write_imagef(out, (int2)(j_hi, j + N3 * 3), (float4)(x3, 0.0f, 0.0f));
}