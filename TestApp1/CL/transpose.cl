#define n 16
#define N 256
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__attribute__((work_group_size_hint(n, n, 1)))
__attribute__((reqd_work_group_size(n, n, 1)))

kernel void transpose(global read_only float2 in[N][N], global write_only float2 out[N][N])
{
	local float2 l[n][n + 1];

	uint iStart = get_group_id(1) * n;
	uint jStart = get_group_id(0) * n;
	uint iL = get_local_id(1);
	uint jL = get_local_id(0);

	l[jL][iL] = in[iL + iStart][jL + jStart];
	barrier(CLK_LOCAL_MEM_FENCE);
	out[iL + jStart][jL + iStart] = l[iL][jL];
}






__attribute__((work_group_size_hint(n, n, 1)))
__attribute__((reqd_work_group_size(n, n, 1)))

kernel void transpose_image(read_only image2d_t in, global write_only float2 out[N][N])
{
	local float2 l[n][n + 1];

	uint iStart = get_group_id(1) * n;
	uint jStart = get_group_id(0) * n;
	uint iL = get_local_id(1);
	uint jL = get_local_id(0);

	float4 in_ = read_imagef(in, sampler, (int2)(iL + iStart, jL + jStart));
	l[jL][iL] = in_.xy;
	barrier(CLK_LOCAL_MEM_FENCE);
	out[iL + jStart][jL + iStart] = l[iL][jL];
}



__attribute__((work_group_size_hint(n, n, 1)))
__attribute__((reqd_work_group_size(n, n, 1)))

kernel void image_to_buffer(read_only image2d_t in, global write_only float2 out[N][N])
{
	uint i = get_global_id(1);
	uint j = get_global_id(0);
	float4 in_ = read_imagef(in, sampler, (int2)(i, j));
	out[i][j] = in_.xy;
}


