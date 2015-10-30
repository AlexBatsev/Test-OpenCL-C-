#define n 16
#define N 256


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