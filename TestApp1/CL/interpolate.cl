#include "CommonDefines.h"

#define N_GRID_NODES_MINUS_1 255


typedef struct
{
	int i;
	float t;
} InterpData;



__attribute__((always_inline))
inline void getIndices(global read_only float2* points, InterpData out[2])
{
	//TODO: inline this var
	float2 fInd = points[get_global_id(0)];
	int2 i = convert_int2_rtn(fInd);
	float2 t = fInd - convert_float2(i);
	out[0] = (InterpData){ i.x & N_GRID_NODES_MINUS_1, t.x };
	out[1] = (InterpData){ i.y & N_GRID_NODES_MINUS_1, t.y };
}


inline float2 linterpG2(__global float2* val, int blockSize, float t)
{
	return *val * (1.0f - t) + *(val + blockSize) * t;
}

inline float2 linterpP2(float2* val, int blockSize, float t)
{
	return *val * (1.0f - t) + *(val + blockSize) * t;
}

inline float2 bilinterpG2(__global float2* val, int blockSize0, int blockSize1, float t0, float t1)
{
	float2 v[2] = { linterpG2(val, blockSize1, t1), linterpG2(val + blockSize0, blockSize1, t1) };
	return linterpP2(v, 1, t0);
}

#define stepPeriodic(ind, period) (ind + 1 >= period ? -ind : 1)




__attribute__((work_group_size_hint(N_INTERPOLATE_POINTS_WG, 1, 1)))
__attribute__((reqd_work_group_size(N_INTERPOLATE_POINTS_WG, 1, 1)))

kernel void interpolate_no_image(global read_only float2 gridsBuff[N_GRID_NODES][N_GRID_NODES], global read_only float2* points, global write_only float2* offsets, uint nPoints)
{
	if (get_global_id(0) >= nPoints)
		return;
	InterpData interp[2];
	getIndices(points, interp);

	float2 result = bilinterpG2(gridsBuff[interp[0].i] + interp[1].i, stepPeriodic(interp[0].i, N_GRID_NODES) * N_GRID_NODES, stepPeriodic(interp[1].i, N_GRID_NODES), interp[0].t, interp[1].t);

	offsets[get_global_id(0)] = result;
}

