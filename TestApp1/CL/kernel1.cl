kernel void testCosinus(global write_only float* fullCos, global write_only float* nativeCos)
{
	float angle = M_PI * 2.0f * ((float)get_global_id(0)) / ((float)get_global_size(0));
	fullCos[get_global_id(0)] = cos(angle);
	nativeCos[get_global_id(0)] = native_cos(angle);
}

kernel void testNormalize(global float2* data)
{
	float2 d = data[get_global_id(0)];
	d = normalize(d);
	data[get_global_id(0)] = d;
}