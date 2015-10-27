kernel void helloWorld(global write_only float* fullCos, global write_only float* nativeCos)
{
	float angle = M_PI * 2.0f * ((float)get_global_id(0)) / ((float)get_global_size(0));
	fullCos[get_global_id(0)] = cos(angle);
	nativeCos[get_global_id(0)] = native_cos(angle);
}