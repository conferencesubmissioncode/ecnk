def constant_code(mapping):
    code =  r'''
#define ONE 1.0f
#define NEG_ONE -1.0f
#define PI 3.1415926535897932384626433832795f
#define INV_NINE 0.1111111111111111111111111111111f
#define INV_PI 0.318309886183790671537767526745f
#define INV_NINE_PI 0.035367765131532296837529725193892080452102143497879f
#define NINE 9.0f
'''
    for name, value in mapping.items(): 
        code += "#define {} {}\n".format(name, value)
    return code


def pool_code(c = 'j'):
	return "max(0, (min(min(x1 + {0} + 1, x2 + {0} + 1), p) - max(max(x1 - {0}, x2 - {0}), 0))) * max(0, (min(min(y1 + {0} + 1, y2 + {0} + 1), p) - max(max(y1 - {0}, y2 - {0}), 0)))".format(c)


def conv_code():
	return r'''
#define conv1(S, T, U, V, x2, y2) \
if (x2 < 1) { \
	if (y2 < 1) { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1]; }\
	else if (y2 > p_minus_2) { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 - 1]; }\
	else { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] + S[x2 + 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] + U[x2 + 1][y2 - 1]; }\
} else if (x2 > p_minus_2) { \
	if (y2 < 1) { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1]; }\
	else if (y2 > p_minus_2) { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 - 1]; }\
	else { \
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1] + S[x2 - 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1] + U[x2 - 1][y2 - 1]; }\
} else { \
	if (y2 < 1) {\
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1]; }\
	else if (y2 > p_minus_2) {\
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 - 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 - 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 - 1]; }\
	else {\
		T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
				  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] + S[x2 + 1][y2 - 1] \
				  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1] + S[x2 - 1][y2 - 1]; \
		V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
				  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] + U[x2 + 1][y2 - 1] \
				  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1] + U[x2 - 1][y2 - 1]; }\
} 

#define conv2(S, T, U, V, x1, y1, x2, y2) \
if (x1 < 1 || x2 < 1) { \
	if (x1 > p_minus_2 || x2 > p_minus_2) { \
		if (y1 < 1 || y2 < 1) { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2]; \
				V[x2][y2] = U[x2	][y2]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1]; }\
		} else { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1]; }\
		} \
	} \
	else { \
		if (y1 < 1 || y2 < 1) { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] \
						  + S[x2 + 1][y2]; \
				V[x2][y2] = U[x2	][y2] \
						  + U[x2 + 1][y2]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1]; }\
		} else { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 - 1]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] + S[x2 + 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] + U[x2 + 1][y2 - 1]; }\
		} \
	} \
} else { \
	if (x1 > p_minus_2 || x2 > p_minus_2) { \
		if (y1 < 1 || y2 < 1) { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] \
						  + S[x2 - 1][y2]; \
				V[x2][y2] = U[x2	][y2] \
						  + U[x2 - 1][y2]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1]; }\
		} else { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 - 1]; }\
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1] + S[x2 - 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1] + U[x2 - 1][y2 - 1]; }\
		} \
	} \
	else { \
		if (y1 < 1 || y2 < 1) { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] \
						  + S[x2 + 1][y2] \
						  + S[x2 - 1][y2]; \
				V[x2][y2] = U[x2	][y2] \
						  + U[x2 + 1][y2] \
						  + U[x2 - 1][y2]; } \
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1]; } \
		} else { \
			if (y1 > p_minus_2 || y2 > p_minus_2) { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 - 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 - 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 - 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 - 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 - 1]; } \
			else { \
				T[x2][y2] = S[x2	][y2] + S[x2	][y2 + 1] + S[x2	][y2 - 1] \
						  + S[x2 + 1][y2] + S[x2 + 1][y2 + 1] + S[x2 + 1][y2 - 1] \
						  + S[x2 - 1][y2] + S[x2 - 1][y2 + 1] + S[x2 - 1][y2 - 1]; \
				V[x2][y2] = U[x2	][y2] + U[x2	][y2 + 1] + U[x2	][y2 - 1] \
						  + U[x2 + 1][y2] + U[x2 + 1][y2 + 1] + U[x2 + 1][y2 - 1] \
						  + U[x2 - 1][y2] + U[x2 - 1][y2 + 1] + U[x2 - 1][y2 - 1]; } \
		} \
	} \
}
''' 

